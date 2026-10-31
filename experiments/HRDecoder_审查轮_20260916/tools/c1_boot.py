#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C1 融合方案显著性：图像级配对 bootstrap（与 e1_gate / collect_round 同一口径）。

统计量：IoU = 先累计 TP/FP/FN 再求比；空并集记 0；
        2000 次，seed 20260917，两臂共用同一组重采样索引；95% 百分位区间。
做两件事：
  1) 融合方案的**自对照**（同一文件内 head vs base0）→ 回答"有没有把 BASE_0 抬起来"
  2) 融合方案 vs 各**单臂**（配对文件内 head vs head）→ 回答"融合是否真的比单臂好"
"""
import argparse
import json

import numpy as np

B = 2000
SEED = 20260917
BASE0_NON_MA = {"EX": 0.457705, "HE": 0.437552, "SE": 0.551806}


def iou(t):
    den = t[0] + t[1] + t[2]
    return t[0] / den if den else 0.0


def agg(rows, idx, tag):
    tp = fp = fn = 0
    for i in idx:
        a, b, c = rows[i][tag]
        tp += a; fp += b; fn += c
    return tp, fp, fn


def miou4(ma, non_ma=None):
    n = non_ma or BASE0_NON_MA
    return float(np.mean([n["EX"], n["HE"], n["SE"], ma]))


def boot_paired(ra, rb):
    """Δ = IoU(rb.head) − IoU(ra.head)，共用重采样索引。"""
    n = min(len(ra), len(rb))
    ia, ib = iou(agg(ra, range(n), "head")), iou(agg(rb, range(n), "head"))
    rng = np.random.RandomState(SEED)
    dl = np.empty(B)
    for b in range(B):
        idx = rng.randint(0, n, n)
        dl[b] = iou(agg(rb, idx, "head")) - iou(agg(ra, idx, "head"))
    lo, hi = np.percentile(dl, [2.5, 97.5])
    return {"iou_a": ia, "iou_b": ib, "delta": ib - ia, "ci95": [float(lo), float(hi)],
            "crosses_zero": bool(lo <= 0 <= hi), "q05_one_sided": float(np.percentile(dl, 5.0)),
            "n": n}


def boot_self(rows):
    n = len(rows)
    ia, ib = iou(agg(rows, range(n), "base0")), iou(agg(rows, range(n), "head"))
    rng = np.random.RandomState(SEED)
    dl = np.empty(B)
    for b in range(B):
        idx = rng.randint(0, n, n)
        dl[b] = iou(agg(rows, idx, "head")) - iou(agg(rows, idx, "base0"))
    lo, hi = np.percentile(dl, [2.5, 97.5])
    return {"iou_base0": ia, "iou_head": ib, "delta": ib - ia, "ci95": [float(lo), float(hi)],
            "crosses_zero": bool(lo <= 0 <= hi), "q05_one_sided": float(np.percentile(dl, 5.0)),
            "n": n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--winner", required=True, help="C1 胜出方案的逐图计数 json")
    ap.add_argument("--ref", action="append", default=[],
                    help="名=逐图计数路径；用其 head 计数与 winner 做配对")
    ap.add_argument("--non-ma", default="",
                    help="覆盖非 MA 三类常数（例 EX=0.5386;HE=0.4451;SE=0.4982）。"
                         "test 上必须覆盖，否则 mIoU 会用 val 常数算错（IoU/Δ 不受影响）。")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    non_ma = dict(BASE0_NON_MA)
    if a.non_ma:
        for kv in a.non_ma.split(";"):
            kv = kv.strip()
            if kv:
                k, v = kv.split("=", 1)
                non_ma[k.strip()] = float(v)

    wj = json.load(open(a.winner))
    rw = wj["per_image"]
    out = {"winner": wj.get("variant"), "n_rows": len(rw), "seed": SEED, "B": B,
           "non_ma_baseline": non_ma,
           "note": "Δ = IoU(后者) − IoU(前者)；跨零=是 ⇒ 与 0 无法区分",
           "self_contrast": None, "vs_ref": {}}

    sc = boot_self(rw)
    sc["mIoU_base0"] = miou4(sc["iou_base0"], non_ma)
    sc["mIoU_head"] = miou4(sc["iou_head"], non_ma)
    out["self_contrast"] = sc
    print("=" * 78)
    print("C1 胜出方案 = %s" % wj.get("variant"))
    print("  自对照 base0=%.6f → head=%.6f  Δ=%+.6f  CI=[%+.6f, %+.6f]  跨零=%s"
          % (sc["iou_base0"], sc["iou_head"], sc["delta"], sc["ci95"][0], sc["ci95"][1],
             sc["crosses_zero"]))
    print("  mIoU %.6f → %.6f（ΔmIoU %+.6f）"
          % (sc["mIoU_base0"], sc["mIoU_head"], sc["mIoU_head"] - sc["mIoU_base0"]))

    print("-" * 78)
    print("C1 胜出方案 vs 单臂（配对，Δ = IoU(单臂) − IoU(融合)）")
    for spec in a.ref:
        name, path = spec.split("=", 1)
        rj = json.load(open(path))
        rr = rj["per_image"]
        if len(rr) != len(rw):
            print("  %-10s 行数不同 %d vs %d → 跳过" % (name, len(rr), len(rw)))
            continue
        b0r = iou(agg(rr, range(len(rr)), "base0"))
        b0w = iou(agg(rw, range(len(rw)), "base0"))
        if abs(b0r - b0w) > 1e-12:
            print("  %-10s base0 不等 %.6f vs %.6f → 跳过（val 集/tiling 不一致）" % (name, b0r, b0w))
            continue
        r = boot_paired(rr, rw)
        r["mIoU_ref"] = miou4(r["iou_a"], non_ma)
        r["mIoU_winner"] = miou4(r["iou_b"], non_ma)
        out["vs_ref"][name] = r
        print("  %-12s 单臂=%.6f  融合=%.6f  Δ=%+.6f  CI=[%+.6f,%+.6f]  跨零=%s  q05=%+.6f"
              % (name, r["iou_a"], r["iou_b"], r["delta"], r["ci95"][0], r["ci95"][1],
                 r["crosses_zero"], r["q05_one_sided"]))
        print("               ΔmIoU=%+.6f（单臂 %.6f → 融合 %.6f）"
              % (r["mIoU_winner"] - r["mIoU_ref"], r["mIoU_ref"], r["mIoU_winner"]))

    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1, ensure_ascii=False)
        print("写出:", a.out)


if __name__ == "__main__":
    main()
