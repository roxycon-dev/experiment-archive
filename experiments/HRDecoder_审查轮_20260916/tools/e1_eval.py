#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E1 对比：续训臂（E1）对原始臂（T0/T1 @8000）的配对显著性。

为什么不直接用 collect_round.py：
  E1 是"从 head_T0/T1 @8000 续训 4000 步"，其终点是 **it4000**，
  而要对比的原始臂终点是 **it8000** —— iter 号天然不同。
  collect_round.py 的配对自检要求两臂 iter 相同（那是为跨臂同阶段对比设的），
  会把这条合法对比判为 invalid。故此处单独实现，并把"允许 iter 不同"的理由写进产物。

统计口径与 e1_gate.py / collect_round.py **完全一致**（不新造）：
  图像级配对 bootstrap，2000 次，seed 20260917，两臂共用重采样索引，
  先累计 TP/FP/FN 再求 IoU，95% 百分位区间，空并集记 0。

仍然强制的不变量（这三条与 iter 无关，必须成立）：
  1. 两臂逐图行数相同（同一 val 集）
  2. 两臂 base0 MA-IoU 逐位相同（同一 tiling / 同一评估口径）
  3. seed 相同
"""
import argparse
import json
import os

import numpy as np

B = 2000
SEED = 20260917
CLS = ["EX", "HE", "SE", "MA"]
# 非 MA 三类恒取 BASE_0（契约：小头只改 MA 通道）→ 用于合成 4 类 mIoU
BASE0_NON_MA = {"EX": 0.457705, "HE": 0.437552, "SE": 0.551806}


def iou(t):
    den = t[0] + t[1] + t[2]
    return t[0] / den if den else 0.0


def load_counts(p):
    d = json.load(open(p))
    return d, d["per_image"]


def agg(rows, idx, tag):
    tp = fp = fn = 0
    for i in idx:
        a, b, c = rows[i][tag]
        tp += a; fp += b; fn += c
    return tp, fp, fn


def paired_bootstrap(ra, rb, tag="head"):
    n = min(len(ra), len(rb))
    ia, ib = iou(agg(ra, range(n), tag)), iou(agg(rb, range(n), tag))
    rng = np.random.RandomState(SEED)
    dl = np.empty(B)
    for b in range(B):
        idx = rng.randint(0, n, n)
        dl[b] = iou(agg(rb, idx, tag)) - iou(agg(ra, idx, tag))
    lo, hi = np.percentile(dl, [2.5, 97.5])
    return {"iou_a": ia, "iou_b": ib, "delta": ib - ia,
            "ci95": [float(lo), float(hi)], "crosses_zero": bool(lo <= 0 <= hi),
            "q05_one_sided": float(np.percentile(dl, 5.0)), "n": n}


def self_contrast(rows):
    n = len(rows)
    ia, ib = iou(agg(rows, range(n), "base0")), iou(agg(rows, range(n), "head"))
    rng = np.random.RandomState(SEED)
    dl = np.empty(B)
    for b in range(B):
        idx = rng.randint(0, n, n)
        dl[b] = iou(agg(rows, idx, "head")) - iou(agg(rows, idx, "base0"))
    lo, hi = np.percentile(dl, [2.5, 97.5])
    return {"iou_base0": ia, "iou_head": ib, "delta": ib - ia,
            "ci95": [float(lo), float(hi)], "crosses_zero": bool(lo <= 0 <= hi),
            "q05_one_sided": float(np.percentile(dl, 5.0)), "n": n}


def miou4(ma):
    return float(np.mean([BASE0_NON_MA["EX"], BASE0_NON_MA["HE"], BASE0_NON_MA["SE"], ma]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="/root/autodl-tmp/audit/audit_20260916_A")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    R = a.run_dir

    pairs = [("T0", 1, 8000, "E1_T0", 4000),
             ("T1", 1, 8000, "E1_T1", 4000)]
    out = {"unit": "percent points (0-100) for reporting",
           "note": "E1 = 从原始臂 it8000 续训 4000 步（固定 lr=1e-5，真续训：恢复采样 RNG 状态）",
           "bootstrap": {"B": B, "seed": SEED, "method": "图像级配对（共用索引），先累计再求 IoU"},
           "self_contrast": {}, "extended_vs_original": {}, "invalid": [], "curves": {}}

    print("=" * 78)
    print("E1 续训臂自对照（head 相对 base0，同文件内配对）")
    for v, s, it, e1tag, e1it in pairs:
        p = os.path.join(R, "E1", "val_counts_%s_s%d_it%d.json" % (v, s, e1it))
        if not os.path.exists(p):
            out["invalid"].append({"arm": e1tag, "reason": "缺产物 %s" % p})
            print("  %-8s 缺产物 %s" % (e1tag, p))
            continue
        d, rows = load_counts(p)
        sc = self_contrast(rows)
        sc["file"] = p
        sc["iter"] = d.get("iter")
        out["self_contrast"][e1tag] = sc
        print("  %-8s it%-5s base0=%.6f head=%.6f  Δ=%+.6f CI=[%+.6f,%+.6f] 跨零=%s  mIoU=%.6f"
              % (e1tag, d.get("iter"), sc["iou_base0"], sc["iou_head"], sc["delta"],
                 sc["ci95"][0], sc["ci95"][1], sc["crosses_zero"], miou4(sc["iou_head"])))

    print("=" * 78)
    print("E1 续训 vs 原始（Δ = IoU(E1) − IoU(原始)；两臂 iter 不同是 E1 的定义使然）")
    for v, s, it, e1tag, e1it in pairs:
        p_ref = os.path.join(R, "val_counts_%s_s%d_it%d.json" % (v, s, it))
        p_e1 = os.path.join(R, "E1", "val_counts_%s_s%d_it%d.json" % (v, s, e1it))
        if not (os.path.exists(p_ref) and os.path.exists(p_e1)):
            out["invalid"].append({"contrast": e1tag, "reason": "缺产物",
                                   "ref": p_ref, "e1": p_e1})
            print("  %-8s 缺产物（ref=%s | e1=%s）" % (e1tag, os.path.exists(p_ref), os.path.exists(p_e1)))
            continue
        dref, rref = load_counts(p_ref)
        de1, re1 = load_counts(p_e1)
        why = []
        if len(rref) != len(re1):
            why.append("逐图行数不同 %d vs %d" % (len(rref), len(re1)))
        if dref.get("seed") != de1.get("seed"):
            why.append("seed 不同")
        ib_ref = iou(agg(rref, range(len(rref)), "base0"))
        ib_e1 = iou(agg(re1, range(len(re1)), "base0"))
        if abs(ib_ref - ib_e1) > 1e-12:
            why.append("base0 MA-IoU 不等 %.6f vs %.6f（val 集/tiling 不一致）" % (ib_ref, ib_e1))
        if why:
            out["invalid"].append({"contrast": e1tag, "reason": "；".join(why),
                                   "ref": p_ref, "e1": p_e1})
            print("  %-8s 判定 INVALID：%s" % (e1tag, "；".join(why)))
            continue
        r = paired_bootstrap(rref, re1)
        r["label"] = "%s(it%d) ｜ Δ = IoU(E1) − IoU(原始 it%d)" % (e1tag, e1it, it)
        r["a_file"], r["b_file"] = p_ref, p_e1
        r["iter_ref"], r["iter_e1"] = dref.get("iter"), de1.get("iter")
        r["mIoU_ref"] = miou4(r["iou_a"])
        r["mIoU_e1"] = miou4(r["iou_b"])
        out["extended_vs_original"][e1tag] = r
        print("  %-8s 原始=%.6f  E1=%.6f  Δ=%+.6f  CI=[%+.6f,%+.6f] 跨零=%s  q05=%+.6f"
              % (e1tag, r["iou_a"], r["iou_b"], r["delta"], r["ci95"][0], r["ci95"][1],
                 r["crosses_zero"], r["q05_one_sided"]))
        print("            ΔmIoU=%+.6f（原始 mIoU=%.6f → E1 mIoU=%.6f）"
              % (r["mIoU_e1"] - r["mIoU_ref"], r["mIoU_ref"], r["mIoU_e1"]))

    # E1 的 val 曲线（4 个点）
    for v in ("T0", "T1"):
        cp = os.path.join(R, "E1", "val_curve_%s_s1.json" % v)
        if os.path.exists(cp):
            c = json.load(open(cp))
            out["curves"]["E1_%s" % v] = {h["iter"]: h["head_MA_IoU"] for h in c.get("curve", [])}
            print("  E1_%s 曲线: %s" % (v, out["curves"]["E1_%s" % v]))

    outp = a.out or os.path.join(R, "experiments", "E1_eval.json")
    os.makedirs(os.path.dirname(outp), exist_ok=True)
    with open(outp, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print("=" * 78)
    print("写出:", outp)


if __name__ == "__main__":
    main()
