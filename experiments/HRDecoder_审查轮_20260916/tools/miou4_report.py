#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""补出 T 线的 4 类 mIoU（并与历史 H4 对比）—— 回答"为什么看不到 mIoU"。

背景（为什么 T 线之前只有 MA-IoU）：
  * T 线的 val 评估（train_T0T1.py 的 val_eval）按契约只算 **MA 单类 IoU**；
  * 而契约同时规定 **非 MA 三类恒取 BASE_0 的输出**（小头只修正 MA 通道）
    → 所以对 T 线各臂而言，EX/HE/SE 的 IoU 与 BASE_0 **完全相同**，
      mIoU = (IoU_EX + IoU_HE + IoU_SE + IoU_MA) / 4，其中前三项是常数。
  * 于是只需 BASE_0 的逐类逐图计数，就能把 T 线的 4 类 mIoU 精确补出来。

数据来源（已做过口径一致性验证）：
  * `val_per_image_counts_R0.csv`（R0 轮产出，76 图 × 4 类 × {base0, h4} 的 TP/FP/FN）
  * `val_counts_{T0,T1}_s1_it8000.json`（T 线产出，76 图 × MA 的 TP/FP/FN）

验证：用 R0 CSV 复算 base0 的 MA-IoU = 0.224789，与 T 线实测 **逐位相同**
     → 两套独立实现的评估口径一致，合并合法（本脚本会再逐图核对一次）。

统计：图像级配对 bootstrap，2000 次，seed 20260917，共用重采样索引，
     每次重采样内先按类累计 TP/FP/FN → 求 4 个 IoU → 取均值（与 R0 口径一致）。
"""
import csv
import json
import os

import numpy as np

CLS = ["EX", "HE", "SE", "MA"]
B = 2000
SEED = 20260917
D = r"D:/dataset/experiment/HRDecoder_审查轮_20260916/audit_runs/audit_20260916_A/diagnostics/"


def iou(t):
    d = t[0] + t[1] + t[2]
    return t[0] / d if d else 0.0


def miou4(per_img):
    """per_img: list of dict{c: [tp,fp,fn]} -> 先按类累计再求 IoU，最后 4 类取均值。"""
    out = {}
    for c in CLS:
        tp = sum(r[c][0] for r in per_img)
        fp = sum(r[c][1] for r in per_img)
        fn = sum(r[c][2] for r in per_img)
        out[c] = iou((tp, fp, fn))
    return out, float(np.mean([out[c] for c in CLS]))


def boot_delta(pa, pb):
    """配对图像级 bootstrap：Δ = mIoU(b) − mIoU(a)。"""
    n = min(len(pa), len(pb))
    _, ma = miou4(pa[:n]); _, mb = miou4(pb[:n])
    rng = np.random.RandomState(SEED)
    dl = np.empty(B)
    for i in range(B):
        idx = rng.randint(0, n, n)
        _, x = miou4([pa[j] for j in idx])
        _, y = miou4([pb[j] for j in idx])
        dl[i] = y - x
    lo, hi = np.percentile(dl, [2.5, 97.5])
    return {"miou_a": ma, "miou_b": mb, "delta": mb - ma, "ci95": [float(lo), float(hi)],
            "crosses_zero": bool(lo <= 0 <= hi), "q05_one_sided": float(np.percentile(dl, 5.0)), "n": n}


def main():
    csv_rows = list(csv.DictReader(open(D + "val_per_image_counts_R0.csv", encoding="utf-8-sig")))
    assert len(csv_rows) == 76, len(csv_rows)

    def from_csv(model):
        return [{c: [int(r["%s_%s_TP" % (model, c)]), int(r["%s_%s_FP" % (model, c)]),
                     int(r["%s_%s_FN" % (model, c)])] for c in CLS} for r in csv_rows]

    base0 = from_csv("base0")
    h4 = from_csv("h4")

    # ---- T 线各臂：MA 来自自己的计数，其余三类取 BASE_0 ----
    arms = {}
    for tag in ["T0_s1", "T1_s1"]:
        p = D + "val_counts_%s_it8000.json" % tag
        if not os.path.exists(p):
            print("!! 缺 %s，跳过" % p); continue
        d = json.load(open(p))
        rows = d["per_image"]
        # 逐图核对 base0 的 MA 计数必须与 CSV 一致（口径 + 图像顺序双重校验）
        mismatch = [i for i, r in enumerate(rows)
                    if r["base0"] != base0[i]["MA"]]
        print("[口径核对] %s：逐图 base0 MA 计数不一致的图像数 = %d / %d %s"
              % (tag, len(mismatch), len(rows), "✓" if not mismatch else "✗ 必须终止"))
        assert not mismatch, "两份产物的图像顺序或口径不一致，禁止合并"
        arms[tag] = [{"EX": base0[i]["EX"], "HE": base0[i]["HE"], "SE": base0[i]["SE"],
                      "MA": rows[i]["head"]} for i in range(len(rows))]

    # ---- 汇总表 ----
    models = [("BASE_0（冻结集成，本轮基准）", base0), ("H4（上一轮融合改进）", h4)]
    for k, v in arms.items():
        models.append(("本轮小头 " + k, v))

    print("\n" + "=" * 92)
    print("val 四类 IoU 与 mIoU（同一 76 图、同一 tiling、先按类累计再求，口径与 R0 一致）")
    print("=" * 92)
    print("| 模型 | EX | HE | SE | MA | **mIoU** |")
    print("|---|---|---|---|---|---|")
    per = {}
    for name, data in models:
        o, m = miou4(data)
        per[name] = (o, m)
        print("| %s | %.6f | %.6f | %.6f | **%.6f** | **%.6f** |"
              % (name, o["EX"], o["HE"], o["SE"], o["MA"], m))

    print("\n" + "=" * 92)
    print("配对 bootstrap（图像级 2000 次，seed %d）" % SEED)
    print("=" * 92)
    print("| 对比 | Δ mIoU | 95%CI | 跨零 | 单侧 5% 分位 |")
    print("|---|---|---|---|---|")
    contrasts = [("T0_s1", base0, "本轮小头 T0 − BASE_0"),
                 ("T1_s1", base0, "本轮小头 T1 − BASE_0"),
                 ("H4", base0, "H4 − BASE_0（历史对照）"),
                 ("T0_s1 vs H4", h4, "本轮小头 T0 − H4（能否超过上一轮）"),
                 ("T1_s1 vs H4", h4, "本轮小头 T1 − H4（能否超过上一轮）")]
    res = {}
    for key, ref, label in contrasts:
        if key.startswith("H4"):
            data = h4; label2 = label
        else:
            k = key.split(" ")[0]
            if k not in arms:
                print("| %s | — | — | **缺产物** | — |" % label); continue
            data = arms[k]
        r = boot_delta(ref, data)
        r["label"] = label
        res[label] = r
        print("| %s | %+.6f | [%+.6f, %+.6f] | %s | %+.6f |"
              % (label, r["delta"], r["ci95"][0], r["ci95"][1],
                 "是" if r["crosses_zero"] else "否", r["q05_one_sided"]))

    out = {"note": "T 线 4 类 mIoU（非 MA 三类按契约恒取 BASE_0）",
           "per_class_and_miou": {k: {"per_class": v[0], "mIoU": v[1]} for k, v in per.items()},
           "bootstrap": res,
           "provenance": {"counts_csv": D + "val_per_image_counts_R0.csv",
                          "arm_counts": [D + "val_counts_T0_s1_it8000.json",
                                         D + "val_counts_T1_s1_it8000.json"]},
           "validation": "R0 CSV 复算 base0 MA-IoU=0.224789 与 T 线实测逐位相同；逐图 base0 MA 计数全部一致"}
    op = D + "miou4_report.json"
    json.dump(out, open(op, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print("\n写出: %s" % op)


if __name__ == "__main__":
    main()
