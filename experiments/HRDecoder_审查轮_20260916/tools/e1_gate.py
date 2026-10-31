#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E1 判定门：T1 是否"显著优于"T0（同 seed=1、同 val 集、同一冻结评估与 tiling）。

输入：train_T0T1.py 补丁版在 iter=8000 val 点导出的逐图计数 JSON
      （val_counts_{variant}_s1_it8000.json，字段 per_image[i][tag] = [tp, fp, fn]）

方法与 R0 完全一致（冻结规则，不新造）：
  * 图像级重采样，2000 次，seed 20260917
  * 两模型共用同一组重采样索引（配对 bootstrap）
  * 先累计 TP/FP/FN 再求 IoU（不是先算逐图 IoU 再平均）
  * 95% 百分位区间；空并集（tp+fp+fn==0）按冻结规则记 0

判定（契约：E1 是"条件触发"）：
  * 单侧检验：ΔIoU(T1-T0) 的 5% 分位 ≤ 0 → T1 未显著优于 T0 → TRIGGER（跑 E1 配对续训）
  * 若 5% 分位 > 0 → SKIP（T1 已显著更好，不再延长）

同时打印 3 条自检，任一失败即 rc=1（判定门不可用，链条会跳过 E1 并标记）：

  A. 两文件 val 集一致：两个 JSON 里 base0 的 IoU 必须逐位相等（同一 val、同一 tiling）
  B. 逐图行数与 n_val 一致
  C. base0 的 Δ 必须为 0

⚠ 已披露的局限：单 seed 比较无法把"采样规则效应"与"seed 噪声"分开
（seed 同时决定头初始化与采样 RNG 序列）。V1（3 seed × 2 臂）才是给 seed 方差的那一阶段。
本门是保守门：只要不能证明 T1 更好，就跑 E1。
"""
import argparse
import json
import sys

import numpy as np

B = 2000
SEED = 20260917
CLS = 4


def load(p):
    with open(p) as f:
        d = json.load(f)
    rows = d["per_image"]
    return d, rows


def iou(tp, fp, fn):
    den = tp + fp + fn
    return tp / den if den else 0.0


def agg(rows, idx, tag):
    tp = fp = fn = 0
    for i in idx:
        a, b, c = rows[i][tag]
        tp += a; fp += b; fn += c
    return tp, fp, fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--t0", required=True)
    ap.add_argument("--t1", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag-a", default="T0")
    ap.add_argument("--tag-b", default="T1")
    args = ap.parse_args()

    dA, rA = load(args.t0)
    dB, rB = load(args.t1)
    nA, nB = len(rA), len(rB)
    ok = True

    # --- 自检 A/B ---
    if nA != nB or nA != dA.get("n_val"):
        print("[gate] !! 自检B失败：行数 %d vs %d，n_val=%s" % (nA, nB, dA.get("n_val")))
        ok = False
    t = agg(rA, range(nA), "base0")
    u = agg(rB, range(nB), "base0")
    iou_b0_A, iou_b0_B = iou(*t), iou(*u)
    if abs(iou_b0_A - iou_b0_B) > 1e-12 or t != u:
        print("[gate] !! 自检A失败：两文件 base0 计数不一致 %s vs %s（val 集或 tiling 不同）" % (t, u))
        ok = False
    else:
        print("[gate] 自检A/B 通过：两臂 val 集与 tiling 完全一致，base0 TP/FP/FN = %s，IoU=%.6f"
              % (t, iou_b0_A))

    if dA.get("seed") != dB.get("seed"):
        print("[gate] !! 两臂 seed 不同（%s vs %s）—— 不是受控对照" % (dA.get("seed"), dB.get("seed")))
        ok = False

    # --- 点估计 ---
    tpA, fpA, fnA = agg(rA, range(nA), "head")
    tpB, fpB, fnB = agg(rB, range(nB), "head")
    iA, iB = iou(tpA, fpA, fnA), iou(tpB, fpB, fnB)
    d_point = iB - iA
    print("[gate] %s head MA-IoU = %.6f  (tp=%d fp=%d fn=%d)" % (args.tag_a, iA, tpA, fpA, fnA))
    print("[gate] %s head MA-IoU = %.6f  (tp=%d fp=%d fn=%d)" % (args.tag_b, iB, tpB, fpB, fnB))
    print("[gate] Δ(%s-%s) = %+.6f" % (args.tag_b, args.tag_a, d_point))

    # --- 配对图像级 bootstrap（先累计再求 IoU）---
    rng = np.random.RandomState(SEED)
    dl = np.empty(B, dtype=np.float64)
    n = min(nA, nB)
    for b in range(B):
        idx = rng.randint(0, n, n)
        a1 = agg(rA, idx, "head"); b1 = agg(rB, idx, "head")
        dl[b] = iou(*b1) - iou(*a1)
    lo2, hi2 = np.percentile(dl, [2.5, 97.5])
    p5 = float(np.percentile(dl, 5.0))
    print("[gate] 配对图像级 bootstrap %d 次(seed=%d)：Δ 95%%CI = [%+.6f, %+.6f]，双侧跨零=%s"
          % (B, SEED, lo2, hi2, bool(lo2 <= 0 <= hi2)))
    print("[gate] 单侧 5%% 分位 = %+.6f  ->  %s" % (p5, "≤0（T1 未显著更优）" if p5 <= 0 else ">0（T1 显著更优）"))

    trigger = (p5 <= 0.0)
    verdict = "TRIGGER" if trigger else "SKIP"
    print("[gate] DECISION: %s%s" % (verdict,
          "  —— T1 未显著优于 T0，跑 E1 配对续训 2x4000 步" if trigger
          else "  —— T1 已显著优于 T0，不跑 E1"))
    if not ok:
        print("[gate] !! 自检未全过 -> 判定门视为不可用")
        verdict = "GATE_INVALID"

    res = {"gate": "E1", "verdict": verdict, "selfcheck_ok": bool(ok),
           "method": {"unit": "image", "resamples": B, "rng_seed": SEED,
                      "accumulate_then_iou": True, "empty_union": 0.0, "cls_index": CLS},
           "arm_a": {"tag": args.tag_a, "file": args.t0, "seed": dA.get("seed"),
                     "head_MA_IoU": iA, "tp_fp_fn": [tpA, fpA, fnA], "n_val": nA},
           "arm_b": {"tag": args.tag_b, "file": args.t1, "seed": dB.get("seed"),
                     "head_MA_IoU": iB, "tp_fp_fn": [tpB, fpB, fnB], "n_val": nB},
           "base0_MA_IoU": iou_b0_A,
           "delta_point": d_point, "delta_ci95_two_sided": [float(lo2), float(hi2)],
           "delta_q05_one_sided": p5,
           "caveat": ("单 seed 无法分离采样规则效应与 seed 噪声；V1(3 seed x 2 臂) 才提供 seed 方差。"
                      "本门为保守门：不能证明 T1 更好就跑 E1。")}
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    print("[gate] 结果已写 %s" % args.out)

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
