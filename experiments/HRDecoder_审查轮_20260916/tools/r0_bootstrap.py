#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""R0 §5.2：BASE_0 vs H4 的配对 bootstrap（图像级重采样）。

契约要求：
  - 2000 次重采样，固定 seed=20260917
  - 两模型使用**相同抽样索引**
  - 每次先累计 TP/FP/FN，再求四类 IoU / mIoU
  - 输出差值（H4 − BASE_0）与 95% percentile 区间
  - 用冻结的空并集规则（该类分母为 0 → 记 0），并报告不可定义比例
  - 不用像素 bootstrap，不用每图 IoU 平均
  - 患者 ID 不可用 → 图像级重采样并标注未处理患者相关性

用法：
  python r0_bootstrap.py --csv <...>/diagnostics/val_per_image_counts_R0.csv \
                         --out-dir <audit_runs/<run_id>>
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

CLASSES = ["EX", "HE", "SE", "MA"]
N_BOOT = 2000
SEED = 20260917


def load(csv_path):
    with open(csv_path, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    return rows


def counts_matrix(rows, tag):
    """返回 (n_img, 4, 3) 的 [TP, FP, FN]。"""
    n = len(rows)
    M = np.zeros((n, 4, 3), dtype=np.int64)
    for i, r in enumerate(rows):
        for c in range(4):
            M[i, c, 0] = int(r["%s_%s_TP" % (tag, CLASSES[c])])
            M[i, c, 1] = int(r["%s_%s_FP" % (tag, CLASSES[c])])
            M[i, c, 2] = int(r["%s_%s_FN" % (tag, CLASSES[c])])
    return M


def iou_from_counts(cnt):
    """cnt: (4,3) 累计计数 -> (mIoU, 每类 IoU, 不可定义类数)。空并集 -> 0。"""
    tp = cnt[:, 0].astype(np.float64)
    fp = cnt[:, 1].astype(np.float64)
    fn = cnt[:, 2].astype(np.float64)
    den = tp + fp + fn
    undef = int((den == 0).sum())
    iou = np.zeros(4, dtype=np.float64)
    nz = den > 0
    iou[nz] = tp[nz] / den[nz]
    return float(iou.mean()), iou, undef


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    rows = load(args.csv)
    n = len(rows)
    B = counts_matrix(rows, "base0")
    H = counts_matrix(rows, "h4")
    print("val 图数 =", n, flush=True)
    assert n == 76, "契约要求完整 76 张 val，实际 %d" % n

    # 点估计（全量，不重采样）
    mB, iouB, undB = iou_from_counts(B.sum(axis=0))
    mH, iouH, undH = iou_from_counts(H.sum(axis=0))
    print("[BASE_0] mIoU=%.4f  %s" % (mB, {c: round(v, 4) for c, v in zip(CLASSES, iouB)}))
    print("[H4]     mIoU=%.4f  %s" % (mH, {c: round(v, 4) for c, v in zip(CLASSES, iouH)}))
    print("[delta]  mIoU=%+.4f  %s"
          % (mH - mB, {c: round(h - b, 4) for c, b, h in zip(CLASSES, iouB, iouH)}))

    rng = np.random.RandomState(args.seed)
    dmiou = np.zeros(args.n_boot)
    dma = np.zeros(args.n_boot)
    per_class_diff = {c: np.zeros(args.n_boot) for c in CLASSES}
    undef_b = 0
    undef_h = 0
    for b in range(args.n_boot):
        idx = rng.randint(0, n, size=n)          # 两模型共用同一索引
        mbi, iB, ub = iou_from_counts(B[idx].sum(axis=0))
        mhi, iH, uh = iou_from_counts(H[idx].sum(axis=0))
        undef_b += ub
        undef_h += uh
        dmiou[b] = mhi - mbi
        dma[b] = iH[3] - iB[3]
        for ci, c in enumerate(CLASSES):
            per_class_diff[c][b] = iH[ci] - iB[ci]

    def pct(a):
        return (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)))

    lo, hi = pct(dmiou)
    lo_ma, hi_ma = pct(dma)
    out = {
        "probe": "R0_paired_bootstrap",
        "csv": args.csv,
        "n_images": n,
        "n_bootstrap": args.n_boot,
        "seed": args.seed,
        "resampling": "image-level (patient ID UNAVAILABLE -> patient correlation NOT handled)",
        "same_indices_for_both_models": True,
        "accumulate_then_divide": True,
        "empty_union_rule": "denominator 0 -> class IoU recorded as 0 (frozen rule)",
        "point_estimate": {
            "base0": {"mIoU": mB, "per_class": {c: float(v) for c, v in zip(CLASSES, iouB)}},
            "h4": {"mIoU": mH, "per_class": {c: float(v) for c, v in zip(CLASSES, iouH)}},
            "delta_mIoU": mH - mB,
            "delta_per_class": {c: float(h - b) for c, b, h in zip(CLASSES, iouB, iouH)},
        },
        "bootstrap": {
            "delta_mIoU": {"mean": float(dmiou.mean()), "ci95": [lo, hi],
                           "frac_gt_0": float((dmiou > 0).mean())},
            "delta_MA": {"mean": float(dma.mean()), "ci95": [lo_ma, hi_ma],
                         "frac_gt_0": float((dma > 0).mean())},
            "delta_per_class": {
                c: {"mean": float(per_class_diff[c].mean()),
                    "ci95": list(pct(per_class_diff[c])),
                    "frac_gt_0": float((per_class_diff[c] > 0).mean())}
                for c in CLASSES},
        },
        "ci_crosses_zero_mIoU": bool(lo <= 0 <= hi),
        "ci_crosses_zero_MA": bool(lo_ma <= 0 <= hi_ma),
        "undefined_class_occurrences": {"base0": undef_b, "h4": undef_h,
                                        "per_bootstrap_max_possible": 4 * args.n_boot},
        "caveats": [
            "该区间只反映固定模型在当前 val 上的图像采样变动；",
            "不消除在同一 val 上选模型的乐观偏差，不代表训练种子方差，也不恢复 test 独立性。",
            "跨零不等于方法无效。",
            "未处理患者相关性（患者 ID 不可用）。",
        ],
    }
    os.makedirs(os.path.join(args.out_dir, "diagnostics"), exist_ok=True)
    p = os.path.join(args.out_dir, "diagnostics", "paired_bootstrap.json")
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    os.replace(tmp, p)
    print("\nΔmIoU 95%% CI = [%+.4f, %+.4f]  跨零=%s" % (lo, hi, out["ci_crosses_zero_mIoU"]))
    print("ΔMA   95%% CI = [%+.4f, %+.4f]  跨零=%s" % (lo_ma, hi_ma, out["ci_crosses_zero_MA"]))
    print("写出:", p)


if __name__ == "__main__":
    main()
