#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""§9.1 首查：训练目标是否等于部署预测 —— 定量诊断（只用 val）。

不做训练，只做推理分解。对每张 val 图分别取：
  prob_s = student 平铺路径概率
  prob_r = 冻结 reference 整图概率
  prob_f = 0.5*prob_r + 0.5*prob_s   （部署实际使用的融合）

在 τ=0.5、严格 `>` 下，融合判阳等价于  P_s > 1 - P_r。
因此可解析地定义：
  - 若 P_r == 0            -> 需要 P_s > 1，sigmoid 开区间不可达 => 不可纠正为阳性
  - 若 P_r >= 0.5          -> 需要 P_s > 1-P_r <= 0.5，reference 主导，学生几乎必然判阳
  - 若 1-P_r >= 1          -> 即 P_r <= 0，同上

同时对比三种判定下的 MA 指标：student-only / reference-only / 0.5 融合，
以判断"融合"本身是增益还是瓶颈。

输出 artifacts：
  diagnostics/val_per_image_counts.csv
  diagnostics/branch_visit_counts.json
  diagnostics/train_inference_contract.md（由本脚本的数据段 + 说明拼装写入）
"""
import argparse
import copy
import json
import os
import sys

import numpy as np
import torch

CLASSES = ["EX", "HE", "SE", "MA"]
IGNORE = 255
TAU = 0.5


def unwrap(x):
    n = 0
    while hasattr(x, "data") and n < 8:
        x = x.data
        n += 1
    return x


def as_img(x):
    x = unwrap(x)
    while isinstance(x, (list, tuple)):
        x = unwrap(x[0])
    return x


def as_meta(x):
    x = unwrap(x)
    if isinstance(x, dict):
        return [x]
    out = []
    for e in x:
        e = unwrap(e)
        out.extend([e] if isinstance(e, dict) else [unwrap(e)])
    return out


def counts_from(prob, gt, c):
    """prob: (4,H,W) in [0,1]; gt: (H,W); c: class index -> (TP, FP, FN)"""
    p = prob[c] > TAU
    g = gt == (c + 1)
    ig = gt == IGNORE
    return int((p & g).sum()), int((p & ~g & ~ig).sum()), int((~p & g).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    from mmcv import Config
    from mmseg.datasets import build_dataset
    from mmseg.models import build_segmentor

    cfg = Config.fromfile(args.config)
    ds_cfg = copy.deepcopy(cfg.data["val"])
    ds_cfg.test_mode = True
    ds = build_dataset(ds_cfg)

    model = build_segmentor(cfg.model)
    from mmcv.runner import load_checkpoint
    load_checkpoint(model, args.ckpt, map_location="cpu", strict=False)
    model = model.cuda().eval()

    n = len(ds) if args.limit <= 0 else min(args.limit, len(ds))
    print("val 图数 =", n, flush=True)
    gt_maps = ds.get_gt_seg_maps()          # 一次取全部；与 _evaluate 用的同一条路径
    assert len(gt_maps) == len(ds), (len(gt_maps), len(ds))
    print("gt_seg_maps 数量 =", len(gt_maps),
          " dtype/值域抽样 =", np.unique(gt_maps[0])[:8], flush=True)

    # 分解累加器
    agg = {k: {c: [0, 0, 0] for c in range(4)}
           for k in ("student", "reference", "blend")}
    per_image = []
    mafn_req_ps = []      # MA FN 像素所需的 P_s = 1 - P_r
    mafp_req_ps = []
    pr_hist = {c: np.zeros(10, dtype=np.int64) for c in range(4)}
    ref_exact_zero = 0
    ref_exact_one = 0
    total_px = 0
    ig_px_total = 0

    for i in range(n):
        s = ds[i]
        img = as_img(s["img"])
        meta = [dict(m) for m in as_meta(s["img_metas"])]
        gt = None
        if "gt_semantic_seg" in s:
            gt = unwrap(s["gt_semantic_seg"])
        if gt is None:
            # 测试模式不带 gt：用 dataset 的原始标签（与 _evaluate 同一路径）
            gt = torch.from_numpy(np.asarray(gt_maps[i])).long()
        gt = gt.squeeze().cpu().numpy()
        ori = meta[0]["ori_shape"][:2]
        if gt.shape != tuple(ori):
            import cv2
            gt = cv2.resize(gt.astype(np.uint8), (ori[1], ori[0]),
                            interpolation=cv2.INTER_NEAREST)

        with torch.no_grad():
            ps = model._student_prob(img.unsqueeze(0).cuda(), meta, True)
            pr = model._reference_prob(img.unsqueeze(0).cuda(), meta, True)
        ps = ps[0].cpu().numpy()
        pr = pr[0].cpu().numpy()
        pf = 0.5 * pr + 0.5 * ps

        total_px += gt.size
        ig_px_total += int((gt == IGNORE).sum())
        ref_exact_zero += int((pr == 0.0).sum())
        ref_exact_one += int((pr == 1.0).sum())
        for c in range(4):
            pr_hist[c] += np.histogram(pr[c], bins=np.linspace(0, 1, 11))[0]

        row = {"idx": i, "filename": meta[0].get("ori_filename", ""),
               "h": int(gt.shape[0]), "w": int(gt.shape[1]),
               "ignore_px": int((gt == IGNORE).sum())}
        for tag, prob in (("student", ps), ("reference", pr), ("blend", pf)):
            for c in range(4):
                tp, fp, fn = counts_from(prob, gt, c)
                agg[tag][c][0] += tp
                agg[tag][c][1] += fp
                agg[tag][c][2] += fn
                row["%s_%s_TP" % (tag, CLASSES[c])] = tp
                row["%s_%s_FP" % (tag, CLASSES[c])] = fp
                row["%s_%s_FN" % (tag, CLASSES[c])] = fn
        per_image.append(row)

        # MA(3) 的错误像素所需 P_s
        g = gt == 4
        ig = gt == IGNORE
        pfb = pf[3] > TAU
        psb = ps[3] > TAU
        fn_mask = g & ~pfb
        fp_mask = ~g & ~ig & pfb
        if fn_mask.any():
            mafn_req_ps.append((1.0 - pr[3][fn_mask]).ravel())
        if fp_mask.any():
            mafp_req_ps.append((1.0 - pr[3][fp_mask]).ravel())
        if (i + 1) % 20 == 0 or i + 1 == n:
            print("  %d/%d" % (i + 1, n), flush=True)

    def miou(tag):
        ious = []
        for c in range(4):
            tp, fp, fn = agg[tag][c]
            den = tp + fp + fn
            ious.append(tp / den if den > 0 else 0.0)
        return float(np.mean(ious)), ious

    res = {}
    for tag in ("student", "reference", "blend"):
        m, ious = miou(tag)
        res[tag] = {"mIoU": m,
                    "per_class": {CLASSES[c]: ious[c] for c in range(4)},
                    "counts": {CLASSES[c]: {"TP": agg[tag][c][0],
                                            "FP": agg[tag][c][1],
                                            "FN": agg[tag][c][2]}
                               for c in range(4)}}
        print("[%s] mIoU=%.4f  MA IoU=%.4f" % (tag, m, ious[3]))

    # 不可纠正性统计
    uc = {}
    for name, arrs in (("MA_FN", mafn_req_ps), ("MA_FP", mafp_req_ps)):
        if not arrs:
            uc[name] = {"n": 0}
            continue
        a = np.concatenate(arrs)
        uc[name] = {
            "n": int(a.size),
            "required_P_student_mean": float(a.mean()),
            "required_P_student_median": float(np.median(a)),
            "frac_required_ge_0.9": float((a >= 0.9).mean()),
            "frac_required_ge_0.99": float((a >= 0.99).mean()),
            "frac_required_ge_1.0": float((a >= 1.0).mean()),
            "note": ("required_P_student = 1 - P_reference；"
                     ">=1.0 表示 sigmoid 开区间内不可达 => 该像素的判定"
                     "无法被学生翻转"),
        }
        print("[%s] n=%d  所需 P_s 中位数=%.3f  >=0.9 比例=%.3f  >=1.0 比例=%.3f"
              % (name, a.size, np.median(a), (a >= 0.9).mean(), (a >= 1.0).mean()))

    out = {
        "probe": "train_inference_contract_quant",
        "tau": TAU, "comparison": "strict_>",
        "n_val_images": n,
        "ignore_pixels_total": ig_px_total,
        "total_pixels": total_px,
        "reference_exact_zero_pixels": ref_exact_zero,
        "reference_exact_one_pixels": ref_exact_one,
        "metrics_under_three_decisions": res,
        "uncorrectable_analysis": uc,
        "reference_prob_histogram_per_class": {
            CLASSES[c]: pr_hist[c].tolist() for c in range(4)},
        "bins": "np.linspace(0,1,11)",
    }

    od = args.out_dir
    os.makedirs(os.path.join(od, "diagnostics"), exist_ok=True)
    p = os.path.join(od, "diagnostics", "val_blend_decomposition.json")
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    os.replace(tmp, p)

    # per-image CSV
    import csv
    cp = os.path.join(od, "diagnostics", "val_per_image_counts.csv")
    if per_image:
        keys = list(per_image[0].keys())
        with open(cp, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(per_image)

    print("\n写出:", p)
    print("写出:", cp)


if __name__ == "__main__":
    main()
