#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""H3：MA 通道融合权重单变量扫描（prompt L4，零训练，只评 val）。

P_MA = w * P_student_MA + (1-w) * P_reference_MA
其余三类保持部署原样（0.5/0.5），逐位不变。
聚合：全 val 累计 TP/FP/FN（与冻结协议一致），阈值 0.5，严格 >。
"""
import argparse
import copy
import json
import os
import sys

import numpy as np
import torch

REPO = "/root/autodl-tmp/hrplus/repo"
os.chdir(REPO)
sys.path.insert(0, REPO)

CLASSES = ["EX", "HE", "SE", "MA"]
TAU = 0.5
WS = [0.5, 0.6, 0.7, 0.8, 1.0]


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


def counts(prob_ma, gt):
    p = prob_ma > TAU
    g = gt == 4
    ig = gt == IGNORE
    return (int((p & g).sum()), int((p & ~g & ~ig).sum()), int((~p & g).sum()))


IGNORE = 255


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    from mmcv import Config
    from mmseg.datasets import build_dataset
    from mmseg.models import build_segmentor
    from mmcv.runner import load_checkpoint

    cfg = Config.fromfile(args.config)
    d = copy.deepcopy(cfg.data["val"]); d.test_mode = True
    ds = build_dataset(d)
    gt_maps = ds.get_gt_seg_maps()
    n = len(ds)
    print("val =", n, flush=True)

    model = build_segmentor(cfg.model)
    load_checkpoint(model, args.ckpt, map_location="cpu", strict=False)
    model = model.cuda().eval()

    # 其余三类的融合概率不随 w 变 -> 只累加一次；w 只影响 MA
    fixed = {c: [0, 0, 0] for c in range(3)}
    scan = {w: [0, 0, 0] for w in WS}
    pr_hist = np.zeros(10, dtype=np.int64)
    ref_exact = 0

    for i in range(n):
        s = ds[i]
        img = as_img(s["img"])
        meta = as_meta(s["img_metas"])
        gt = np.asarray(gt_maps[i])
        ori = meta[0]["ori_shape"][:2]
        if gt.shape != tuple(ori):
            import cv2
            gt = cv2.resize(gt.astype(np.uint8), (ori[1], ori[0]),
                            interpolation=cv2.INTER_NEAREST)
        with torch.no_grad():
            ps = model._student_prob(img.unsqueeze(0).cuda(), meta, True)[0].cpu().numpy()
            pr = model._reference_prob(img.unsqueeze(0).cuda(), meta, True)[0].cpu().numpy()
        pf_others = 0.5 * ps[:3] + 0.5 * pr[:3]
        ps_ma, pr_ma = ps[3], pr[3]
        pr_hist += np.histogram(pr_ma, bins=np.linspace(0, 1, 11))[0]
        ref_exact += int((pr_ma == 0.0).sum()) + int((pr_ma == 1.0).sum())

        for c in range(3):
            p = pf_others[c] > TAU
            g = gt == (c + 1)
            ig = gt == IGNORE
            fixed[c][0] += int((p & g).sum())
            fixed[c][1] += int((p & ~g & ~ig).sum())
            fixed[c][2] += int((~p & g).sum())
        for w in WS:
            p = (w * ps_ma + (1.0 - w) * pr_ma) > TAU
            g = gt == 4
            ig = gt == IGNORE
            scan[w][0] += int((p & g).sum())
            scan[w][1] += int((p & ~g & ~ig).sum())
            scan[w][2] += int((~p & g).sum())
        if (i + 1) % 20 == 0 or i + 1 == n:
            print("  %d/%d" % (i + 1, n), flush=True)

    rows = []
    for c, cn in zip(range(3), CLASSES[:3]):
        tp, fp, fn = fixed[c]
        den = tp + fp + fn
        rows.append({"class": cn, "w": 0.5, "TP": tp, "FP": fp, "FN": fn,
                     "IoU": tp / den if den else 0.0})
    best = None
    for w in WS:
        tp, fp, fn = scan[w]
        den = tp + fp + fn
        iou = tp / den if den else 0.0
        others = [r["IoU"] for r in rows]
        miou = (sum(others) + iou) / 4.0
        rows.append({"class": "MA", "w": w, "TP": tp, "FP": fp, "FN": fn,
                     "IoU": iou, "mIoU": miou})
        print("  w=%.2f  MA TP=%-7d FP=%-7d FN=%-7d MA_IoU=%.4f  mIoU=%.4f"
              % (w, tp, fp, fn, iou, miou), flush=True)
        if best is None or miou > best[1]:
            best = (w, miou)

    out = {"probe": "H3_fusion_weight_scan", "n_val": n, "weights": WS,
           "tau": TAU, "comparison": "strict_>",
           "aggregation": "dataset_cumulative",
           "results": rows,
           "best": {"w": best[0], "mIoU": best[1]},
           "reference_prob_histogram": pr_hist.tolist(),
           "reference_prob_at_exactly_0_or_1": ref_exact,
           "baseline_w05_miou": [r for r in rows if r.get("w") == 0.5][0]["mIoU"],
           "note": "其余三类在任何 w 下逐位不变（结构隔离）"}
    os.makedirs(os.path.join(args.out_dir, "experiments"), exist_ok=True)
    p = os.path.join(args.out_dir, "experiments", "h3_fusion_scan.json")
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    os.replace(tmp, p)
    print("\n最优 w=%.2f mIoU=%.4f（w=0.5 基线 %.4f）" % (best[0], best[1], out["baseline_w05_miou"]))
    print("写出:", p)


if __name__ == "__main__":
    main()
