#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""H4 的 test 确认：把融合后 MA 修正头接进正式推理路径，对比 fused vs corrected。

协议冻结（与前几轮完全一致）：
  - test 227 张，4 类 IoU 不含背景，阈值 0.5，比较符严格 >
  - 全数据集累计 TP/FP/FN
  - 只改 MA 通道；EX/HE/SE 必须逐位不变（结构隔离，用作正确性断言）
"""
import argparse
import copy
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

REPO = "/root/autodl-tmp/hrplus/repo"
os.chdir(REPO)
sys.path.insert(0, REPO)
CLASSES = ["EX", "HE", "SE", "MA"]
TAU = 0.5


class MACorrectorHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(7, 16, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, f):
        return 6.0 * torch.tanh(self.net(f))


def logit(p, eps=1e-4):
    p = p.clamp(eps, 1 - eps)
    return torch.log(p / (1 - p))


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--head", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    from mmcv import Config
    from mmseg.datasets import build_dataset
    from mmseg.models import build_segmentor
    from mmcv.runner import load_checkpoint

    cfg = Config.fromfile(args.config)
    d = copy.deepcopy(cfg.data["test"]); d.test_mode = True
    ds = build_dataset(d)
    gt_maps = ds.get_gt_seg_maps()
    n = len(ds)
    print("test =", n, flush=True)

    model = build_segmentor(cfg.model)
    load_checkpoint(model, args.ckpt, map_location="cpu", strict=False)
    model = model.cuda().eval()
    for p in model.parameters():
        p.requires_grad_(False)

    head = MACorrectorHead().cuda()
    ck = torch.load(args.head, map_location="cpu")
    head.load_state_dict(ck["head"] if "head" in ck else ck)
    head.eval()
    print("head 载入自", args.head, "step=", ck.get("step") if isinstance(ck, dict) else "?", flush=True)

    agg = {"fused": {c: [0, 0, 0] for c in range(4)},
           "corrected": {c: [0, 0, 0] for c in range(4)}}
    nonma_identical = True
    t0 = time.time()

    for i in range(n):
        s = ds[i]
        img = as_img(s["img"]).unsqueeze(0).cuda()
        meta = as_meta(s["img_metas"])
        gt = np.asarray(gt_maps[i])
        ori = meta[0]["ori_shape"][:2]
        if gt.shape != tuple(ori):
            import cv2
            gt = cv2.resize(gt.astype(np.uint8), (ori[1], ori[0]),
                            interpolation=cv2.INTER_NEAREST)
        gtf = torch.from_numpy(gt).cuda()
        with torch.no_grad():
            ps = model._student_prob(img, meta, True)[0]     # (4,H,W) prob
            pr = model._reference_prob(img, meta, True)[0]
            pf = 0.5 * ps + 0.5 * pr
            img_f = torch.nn.functional.interpolate(
                img, size=tuple(ori), mode="bilinear", align_corners=False)[0]
            f7 = torch.cat([img_f, logit(pf[3:4]), logit(ps[3:4]),
                            logit(pr[3:4]), (ps[3:4] - pr[3:4])], dim=0).unsqueeze(0)
            delta = head(f7)[0, 0]
            p_new = torch.sigmoid(logit(pf[3:4][0]) + delta)
            pc = torch.cat([pf[:3], p_new.unsqueeze(0)], dim=0)

        for c in range(4):
            for tag, prob in (("fused", pf), ("corrected", pc)):
                p = prob[c] > TAU
                g = gtf == (c + 1)
                ig = gtf == 255
                agg[tag][c][0] += int((p & g).sum())
                agg[tag][c][1] += int((p & ~g & ~ig).sum())
                agg[tag][c][2] += int((~p & g).sum())
        # 结构隔离断言：非 MA 三通道的判定必须逐位相同
        for c in range(3):
            if not torch.equal(pf[c] > TAU, pc[c] > TAU):
                nonma_identical = False
        if (i + 1) % 40 == 0 or i + 1 == n:
            print("  %d/%d  %.1fs" % (i + 1, n, time.time() - t0), flush=True)

    def miou(tag):
        ious = {}
        for c in range(4):
            tp, fp, fn = agg[tag][c]
            den = tp + fp + fn
            ious[CLASSES[c]] = tp / den if den else 0.0
        return float(np.mean(list(ious.values()))), ious

    mf, iouf = miou("fused")
    mc, iouc = miou("corrected")
    print("\n[fused]     mIoU=%.4f  %s" % (mf, {k: round(v, 4) for k, v in iouf.items()}))
    print("[corrected] mIoU=%.4f  %s" % (mc, {k: round(v, 4) for k, v in iouc.items()}))
    print("[delta]     mIoU=%+.4f  MA=%+.4f" % (mc - mf, iouc["MA"] - iouf["MA"]))
    print("[非MA 三通道判定逐位相同] =", nonma_identical)

    out = {"probe": "H4_test_confirmation", "n_test": n, "tau": TAU,
           "comparison": "strict_>", "aggregation": "dataset_cumulative",
           "fused": {"mIoU": mf, "per_class": iouf,
                     "counts": {CLASSES[c]: dict(zip(("TP", "FP", "FN"), agg["fused"][c]))
                                for c in range(4)}},
           "corrected": {"mIoU": mc, "per_class": iouc,
                         "counts": {CLASSES[c]: dict(zip(("TP", "FP", "FN"), agg["corrected"][c]))
                                    for c in range(4)}},
           "delta_mIoU": mc - mf,
           "delta_MA_IoU": iouc["MA"] - iouf["MA"],
           "non_MA_bit_identical": bool(nonma_identical),
           "historical_ref": {"D@2000 fused test mIoU": 0.42576397633218704},
           "status": ("PASS" if (mc > mf and nonma_identical) else
                      ("NEUTRAL" if mc == mf else "FAIL_or_ISOLATION_BROKEN"))}
    p = os.path.join(args.out_dir, "experiments", "h4_test_confirmation.json")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    os.replace(tmp, p)
    print("\n状态:", out["status"])
    print("写出:", p)


if __name__ == "__main__":
    main()
