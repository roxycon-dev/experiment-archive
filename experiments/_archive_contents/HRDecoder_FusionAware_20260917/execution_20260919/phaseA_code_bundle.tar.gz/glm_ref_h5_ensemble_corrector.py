#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""H5：把 H4 的修正头接到集成的融合输出上（先在 val 验证，再上 test）。

动机：集成的 MA(24.73) 与 D 的 MA(24.73) 完全相同 → 集成对 MA 零贡献；
      但集成的 EX/HE/SE 更强。若 H4 头能把集成的 MA 也提升，则可组合出
      「集成 EX/HE/SE + 修正后的 MA」。
做法：P_D = 加权集成的融合 MA 概率（含翻转 TTA）；
      P_s / P_r 仍取 D 的 student / reference（头的特征语义不变）。
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


class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(7, 16, 3, 1, 1), nn.ReLU(inplace=True),
                                 nn.Conv2d(16, 16, 3, 1, 1), nn.ReLU(inplace=True),
                                 nn.Conv2d(16, 1, 1))

    def forward(self, f):
        return 6.0 * torch.tanh(self.net(f))


def logit(p, eps=1e-4):
    return torch.log(p.clamp(eps, 1 - eps) / (1 - p.clamp(eps, 1 - eps)))


def unwrap(x):
    n = 0
    while hasattr(x, "data") and n < 8:
        x = x.data; n += 1
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
    ap.add_argument("--split", default="val")
    ap.add_argument("--head", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--limit", type=int, default=0)
    # 集成成员：cfg=ckpt:weight（分号分隔）
    ap.add_argument("--models", required=True)
    ap.add_argument("--flip", action="store_true")
    args = ap.parse_args()

    from mmcv import Config
    from mmseg.datasets import build_dataset
    from mmseg.models import build_segmentor
    from mmcv.runner import load_checkpoint

    base = Config.fromfile("configs/lesion/exp/hrplus_patch_dice.py")
    d = copy.deepcopy(base.data[args.split]); d.test_mode = True
    ds = build_dataset(d)
    gt_maps = ds.get_gt_seg_maps()
    n = len(ds) if args.limit <= 0 else min(args.limit, len(ds))

    specs = []
    for sp in args.models.split(";"):
        sp = sp.strip()
        if not sp:
            continue
        cfgp, rest = sp.split("=", 1)
        if ":" in rest:
            ck, w = rest.rsplit(":", 1); w = float(w)
        else:
            ck, w = rest, 1.0
        specs.append((cfgp, ck, w))
    print("成员数 =", len(specs), " 权重 =", [s[2] for s in specs], flush=True)

    models = []
    for cfgp, ck, w in specs:
        c = Config.fromfile(cfgp)
        m = build_segmentor(c.model)
        load_checkpoint(m, ck, map_location="cpu", strict=False)
        models.append((m.cuda().eval(), w))
    tot = sum(w for _, w in models) * (2 if args.flip else 1)

    # D 的 student/reference —— 用于构造头的 P_s / P_r
    dc = Config.fromfile("configs/lesion/exp/hrplus_patch_dice.py")
    dm = build_segmentor(dc.model)
    load_checkpoint(dm, "/root/autodl-tmp/hrplus/D_0913_retry/best_mIoU_iter_2000.pth",
                    map_location="cpu", strict=False)
    dm = dm.cuda().eval()

    head = Head().cuda()
    ck = torch.load(args.head, map_location="cpu")
    head.load_state_dict(ck["head"] if "head" in ck else ck)
    head.eval()

    agg = {"fused": {c: [0, 0, 0] for c in range(4)},
           "corrected": {c: [0, 0, 0] for c in range(4)}}
    nonma_same = True
    t0 = time.time()
    for i in range(n):
        s = ds[i]
        img = as_img(s["img"]).unsqueeze(0).cuda()
        meta = as_meta(s["img_metas"])
        gt = np.asarray(gt_maps[i]); ori = meta[0]["ori_shape"][:2]
        if gt.shape != tuple(ori):
            import cv2
            gt = cv2.resize(gt.astype(np.uint8), (ori[1], ori[0]),
                            interpolation=cv2.INTER_NEAREST)
        gtf = torch.from_numpy(gt).cuda()
        acc = None
        with torch.no_grad():
            for m, w in models:
                p = m(return_loss=False, rescale=True, img=[img], img_metas=[meta])[0]
                p = np.asarray(p[0] if isinstance(p, tuple) else p, dtype=np.float32)
                acc = p * w if acc is None else acc + p * w
                if args.flip:
                    imf = img.flip(-1)
                    mf = copy.deepcopy(meta)
                    mf[0]["flip"] = True; mf[0]["flip_direction"] = "horizontal"
                    p2 = m(return_loss=False, rescale=True, img=[imf], img_metas=[mf])[0]
                    p2 = np.asarray(p2[0] if isinstance(p2, tuple) else p2, dtype=np.float32)
                    acc = acc + p2 * w
            pf = torch.from_numpy(acc / tot).cuda()
            # D 的 student / reference（MA 通道）
            ps = dm._student_prob(img, meta, True)[0]
            pr = dm._reference_prob(img, meta, True)[0]
            img_f = torch.nn.functional.interpolate(
                img, size=tuple(ori), mode="bilinear", align_corners=False)[0]
            f7 = torch.cat([img_f, logit(pf[3:4]), logit(ps[3:4]),
                            logit(pr[3:4]), (ps[3:4] - pr[3:4])], dim=0).unsqueeze(0)
            delta = head(f7)[0, 0]
            p_new = torch.sigmoid(logit(pf[3:4][0]) + delta)
            pc = torch.cat([pf[:3], p_new.unsqueeze(0)], dim=0)
        for c in range(4):
            for tag, prob in (("fused", pf), ("corrected", pc)):
                p = prob[c] > TAU; g = gtf == (c + 1); ig = gtf == 255
                agg[tag][c][0] += int((p & g).sum())
                agg[tag][c][1] += int((p & ~g & ~ig).sum())
                agg[tag][c][2] += int((~p & g).sum())
        for c in range(3):
            if not torch.equal(pf[c] > TAU, pc[c] > TAU):
                nonma_same = False
        if (i + 1) % 20 == 0 or i + 1 == n:
            print("  %d/%d %.1fs" % (i + 1, n, time.time() - t0), flush=True)

    def miou(tag):
        ious = {}
        for c in range(4):
            tp, fp, fn = agg[tag][c]; den = tp + fp + fn
            ious[CLASSES[c]] = tp / den if den else 0.0
        return float(np.mean(list(ious.values()))), ious

    mf, iouf = miou("fused"); mc, iouc = miou("corrected")
    print("\n[集成 fused]     mIoU=%.4f %s" % (mf, {k: round(v, 4) for k, v in iouf.items()}))
    print("[集成 corrected] mIoU=%.4f %s" % (mc, {k: round(v, 4) for k, v in iouc.items()}))
    print("[delta] mIoU=%+.4f MA=%+.4f  非MA逐位相同=%s"
          % (mc - mf, iouc["MA"] - iouf["MA"], nonma_same))
    out = {"probe": "H5_ensemble_plus_corrector", "split": args.split, "n": n,
           "models": [{"cfg": a, "ckpt": b, "w": c} for a, b, c in specs],
           "flip": bool(args.flip), "fused": {"mIoU": mf, "per_class": iouf},
           "corrected": {"mIoU": mc, "per_class": iouc},
           "delta_mIoU": mc - mf, "delta_MA": iouc["MA"] - iouf["MA"],
           "non_MA_bit_identical": bool(nonma_same),
           "status": "PASS" if (mc > mf and nonma_same) else "FAIL"}
    os.makedirs(os.path.join(args.out_dir, "experiments"), exist_ok=True)
    p = os.path.join(args.out_dir, "experiments", "h5_%s.json" % args.split)
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    os.replace(tmp, p)
    print("状态:", out["status"], " 写出:", p)


if __name__ == "__main__":
    main()
