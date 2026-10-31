#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""H7：把多个类别校正头一起接到集成输出上（最终方案）。

对每个类别 c ∈ {MA, HE, SE}：用该类别的头修正融合概率
    P_new_c = sigmoid( logit(P_fused_c) + 6*tanh(head_c(feat_7)) )
EX 不修正（已是最强类，无头）。
特征 = [RGB, logit(P_fused_c), logit(P_s_c), logit(P_r_c), P_s_c - P_r_c]

协议冻结：阈值 0.5、严格 >、全数据集累计、背景不计入。
正确性断言：未接头的类别必须逐位不变。
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
IDX = {"EX": 0, "HE": 1, "SE": 2, "MA": 3}


class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(7, 16, 3, 1, 1), nn.ReLU(inplace=True),
                                 nn.Conv2d(16, 16, 3, 1, 1), nn.ReLU(inplace=True),
                                 nn.Conv2d(16, 1, 1))

    def forward(self, f):
        return 6.0 * torch.tanh(self.net(f))


def logit(p, eps=1e-4):
    q = p.clamp(eps, 1 - eps)
    return torch.log(q / (1 - q))


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
    ap.add_argument("--split", default="test")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--heads", required=True,
                    help="如 MA=/path/a.pth,HE=/path/b.pth,SE=/path/c.pth")
    ap.add_argument("--models", required=True)
    ap.add_argument("--flip", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    from mmcv import Config
    from mmseg.datasets import build_dataset
    from mmseg.models import build_segmentor
    from mmcv.runner import load_checkpoint

    # 类别 -> 头
    heads = {}
    for item in args.heads.split(","):
        item = item.strip()
        if not item:
            continue
        cname, hp = item.split("=", 1)
        if not os.path.isfile(hp):
            print("!! 头文件不存在，跳过:", cname, hp, flush=True)
            continue
        h = Head().cuda()
        ck = torch.load(hp, map_location="cpu")
        h.load_state_dict(ck["head"] if isinstance(ck, dict) and "head" in ck else ck)
        h.eval()
        for p in h.parameters():
            p.requires_grad_(False)
        heads[cname] = (h, ck.get("step") if isinstance(ck, dict) else None)
    print("生效的头 =", {k: v[1] for k, v in heads.items()}, flush=True)

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
    models = []
    for cfgp, ck, w in specs:
        c = Config.fromfile(cfgp)
        m = build_segmentor(c.model)
        load_checkpoint(m, ck, map_location="cpu", strict=False)
        models.append((m.cuda().eval(), w))
    tot = sum(w for _, w in models) * (2 if args.flip else 1)
    print("成员 =", [s[2] for s in specs], " split =", args.split, " n =", n, flush=True)

    dm = build_segmentor(base.model)
    load_checkpoint(dm, "/root/autodl-tmp/hrplus/D_0913_retry/best_mIoU_iter_2000.pth",
                    map_location="cpu", strict=False)
    dm = dm.cuda().eval()

    agg = {"fused": {c: [0, 0, 0] for c in range(4)},
           "final": {c: [0, 0, 0] for c in range(4)}}
    nohead_same = True
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
            ps = dm._student_prob(img, meta, True)[0]
            pr = dm._reference_prob(img, meta, True)[0]
            img_f = torch.nn.functional.interpolate(
                img, size=tuple(ori), mode="bilinear", align_corners=False)[0]
            chans = []
            for c in range(4):
                cname = CLASSES[c]
                if cname in heads:
                    h, _ = heads[cname]
                    f7 = torch.cat([img_f, logit(pf[c:c+1]), logit(ps[c:c+1]),
                                    logit(pr[c:c+1]), (ps[c:c+1] - pr[c:c+1])],
                                   dim=0).unsqueeze(0)
                    newp = torch.sigmoid(logit(pf[c:c+1][0]) + h(f7)[0, 0])
                    chans.append(newp.unsqueeze(0))
                else:
                    chans.append(pf[c:c+1])
            pfin = torch.cat(chans, dim=0)      # (4,H,W)
        for c in range(4):
            for tag, prob in (("fused", pf), ("final", pfin)):
                p = prob[c] > TAU; g = gtf == (c + 1); ig = gtf == 255
                agg[tag][c][0] += int((p & g).sum())
                agg[tag][c][1] += int((p & ~g & ~ig).sum())
                agg[tag][c][2] += int((~p & g).sum())
        for c in range(4):
            if CLASSES[c] not in heads:
                if not torch.equal(pf[c] > TAU, pfin[c] > TAU):
                    nohead_same = False
        if (i + 1) % 25 == 0 or i + 1 == n:
            print("  %d/%d %.1fs" % (i + 1, n, time.time() - t0), flush=True)

    def miou(tag):
        ious = {}
        for c in range(4):
            tp, fp, fn = agg[tag][c]; den = tp + fp + fn
            ious[CLASSES[c]] = tp / den if den else 0.0
        return float(np.mean(list(ious.values()))), ious

    mf, iouf = miou("fused"); mc, iouc = miou("final")
    print("\n[fused] mIoU=%.4f  %s" % (mf, {k: round(v, 4) for k, v in iouf.items()}))
    print("[final] mIoU=%.4f  %s" % (mc, {k: round(v, 4) for k, v in iouc.items()}))
    print("[delta] mIoU=%+.4f  %s" % (mc - mf,
          {k: round(iouc[k] - iouf[k], 4) for k in CLASSES}))
    print("[未接头类别逐位不变] =", nohead_same)
    out = {"probe": "H7_all_heads", "split": args.split, "n": n,
           "heads": {k: v[1] for k, v in heads.items()},
           "models": [{"cfg": a, "ckpt": b, "w": c} for a, b, c in specs],
           "flip": bool(args.flip),
           "fused": {"mIoU": mf, "per_class": iouf,
                     "counts": {CLASSES[c]: dict(zip(("TP", "FP", "FN"), agg["fused"][c]))
                                for c in range(4)}},
           "final": {"mIoU": mc, "per_class": iouc,
                     "counts": {CLASSES[c]: dict(zip(("TP", "FP", "FN"), agg["final"][c]))
                                for c in range(4)}},
           "delta_mIoU": mc - mf,
           "no_head_classes_bit_identical": bool(nohead_same),
           "status": "PASS" if (mc > mf and nohead_same) else "FAIL"}
    p = os.path.join(args.out_dir, "experiments", "h7_%s.json" % args.split)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    os.replace(tmp, p)
    print("\n状态:", out["status"], " 写出:", p)


if __name__ == "__main__":
    main()
