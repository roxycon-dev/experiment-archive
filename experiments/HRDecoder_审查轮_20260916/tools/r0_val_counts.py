#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""R0：生成配对 bootstrap 所需的逐图逐类计数（BASE_0 vs H4-corrected）。

严格按 2026-09-17 契约 §5.2：
  - 两个被比较对象必须在**同一完整 76 张 val** 上产出的逐图、逐类 TP/FP/FN
  - BASE_0 = CHAMPION_0 在 MA 修正之前的固定集成/TTA 管线
  - H4-corrected = 在同一 BASE_0 上施加 h4_best.pth
  - 非 MA 三类在两者中完全相同（结构隔离），也要写出以便核对
  - 患者 ID 不可用 → 图像级重采样，并在输出中标注未处理患者相关性

输出：diagnostics/val_per_image_counts_R0.csv
"""
import argparse
import copy
import csv
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


def logit(p, eps=1e-4):
    q = p.clamp(eps, 1 - eps)
    return torch.log(q / (1 - q))


class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(7, 16, 3, 1, 1), nn.ReLU(inplace=True),
                                 nn.Conv2d(16, 16, 3, 1, 1), nn.ReLU(inplace=True),
                                 nn.Conv2d(16, 1, 1))

    def forward(self, f):
        return 6.0 * torch.tanh(self.net(f))


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
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--head", required=True)
    ap.add_argument("--models", required=True)
    ap.add_argument("--flip", action="store_true")
    args = ap.parse_args()

    from mmcv import Config
    from mmseg.datasets import build_dataset
    from mmseg.models import build_segmentor
    from mmcv.runner import load_checkpoint

    base = Config.fromfile("configs/lesion/exp/hrplus_patch_dice.py")
    d = copy.deepcopy(base.data["val"]); d.test_mode = True
    ds = build_dataset(d)
    gt_maps = ds.get_gt_seg_maps()
    n = len(ds)
    print("val =", n, flush=True)

    specs = []
    for sp in args.models.split(";"):
        sp = sp.strip()
        if not sp:
            continue
        cfgp, rest = sp.split("=", 1)
        ck, w = (rest.rsplit(":", 1) + ["1.0"])[:2] if ":" in rest else (rest, "1.0")
        specs.append((cfgp, ck, float(w)))
    members = []
    for cfgp, ck, w in specs:
        c = Config.fromfile(cfgp)
        m = build_segmentor(c.model)
        load_checkpoint(m, ck, map_location="cpu", strict=False)
        members.append((m.cuda().eval(), w))
    flip = args.flip
    tot = sum(w for _, w in members) * (2 if flip else 1)
    print("BASE_0 成员权重 =", [s[2] for s in specs], " flip =", flip, flush=True)

    dm = build_segmentor(base.model)
    load_checkpoint(dm, "/root/autodl-tmp/hrplus/D_0913_retry/best_mIoU_iter_2000.pth",
                    map_location="cpu", strict=False)
    dm = dm.cuda().eval()

    head = Head().cuda()
    ck = torch.load(args.head, map_location="cpu")
    head.load_state_dict(ck["head"] if isinstance(ck, dict) and "head" in ck else ck)
    head.eval()
    print("H4 头载入 step =", ck.get("step") if isinstance(ck, dict) else "?", flush=True)

    rows = []
    agg = {"base0": {c: [0, 0, 0] for c in range(4)},
           "h4": {c: [0, 0, 0] for c in range(4)}}
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
            for m, w in members:
                p = m(return_loss=False, rescale=True, img=[img], img_metas=[meta])[0]
                p = np.asarray(p[0] if isinstance(p, tuple) else p, dtype=np.float32)
                acc = p * w if acc is None else acc + p * w
                if flip:
                    imf = img.flip(-1)
                    mf = copy.deepcopy(meta)
                    mf[0]["flip"] = True; mf[0]["flip_direction"] = "horizontal"
                    p2 = m(return_loss=False, rescale=True, img=[imf], img_metas=[mf])[0]
                    p2 = np.asarray(p2[0] if isinstance(p2, tuple) else p2, dtype=np.float32)
                    acc = acc + p2 * w
            pf = torch.from_numpy(acc / tot).cuda()          # BASE_0（4 通道概率）
            ps = dm._student_prob(img, meta, True)[0]
            pr = dm._reference_prob(img, meta, True)[0]
            img_f = torch.nn.functional.interpolate(
                img, size=tuple(ori), mode="bilinear", align_corners=False)[0]
            f7 = torch.cat([img_f, logit(pf[3:4]), logit(ps[3:4]),
                            logit(pr[3:4]), (ps[3:4] - pr[3:4])], dim=0).unsqueeze(0)
            pnew = torch.sigmoid(logit(pf[3:4][0]) + head(f7)[0, 0])
            ph4 = torch.cat([pf[:3], pnew.unsqueeze(0)], dim=0)

        row = {"idx": i, "filename": meta[0].get("ori_filename", ""),
               "h": int(gt.shape[0]), "w": int(gt.shape[1]),
               "ignore_px": int((gt == 255).sum())}
        for tag, prob in (("base0", pf), ("h4", ph4)):
            for c in range(4):
                p = prob[c] > TAU; g = gtf == (c + 1); ig = gtf == 255
                tp = int((p & g & ~ig).sum()); fp = int((p & ~g & ~ig).sum())
                fn = int((~p & g).sum())
                agg[tag][c][0] += tp; agg[tag][c][1] += fp; agg[tag][c][2] += fn
                row["%s_%s_TP" % (tag, CLASSES[c])] = tp
                row["%s_%s_FP" % (tag, CLASSES[c])] = fp
                row["%s_%s_FN" % (tag, CLASSES[c])] = fn
        for c in range(3):
            if not torch.equal(pf[c] > TAU, ph4[c] > TAU):
                nonma_same = False
        rows.append(row)
        if (i + 1) % 20 == 0 or i + 1 == n:
            print("  %d/%d %.1fs" % (i + 1, n, time.time() - t0), flush=True)

    def miou(tag):
        o = {}
        for c in range(4):
            tp, fp, fn = agg[tag][c]; den = tp + fp + fn
            o[CLASSES[c]] = tp / den if den else 0.0
        return float(np.mean(list(o.values()))), o

    mb, ib = miou("base0"); mh, ih = miou("h4")
    print("\n[BASE_0] mIoU=%.4f %s" % (mb, {k: round(v, 4) for k, v in ib.items()}))
    print("[H4]     mIoU=%.4f %s" % (mh, {k: round(v, 4) for k, v in ih.items()}))
    print("[delta]  mIoU=%+.4f  MA=%+.4f  非MA逐位相同=%s"
          % (mh - mb, ih["MA"] - ib["MA"], nonma_same))

    od = args.out_dir
    os.makedirs(os.path.join(od, "diagnostics"), exist_ok=True)
    cp = os.path.join(od, "diagnostics", "val_per_image_counts_R0.csv")
    with open(cp, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    meta_out = {
        "purpose": "R0 paired bootstrap counts",
        "n_val": n, "tau": TAU, "comparison": "strict_>",
        "aggregation": "dataset_cumulative",
        "base0_identity": {"members": [{"cfg": a, "ckpt": b, "w": c} for a, b, c in specs],
                           "flip_tta": flip, "forward_passes_per_image": len(specs) * (2 if flip else 1)},
        "h4_head": args.head, "h4_step": (ck.get("step") if isinstance(ck, dict) else None),
        "student_ref_source": "D@2000 (hrplus_patch_dice) 的 _student_prob / _reference_prob",
        "feature_order": ["R","G","B","logit(P_BASE0_MA)","logit(P_student_MA)",
                          "logit(P_reference_MA)","P_student_MA-P_reference_MA"],
        "logit_clamp": "clamp(1e-4, 1-1e-4)  ← 旧 H4 实际实现；契约要求 1e-6 的差异已登记 DEVIATIONS",
        "aggregate": {"base0": {"mIoU": mb, "per_class": ib},
                      "h4": {"mIoU": mh, "per_class": ih},
                      "delta_mIoU": mh - mb, "delta_MA": ih["MA"] - ib["MA"]},
        "non_MA_bit_identical": bool(nonma_same),
        "patient_id": "UNAVAILABLE -> 图像级重采样，未处理患者相关性",
        "status": "PASS" if nonma_same else "FAIL",
    }
    p = os.path.join(od, "diagnostics", "R0_counts_meta.json")
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(meta_out, f, indent=2, ensure_ascii=False)
    os.replace(tmp, p)
    print("\n状态:", meta_out["status"])
    print("写出:", cp)
    print("写出:", p)


if __name__ == "__main__":
    main()
