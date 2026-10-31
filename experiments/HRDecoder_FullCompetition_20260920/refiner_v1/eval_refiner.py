#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""eval_refiner — val (or train) evaluation of the MA/HE refiner on FULL images.

The reported quantity is per-class IoU over the whole image, so the evaluation
reproduces the real inference path end to end:

    frozen champion -> frozen candidate rule -> refiner inside candidates only
    -> fixed weighted overlap average -> fixed p > 0.5 -> global IoU

Candidate patches are RAGGED (tiles are clipped to their cluster bbox and
windows to the image border), so they are refined per shape bucket with no
padding at all -- see `refiner/tiled_infer.py`.  Stacking them into one array
is impossible, and padding them to a common grid would change the GroupNorm
statistics of every padded sample.

Self-check (fail closed): the champion baseline recomputed here must match the
frozen contract numbers (MA 0.259672, HE 0.437583, mIoU 0.426694) to 1e-6, and
the refiner must leave EX/SE and every non-candidate pixel bitwise untouched.
If either check fails the run is rejected instead of reported.

Usage
-----
  python eval_refiner.py --limit 2 --no-refine      # champion baseline smoke
  python eval_refiner.py --ckpt runs/refiner_best.pth
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if os.path.dirname(_HERE) not in sys.path:
    sys.path.insert(0, os.path.dirname(_HERE))

from refiner import assemble as A                          # noqa: E402
from refiner import candidate as C                          # noqa: E402
from refiner import guards as G                             # noqa: E402
from refiner import tiled_infer as TI                       # noqa: E402
from refiner.model import HaarHeteroRefiner                 # noqa: E402
from refiner.reference import FrozenChampion, DEFAULT_DATA_ROOT, DEFAULT_REPO  # noqa: E402

CONTRACT = {"mIoU": 0.42669428454584546, "MA": 0.259672, "HE": 0.437583}
TOL = 1e-6


def iou(tp, fp, fn):
    d = tp + fp + fn
    return float(tp) / float(d) if d > 0 else None


def counts_for(prob4, gt):
    out = {}
    for c in C.CLASSES:
        b = prob4[C.CLS_INDEX[c]] > A.THRESHOLD
        g = gt[c]
        out[c] = [int(np.count_nonzero(b & g)), int(np.count_nonzero(b & ~g)),
                  int(np.count_nonzero(~b & g))]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(_HERE, "configs", "refiner_v1.json"))
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--split", default="val", choices=["train", "val"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    ap.add_argument("--out", default=os.path.join(_HERE, "logs", "eval_refiner.json"))
    ap.add_argument("--no-refine", action="store_true")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    # the legacy runtime chdir()s into settings['repo']; make every path that
    # this script writes to absolute BEFORE the runtime is imported (R10)
    args.out = os.path.abspath(args.out)
    args.config = os.path.abspath(args.config)
    args.data_root = os.path.abspath(args.data_root)

    cfg = json.load(open(args.config, encoding="utf-8"))
    G.assert_val_only_inference(args.split, args.out, args.config, args.data_root)
    if args.ckpt:
        G.assert_no_test_paths(args.ckpt)

    import cv2
    t = float(cfg["candidate"]["threshold"])
    dl = int(cfg["candidate"]["dilate"])
    ctx = int(cfg["cache"]["context"])
    maxp = int(cfg["cache"]["max_patch"])
    scale = float(cfg["model"]["residual_scale"])

    champ = FrozenChampion(repo=DEFAULT_REPO, data_root=args.data_root,
                           device=args.device)
    ds = champ.dataset(args.split)
    n_all = len(ds)
    limit = n_all if args.limit is None else min(args.limit, n_all)

    refiner = None
    if not args.no_refine:
        refiner = HaarHeteroRefiner(width=int(cfg["model"]["width"]),
                                    residual_scale=scale)
        if args.ckpt:
            ck = torch.load(args.ckpt, map_location="cpu", weights_only=True)
            state = ck.get("model", ck)
            refiner.load_state_dict(state, strict=True)
            print("[eval] loaded %s (stage=%s step=%s)"
                  % (args.ckpt, ck.get("stage"), ck.get("step")))
        refiner = refiner.to(args.device).eval()
        for p in refiner.parameters():
            p.requires_grad_(False)

    cnt_champ = {c: [0, 0, 0] for c in C.CLASSES}
    cnt_ref = {c: [0, 0, 0] for c in C.CLASSES}
    n_tiles_total = 0
    checks = {"ex_se_changed": 0, "outside_cand_changed": 0, "n_pixels": 0,
              "n_candidate_px": 0, "images": 0, "nonfinite_refined": 0}
    pstats = {"mode": "shape-bucketed exact windows (no padding)",
              "images_with_tiles": 0, "n_buckets_total": 0,
              "max_bucket_size": 0, "max_distinct_shapes_per_image": 0,
              "max_window": 0, "n_forward_total": 0, "batch_cap": args.batch}
    t0 = time.time()
    for i in range(limit):
        s = ds[i]
        meta = champ.rt._unwrap_meta(s["img_metas"])
        ori = tuple(int(x) for x in meta[0]["ori_shape"][:2])
        stem = os.path.splitext(os.path.basename(meta[0]["ori_filename"]))[0]
        H, W = ori
        img = champ.rt._unwrap_img(s["img"]).unsqueeze(0).to(args.device)
        p4 = champ.champion_prob4(img, meta)
        del img
        torch.cuda.empty_cache()

        lb = cv2.imread(os.path.join(args.data_root, "labels", args.split,
                                     stem + ".png"), cv2.IMREAD_UNCHANGED)
        if lb is None:
            raise C.ContractError("missing GT for %s" % stem)
        if lb.shape[:2] != ori:
            lb = cv2.resize(lb, (W, H), interpolation=cv2.INTER_NEAREST)
        gt = {c: (lb == C.GT_VALUE[c]) for c in C.CLASSES}

        tiles, _boxes = C.candidate_tiles(p4[C.CLS_INDEX["MA"]],
                                          p4[C.CLS_INDEX["HE"]], t, dl)
        n_tiles_total += len(tiles)
        checks["n_pixels"] += H * W
        checks["n_candidate_px"] += int(C.mask_from_tiles(tiles, ori).sum()) \
            if tiles else 0

        c_champ = counts_for(p4, gt)
        for c in C.CLASSES:
            for k in range(3):
                cnt_champ[c][k] += c_champ[c][k]

        if refiner is not None and tiles:
            bgr = cv2.imread(os.path.join(args.data_root, "images", args.split,
                                          stem + ".jpg"), cv2.IMREAD_COLOR)
            if bgr.shape[:2] != ori:
                bgr = cv2.resize(bgr, (W, H), interpolation=cv2.INTER_LINEAR)
            rgb_full = np.ascontiguousarray(bgr[:, :, ::-1])
            ma_he = np.ascontiguousarray(np.stack(
                [p4[C.CLS_INDEX["MA"]], p4[C.CLS_INDEX["HE"]]], 0))
            # patches are RAGGED (tiles are clipped to their cluster bbox and
            # windows to the image border), so they are produced per shape
            # bucket -- never stacked into one padded array, which would change
            # the GroupNorm statistics of the samples that got padded.
            patches, _wins, st = TI.refine_tiles(
                refiner, rgb_full, ma_he, tiles, H, W, context=ctx,
                max_patch=maxp, batch=args.batch, device=args.device)
            refined = A.assemble(p4, tiles, patches)
            rep = A.assert_bitwise_unchanged(refined, p4, tiles)
            for c in ("EX", "SE"):
                checks["ex_se_changed"] += rep["%s_total_changed" % c]
            checks["outside_cand_changed"] += sum(
                rep["%s_changed_outside" % c] for c in C.CLASSES)
            for c in C.REFINED_CLASSES:
                checks["nonfinite_refined"] += int(np.count_nonzero(
                    ~np.isfinite(refined[C.CLS_INDEX[c]])))
            pstats["images_with_tiles"] += 1
            pstats["n_buckets_total"] += st["n_buckets"]
            pstats["n_forward_total"] += st["n_forward"]
            pstats["max_bucket_size"] = max(pstats["max_bucket_size"],
                                            st["max_bucket_size"])
            pstats["max_window"] = max(pstats["max_window"], st["max_window"])
            pstats["max_distinct_shapes_per_image"] = max(
                pstats["max_distinct_shapes_per_image"], st["n_distinct_shapes"])
            del rgb_full, ma_he, patches
        else:
            refined = p4

        c_ref = counts_for(refined, gt)
        for c in C.CLASSES:
            for k in range(3):
                cnt_ref[c][k] += c_ref[c][k]
        checks["images"] += 1
        del p4, refined, lb, gt
        torch.cuda.empty_cache()
        if (i + 1) % 10 == 0 or i + 1 == limit:
            print("[eval] %d/%d %.1fs" % (i + 1, limit, time.time() - t0), flush=True)

    def summarize(cnt):
        per = {c: iou(*cnt[c]) for c in C.CLASSES}
        vals = [v for v in per.values() if v is not None]
        return per, float(np.mean(vals))

    per_c, miou_c = summarize(cnt_champ)
    per_r, miou_r = summarize(cnt_ref)
    ok = True
    report = {
        "schema": "hrdecoder-refiner-eval/v1", "split": args.split,
        "n_images": limit, "n_candidate_tiles": n_tiles_total,
        "candidate": {"threshold": t, "dilate": dl, "margin": C.MARGIN},
        "ckpt": args.ckpt, "refiner_applied": bool(refiner is not None),
        "champion": {"mIoU": miou_c, "per_class": per_c, "counts": cnt_champ},
        "refined": {"mIoU": miou_r, "per_class": per_r, "counts": cnt_ref},
        "delta": {"mIoU": miou_r - miou_c,
                  "per_class": {c: (None if (per_r[c] is None or per_c[c] is None)
                                    else per_r[c] - per_c[c]) for c in C.CLASSES}},
        "bitwise_checks": checks, "patch_protocol": pstats,
        "elapsed_s": time.time() - t0,
    }
    if args.split == "val" and limit == 76:
        d_ma = abs(per_c["MA"] - CONTRACT["MA"])
        d_he = abs(per_c["HE"] - CONTRACT["HE"])
        d_mi = abs(miou_c - CONTRACT["mIoU"])
        report["contract_selfcheck"] = {
            "expected": CONTRACT, "observed": {"mIoU": miou_c, "MA": per_c["MA"],
                                               "HE": per_c["HE"]},
            "abs_diff": {"mIoU": d_mi, "MA": d_ma, "HE": d_he},
            "pass": bool(d_mi < TOL and d_ma < TOL and d_he < TOL)}
        ok = report["contract_selfcheck"]["pass"]
    report["accepted"] = bool(ok and checks["ex_se_changed"] == 0
                              and checks["outside_cand_changed"] == 0
                              and checks["nonfinite_refined"] == 0)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(report, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("[eval] champion mIoU=%.6f (MA %.6f HE %.6f)  refined mIoU=%.6f (MA %s HE %s)"
          % (miou_c, per_c["MA"], per_c["HE"], miou_r, per_r["MA"], per_r["HE"]))
    print("[eval] bitwise: EX/SE changed=%d  outside-candidate changed=%d  "
          "non-finite refined=%d  accepted=%s"
          % (checks["ex_se_changed"], checks["outside_cand_changed"],
             checks["nonfinite_refined"], report["accepted"]))
    print("[eval] patches: %s  tiles=%d images_with_tiles=%d buckets=%d "
          "max_bucket=%d max_shapes/image=%d max_window=%d forwards=%d"
          % (pstats["mode"], n_tiles_total, pstats["images_with_tiles"],
             pstats["n_buckets_total"], pstats["max_bucket_size"],
             pstats["max_distinct_shapes_per_image"], pstats["max_window"],
             pstats["n_forward_total"]))
    print("[eval] wrote %s" % args.out)
    return 0 if report["accepted"] else 2


if __name__ == "__main__":
    sys.exit(main())
