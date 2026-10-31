#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""build_patch_cache — one-time, GPU, frozen-champion patch cache for the refiner.

This is a DATA BUILD, not training.  It runs the frozen champion once per image
and writes only the candidate patches (plus a fixed number of background
patches).  Full-resolution probability maps are never written: an fp32
4-channel map for one 1728x2592 image is 72 MB, and 530 images would be 38 GB.

Storage discipline
------------------
  * probability fp16, RGB uint8, GT uint8 (all as instructed)
  * patches are stored RAGGED (natural size, no padding to a common grid):
    padding to the largest window in an image would have inflated the run to
    ~4 MB/image.  The reader rebuilds offsets from a per-patch `shape` table.
  * every array for a patch is cut from the SAME rectangle, so channel/space
    alignment is structural rather than a convention.
  * the window is the candidate tile grown by PATCH_CONTEXT px of REAL image
    pixels (clipped at the borders).  Zero padding would silently change what
    the model sees at a tile boundary; real context does not.
  * `valid` marks the tile itself = the write-back region; the context ring is
    valid=0, so it never contributes to the loss and is never written back.

Usage
-----
  python build_patch_cache.py --split train            # 454 images
  python build_patch_cache.py --split val              # 76 images
  python build_patch_cache.py --split val --limit 3    # smoke
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

from refiner import candidate as C                                # noqa: E402
from refiner.reference import (FrozenChampion, DEFAULT_DATA_ROOT,  # noqa: E402
                               DEFAULT_REPO)

CACHE_ROOT = ("/root/autodl-tmp/HRDecoder_FullCompetition_20260920/"
              "refiner_v1/patch_cache")
ROLE_CAND_POS = 0
ROLE_CAND_NEG_FP = 1
ROLE_CAND_NEG_PLAIN = 2
ROLE_BACKGROUND = 3


def role_of(gt2, prob2, t, is_candidate):
    if not is_candidate:
        return ROLE_BACKGROUND
    if bool(np.asarray(gt2).max() > 0.5):
        return ROLE_CAND_POS
    if bool((np.asarray(prob2) > 0.5).any()):
        return ROLE_CAND_NEG_FP
    if bool((np.asarray(prob2) > t).any()):
        return ROLE_CAND_NEG_PLAIN
    return ROLE_CAND_NEG_PLAIN


def window_for(tile, H, W, context, max_patch):
    """Delegates to the frozen shared implementation (single source of truth)."""
    return C.window_for(tile, H, W, context, max_patch)


def cut(rgb, prob, gt, tile, win, H, W):
    x0, y0, x1, y1 = tile
    ex0, ey0, ex1, ey1 = win
    r = np.ascontiguousarray(rgb[ey0:ey1, ex0:ex1])
    p = np.ascontiguousarray(prob[:, ey0:ey1, ex0:ex1])
    g = np.ascontiguousarray(gt[:, ey0:ey1, ex0:ex1])
    valid = np.zeros(r.shape[:2], np.uint8)
    valid[y0 - ey0:y1 - ey0, x0 - ex0:x1 - ex0] = 1
    return r, p, g, valid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["train", "val"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--cache-root", default=CACHE_ROOT)
    ap.add_argument("--config", default=os.path.join(_HERE, "configs", "refiner_v1.json"))
    ap.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    args = ap.parse_args()
    # the legacy runtime chdir()s into settings['repo']; resolve everything up
    # front so no cache is ever written into the repo directory (R10)
    args.cache_root = os.path.abspath(args.cache_root)
    args.config = os.path.abspath(args.config)
    args.data_root = os.path.abspath(args.data_root)

    cfg = json.load(open(args.config, encoding="utf-8"))
    t = float(cfg["candidate"]["threshold"])
    dl = int(cfg["candidate"]["dilate"])
    if t not in C.SEED_THRESHOLDS or dl not in C.SEED_DILATIONS:
        raise C.ContractError("config rule (t=%r,d=%r) is outside the pre-declared "
                              "audit grid %r x %r" % (t, dl, C.SEED_THRESHOLDS,
                                                      C.SEED_DILATIONS))
    ctx = int(cfg["cache"]["context"])
    maxp = int(cfg["cache"]["max_patch"])
    n_bg = int(cfg["cache"]["n_background_per_image"])
    bg_tile = int(cfg["cache"]["background_tile"])
    if ctx != C.PATCH_CONTEXT or bg_tile != C.BACKGROUND_TILE:
        raise C.ContractError("config extraction geometry (%d/%d) disagrees with the "
                              "frozen constants (%d/%d) -- edit both or neither"
                              % (ctx, bg_tile, C.PATCH_CONTEXT, C.BACKGROUND_TILE))

    C.check_no_forbidden_paths(args.cache_root, args.split, args.data_root, args.config)
    os.makedirs(os.path.join(args.cache_root, "patch", args.split), exist_ok=True)

    import cv2
    champ = FrozenChampion(repo=DEFAULT_REPO, data_root=args.data_root)
    ds = champ.dataset(args.split)
    n_all = len(ds)
    limit = n_all if args.limit is None else min(args.limit, n_all)
    print("[cache] split=%s n=%d limit=%d rule=(t=%.2f,d=%d,margin=%d,tile=%d) "
          "ctx=%d bg_tile=%d" % (args.split, n_all, limit, t, dl, C.MARGIN,
                                 C.TILE, ctx, bg_tile), flush=True)

    rng = np.random.default_rng(int(cfg["cache"]["seed"]))
    images, tot = [], 0
    st = {"n_tiles": 0, "n_cand": 0, "n_bg": 0, "roles": {str(k): 0 for k in range(4)},
          "gt_ma": 0, "gt_he": 0, "gt_ma_cand": 0, "gt_he_cand": 0,
          "fp_ma_cand": 0, "fp_he_cand": 0, "raw_bytes": 0}
    t0 = time.time()
    for i in range(limit):
        s = ds[i]
        meta = champ.rt._unwrap_meta(s["img_metas"])
        ori = tuple(int(x) for x in meta[0]["ori_shape"][:2])
        stem = os.path.splitext(os.path.basename(meta[0]["ori_filename"]))[0]
        if "test" in stem.lower():
            raise C.ContractError("stem contains 'test': %s" % stem)
        H, W = ori
        img = champ.rt._unwrap_img(s["img"]).unsqueeze(0).cuda()
        p2 = champ.champion_ma_he(img, meta)
        del img
        torch.cuda.empty_cache()

        bgr = cv2.imread(os.path.join(args.data_root, "images", args.split,
                                      stem + ".jpg"), cv2.IMREAD_COLOR)
        lb = cv2.imread(os.path.join(args.data_root, "labels", args.split,
                                     stem + ".png"), cv2.IMREAD_UNCHANGED)
        if bgr is None or lb is None:
            raise C.ContractError("missing raw data for %s" % stem)
        if lb.shape[:2] != ori:
            lb = cv2.resize(lb, (W, H), interpolation=cv2.INTER_NEAREST)
        if bgr.shape[:2] != ori:
            bgr = cv2.resize(bgr, (W, H), interpolation=cv2.INTER_LINEAR)
        rgb = np.ascontiguousarray(bgr[:, :, ::-1])
        gt2 = np.stack([lb == C.GT_VALUE["MA"], lb == C.GT_VALUE["HE"]], 0).astype(np.uint8)

        tiles, boxes = C.candidate_tiles(p2[0], p2[1], t, dl)
        cand_mask = C.mask_from_tiles(tiles, ori)
        st["gt_ma"] += int(gt2[0].sum()); st["gt_he"] += int(gt2[1].sum())
        st["gt_ma_cand"] += _n(gt2[0], cand_mask)
        st["gt_he_cand"] += _n(gt2[1], cand_mask)
        ch = p2 > 0.5
        st["fp_ma_cand"] += _n(ch[0] & ~gt2[0].astype(bool), cand_mask)
        st["fp_he_cand"] += _n(ch[1] & ~gt2[1].astype(bool), cand_mask)

        jobs = [(tl, True) for tl in tiles]
        drawn = 0
        tries = 0
        while drawn < n_bg and tries < 80 and H > bg_tile and W > bg_tile:
            tries += 1
            gy = int(rng.integers(0, H - bg_tile + 1))
            gx = int(rng.integers(0, W - bg_tile + 1))
            if gt2[:, gy:gy + bg_tile, gx:gx + bg_tile].any():
                continue
            if (p2[:, gy:gy + bg_tile, gx:gx + bg_tile] > t).any():
                continue
            jobs.append(((gx, gy, gx + bg_tile, gy + bg_tile), False))
            drawn += 1

        recs = []
        for tile, is_cand in jobs:
            win = window_for(tile, H, W, ctx, maxp)
            r, p, g, valid = cut(rgb, p2, gt2, tile, win, H, W)
            recs.append({"tile": tile, "rect": win, "rgb": r,
                         "prob": p.astype(np.float16), "gt": g, "valid": valid,
                         "role": role_of(g, p, t, is_cand)})
        if not recs:
            images.append({"stem": stem, "n": 0, "n_cand": 0, "n_bg": 0,
                           "npz_bytes": 0, "roles": [], "shapes": []})
            print("[cache] %d/%d %s: 0 patches" % (i + 1, limit, stem), flush=True)
            continue

        N = len(recs)
        shapes = np.array([[r["rgb"].shape[0], r["rgb"].shape[1]] for r in recs], np.int32)
        rgb_flat = np.concatenate([r["rgb"].transpose(2, 0, 1).ravel() for r in recs])
        prob_flat = np.concatenate([r["prob"].ravel() for r in recs])
        gt_flat = np.concatenate([r["gt"].ravel() for r in recs])
        valid_flat = np.concatenate([r["valid"].ravel() for r in recs])
        out_p = os.path.join(args.cache_root, "patch", args.split, stem + ".npz")
        np.savez_compressed(
            out_p,
            tiles=np.array([r["tile"] for r in recs], np.int32),
            rects=np.array([r["rect"] for r in recs], np.int32),
            shape=shapes,
            role=np.array([r["role"] for r in recs], np.uint8),
            rgb=rgb_flat, prob=prob_flat, gt=gt_flat, valid=valid_flat)
        raw = int(rgb_flat.nbytes + prob_flat.nbytes + gt_flat.nbytes + valid_flat.nbytes)
        sz = os.path.getsize(out_p)
        n_cand = sum(1 for r in recs if r["role"] != ROLE_BACKGROUND)
        st["n_tiles"] += N; st["n_cand"] += n_cand; st["n_bg"] += N - n_cand
        st["raw_bytes"] += raw
        for r in recs:
            st["roles"][str(int(r["role"]))] += 1
        images.append({"stem": stem, "n": int(N), "n_cand": int(n_cand),
                       "n_bg": int(N - n_cand), "npz_bytes": int(sz),
                       "raw_bytes": int(raw),
                       "roles": [int(x) for x in np.array([r["role"] for r in recs])],
                       "shapes": shapes.tolist(),
                       "tiles": [[int(v) for v in r["tile"]] for r in recs]})
        tot += sz
        del r, p, g, valid, recs, rgb_flat, prob_flat, gt_flat, valid_flat
        del rgb, p2, bgr, lb, gt2, cand_mask
        if (i + 1) % 10 == 0 or i + 1 == limit:
            print("[cache] %d/%d %.1fs tiles=%d cand=%d bg=%d bytes=%.1fMB"
                  % (i + 1, limit, time.time() - t0, st["n_tiles"], st["n_cand"],
                     st["n_bg"], tot / 1e6), flush=True)

    index = {
        "schema": "hrdecoder-refiner-patch-cache/v1",
        "split": args.split,
        "n_images": len(images),
        "complete": bool(limit == n_all),
        "data_root": args.data_root,
        "champion_settings": champ.settings_path,
        "champion": {"members": ["D x3", "E x2", "B x1", "E4 x1"], "tta": "hflip",
                     "head": "H4 (MA only)",
                     "val_reference_miou": 0.42669428454584546,
                     "val_reference_ma_iou": 0.259672,
                     "val_reference_he_iou": 0.437583},
        "candidate": {"threshold": t, "dilate": dl, "margin": C.MARGIN,
                      "tile": C.TILE, "tile_stride": C.TILE_STRIDE,
                      "structuring_element": "rect"},
        "extraction": {"context": ctx, "max_patch": maxp,
                       "background_tile": bg_tile, "padding": "none (ragged)"},
        "sampling": {"n_background_per_image": n_bg,
                     "all_candidate_tiles_kept": True},
        "storage": {"prob_dtype": "float16", "rgb_dtype": "uint8",
                    "gt_dtype": "uint8", "compression": "npz deflate",
                    "layout": "flat per image; offsets derived from `shape`",
                    "total_npz_bytes": int(tot), "total_raw_bytes": int(st["raw_bytes"]),
                    "compression_ratio": float(st["raw_bytes"]) / max(1, tot)},
        "counts": {"n_tiles": st["n_tiles"], "n_candidate_tiles": st["n_cand"],
                   "n_background_tiles": st["n_bg"], "roles": st["roles"],
                   "gt_ma_px": st["gt_ma"], "gt_he_px": st["gt_he"],
                   "gt_ma_px_in_cand": st["gt_ma_cand"],
                   "gt_he_px_in_cand": st["gt_he_cand"],
                   "ma_recall_upper_bound": st["gt_ma_cand"] / max(1, st["gt_ma"]),
                   "he_recall_upper_bound": st["gt_he_cand"] / max(1, st["gt_he"]),
                   "champ_fp_ma_in_cand": st["fp_ma_cand"],
                   "champ_fp_he_in_cand": st["fp_he_cand"]},
        "images": images,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "elapsed_s": time.time() - t0,
    }
    ip = os.path.join(args.cache_root, "index_%s.json" % args.split)
    with open(ip, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)
    print("[cache] wrote %s tiles=%d cand=%d bg=%d bytes=%.2fMB ratio=%.1fx "
          "MA_recall_ub=%.4f HE_recall_ub=%.4f"
          % (ip, st["n_tiles"], st["n_cand"], st["n_bg"], tot / 1e6,
             index["storage"]["compression_ratio"],
             index["counts"]["ma_recall_upper_bound"],
             index["counts"]["he_recall_upper_bound"]))
    return 0


def _n(a, mask):
    return int(np.count_nonzero(np.asarray(a, bool) & np.asarray(mask, bool)))


if __name__ == "__main__":
    sys.exit(main())
