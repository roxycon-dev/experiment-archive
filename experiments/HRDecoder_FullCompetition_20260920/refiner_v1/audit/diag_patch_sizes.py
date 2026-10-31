#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""diag_patch_sizes — why the eval path saw variable-size patches.

CPU-only forensic read of the frozen patch cache (`index_{split}.json`) plus the
exact geometry the evaluator recomputes (`candidate.window_for`).  Nothing is
recomputed from the champion, nothing is written, no test data is touched.

The point is to show, from the *shipped* artefacts, that

  * the candidate rule deliberately CLIPS tiles at each cluster bbox (so tiles
    are 512x512 only in the interior -- borders and corners are smaller), and
  * the extraction window is clipped again at the image border, so two patches
    of the same image generally have different (h, w).

Run:  python audit/diag_patch_sizes.py --split val
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from refiner import candidate as C  # noqa: E402

CACHE_ROOT = ("/root/autodl-tmp/HRDecoder_FullCompetition_20260920/"
              "refiner_v1/patch_cache")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", choices=["train", "val"])
    ap.add_argument("--cache-root", default=CACHE_ROOT)
    ap.add_argument("--config", default=os.path.join(os.path.dirname(_HERE),
                                                     "configs", "refiner_v1.json"))
    ap.add_argument("--examples", type=int, default=3)
    args = ap.parse_args()

    cfg = json.load(open(os.path.abspath(args.config), encoding="utf-8"))
    ctx = int(cfg["cache"]["context"])
    maxp = int(cfg["cache"]["max_patch"])

    ip = os.path.join(args.cache_root, "index_%s.json" % args.split)
    with open(ip, encoding="utf-8") as f:
        idx = json.load(f)

    full = (C.TILE, C.TILE)
    # candidates only: the eval path never touches background patches
    cand_rect = Counter()       # extraction windows of CANDIDATE patches
    cand_tile = Counter()       # candidate tiles as produced by candidate_tiles
    cand_per_image = []
    n_cand = 0
    for rec in idx["images"]:
        shapes = [(int(h), int(w)) for h, w in rec["shapes"]]
        roles = [int(r) for r in rec["roles"]]
        tiles = rec.get("tiles") or []
        assert len(shapes) == len(roles) == len(tiles), rec["stem"]
        k = 0
        for sh, role, t in zip(shapes, roles, tiles):
            if role == 3:                     # background: training only
                continue
            cand_rect[sh] += 1
            x0, y0, x1, y1 = (int(v) for v in t)
            cand_tile[(y1 - y0, x1 - x0)] += 1
            k += 1
        n_cand += k
        cand_per_image.append(len({s for s, r in zip(shapes, roles) if r != 3}))

    print("[diag] cache        : %s" % ip)
    print("[diag] split=%s images=%d candidate_patches=%d ctx=%d max_patch=%d"
          % (args.split, len(idx["images"]), n_cand, ctx, maxp))
    print("[diag] candidate tiles exactly %dx%d : %d / %d (%.1f%%)"
          % (C.TILE, C.TILE, cand_tile.get(full, 0), n_cand,
             100.0 * cand_tile.get(full, 0) / max(1, n_cand)))
    print("[diag] candidate windows exactly %dx%d : %d / %d (%.1f%%)"
          % (ctx * 2 + C.TILE, ctx * 2 + C.TILE, cand_rect.get(
              (ctx * 2 + C.TILE, ctx * 2 + C.TILE), 0), n_cand,
             100.0 * cand_rect.get((ctx * 2 + C.TILE, ctx * 2 + C.TILE), 0)
             / max(1, n_cand)))
    print("[diag] distinct CANDIDATE window shapes = %d   tile shapes = %d"
          % (len(cand_rect), len(cand_tile)))
    print("[diag] candidate window shapes by count (top 12):")
    for (h, w), n in cand_rect.most_common(12):
        print("        %4dx%-4d  x%d" % (h, w, n))
    print("[diag] per-image distinct CANDIDATE window shapes: min=%d max=%d mean=%.2f"
          % (min(cand_per_image), max(cand_per_image),
             sum(cand_per_image) / max(1, len(cand_per_image))))
    print("[diag] images where >1 candidate window shape occurs: %d / %d"
          % (sum(1 for d in cand_per_image if d > 1), len(cand_per_image)))

    # what np.stack(patches) did: it needs ONE common shape
    print("[diag] => one image mixes %d..%d candidate patch shapes, so the "
          "shipped eval path (np.stack(patches)) raises ValueError: all input "
          "arrays must have the same shape."
          % (min(cand_per_image), max(cand_per_image)))

    if args.examples:
        print("[diag] first images (stem, n_cand, distinct shapes):")
        shown = 0
        for rec in idx["images"]:
            shapes = [(int(h), int(w)) for h, w in rec["shapes"]]
            roles = [int(r) for r in rec["roles"]]
            cs = sorted({s for s, r in zip(shapes, roles) if r != 3})
            if len(cs) > 1:
                print("        %-16s n_cand=%d distinct=%d  %s"
                      % (rec["stem"], sum(1 for r in roles if r != 3),
                         len(cs), cs))
                shown += 1
                if shown >= args.examples:
                    break
    return 0


if __name__ == "__main__":
    sys.exit(main())
