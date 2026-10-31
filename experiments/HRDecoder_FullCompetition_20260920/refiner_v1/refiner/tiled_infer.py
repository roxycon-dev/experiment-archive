#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tiled write-back inference: champion -> ragged per-tile refined patches.

This is the ONLY place that turns a list of candidate tiles into refiner
outputs, so the evaluator and any future inference entry point share one
geometry.

Why it is not a plain `np.stack`:
  candidate tiles are clipped to their 8-connected cluster bbox (a cluster is
  usually a small lesion blob, not a 512 block) and the extraction window is
  clipped again at the image border.  Measured on the frozen val cache: only
  34 of 1599 candidate patches are full 512x512, there are 892 distinct
  candidate window shapes, and every one of the 76 val images mixes 2..51 of
  them.  So patches are RAGGED and must never be stacked into one array.

Protocol (fixed, no padding anywhere):
  1. window per tile       = candidate.window_for(tile, H, W, ctx, max_patch)
  2. group the windows by their exact (h, w)
  3. inside a group, forward the refiner on batches of identical-size tensors
     cut from REAL image pixels (candidate.extract_exact_batch)
  4. write each output back at its own tile offset (x0-ex0, y0-ey0) inside its
     own window, producing one (2, th, tw) patch per tile in tile order

Step 3 means the value the model produces for a patch does not depend on which
other patches happened to be in the same forward pass, and no zero padding can
reach a GroupNorm statistic.
"""
from __future__ import annotations

import numpy as np
import torch

from . import candidate as C


def refine_tiles(model, rgb_full: np.ndarray, ma_he: np.ndarray, tiles, H: int,
                 W: int, *, context: int = C.PATCH_CONTEXT,
                 max_patch: int = 768, batch: int = 8, device="cuda"):
    """Refine every candidate tile of one image.

    model    : HaarHeteroRefiner in eval() mode (parameters frozen by caller)
    rgb_full : (H, W, 3) uint8 full-resolution RGB image
    ma_he    : (2, H, W) float32 champion (MA, HE) at full resolution
    tiles    : list of (x0, y0, x1, y1) in ORIGINAL image coordinates

    returns  : (patches, windows, stats)
               patches -- ragged list, patches[i] is (2, y1-y0, x1-x0) float32,
                          aligned one-to-one with `tiles`
               windows -- windows[i] is the extraction rectangle used
               stats   -- bucketing report for the run log / eval report
    """
    tiles = [tuple(int(v) for v in t) for t in tiles]
    wins = C.windows_for(tiles, H, W, context, max_patch)
    out = [None] * len(tiles)
    batch = max(1, int(batch))
    stats = {"mode": "shape-bucketed exact windows (no padding)",
             "n_tiles": len(tiles), "n_buckets": 0, "n_forward": 0,
             "max_bucket_size": 0, "batch_cap": batch,
             "max_window": 0, "n_distinct_shapes": 0}
    buckets = C.bucket_windows_by_shape(wins)
    stats["n_buckets"] = len(buckets)
    stats["n_distinct_shapes"] = len(buckets)
    for (h, w), idxs in buckets:
        stats["max_bucket_size"] = max(stats["max_bucket_size"], len(idxs))
        stats["max_window"] = max(stats["max_window"], h, w)
        for j0 in range(0, len(idxs), batch):
            sel = idxs[j0:j0 + batch]
            br, bp = C.extract_exact_batch(rgb_full, ma_he, wins, sel)
            with torch.no_grad():
                _r, ref = model.forward_refine(
                    torch.from_numpy(br).to(device), torch.from_numpy(bp).to(device))
            ref = ref.detach().to("cpu").numpy()
            stats["n_forward"] += 1
            for k, i in enumerate(sel):
                x0, y0, x1, y1 = tiles[i]
                ex0, ey0, _ex1, _ey1 = wins[i]
                oy, ox = y0 - ey0, x0 - ex0
                sub = ref[k][:, oy:oy + (y1 - y0), ox:ox + (x1 - x0)]
                if sub.shape != (len(C.REFINED_CLASSES), y1 - y0, x1 - x0):
                    raise C.ContractError(
                        "tile %d: refined crop %r != tile %r"
                        % (i, tuple(sub.shape), (y1 - y0, x1 - x0)))
                out[i] = np.ascontiguousarray(sub, dtype=np.float32)
    if any(p is None for p in out):
        raise C.ContractError("not every tile produced a patch")
    return out, wins, stats
