#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Frozen candidate rule for the HRDecoder MA/HE candidate patch refiner.

SINGLE SOURCE OF TRUTH.  The GO/NO-GO audit (`audit/audit_candidate_rules.py`)
and the patch-cache builder (`build_patch_cache.py`) both import these
functions, so the rule that was audited is *the same code* that runs at
inference.  There is no second implementation to drift.

FROZEN DEFINITION (declared before any val number was inspected)
---------------------------------------------------------------
    seeds  = (P_MA > t) | (P_HE > t)          P_* = frozen champion probability,
                                              original resolution, float32
    grown  = dilate(seeds, rect(2*(d+margin)+1))      # bridging + box margin
    boxes  = 8-connected component bboxes of `grown`
    tiles  = fixed 512x512 grid, stride 384, clipped to each box
    cand   = union of tiles                      <- the ONLY pixels a refiner
                                                    may change

`t` in {0.05, 0.10, 0.20} is the only scanned axis and is used solely for the
GO/NO-GO gate.  The value finally written into configs/refiner_v1.json is
frozen and must never be re-scanned (see GO_NO_GO.md).

Rectangular (not elliptical) structuring element: OpenCV's rectangular
morphology is separable, so radius up to 56 px stays cheap on 4.5 MP fundus
images.  The shape is part of the frozen rule, not a tuned parameter.
"""
from __future__ import annotations

import numpy as np

# ---- frozen constants ------------------------------------------------------
SEED_THRESHOLDS = (0.05, 0.10, 0.20)   # GO/NO-GO audit axis only
SEED_DILATIONS = (0, 8, 16, 32)        # GO/NO-GO audit axis only
MARGIN = 24                            # px, box margin / bridging radius
TILE = 512                             # px, patch tile edge
TILE_STRIDE = 384                      # px, tile stride (overlap = TILE-STRIDE)
HAAR_LEVELS = 2
NUM_TILE_INPUT_CHANNELS = 3 + 2 + 6    # RGB + champion MA/HE prob + 6 Haar HF

# Extraction geometry.  The refiner's effective receptive field is
# 1 + 1 + (4*3) + (7*2) = ~28 px half-width (stem 3x3 + two HeteroDW blocks),
# so 32 px of REAL neighbouring pixels is enough for the boundary pixels of a
# tile to see the same context they would see in a whole-image forward.
# Nothing is zero-padded: the stored window is real image content.
PATCH_CONTEXT = 32
BACKGROUND_TILE = 256                  # bg patches are training-only; 256 is enough

FORBIDDEN_PATH_COMPONENTS = ("test",)  # fail-closed path guard

CLASSES = ("EX", "HE", "SE", "MA")
CLS_INDEX = {c: i for i, c in enumerate(CLASSES)}
REFINED_CLASSES = ("MA", "HE")          # EX/SE are never touched
GT_VALUE = {c: i + 1 for i, c in enumerate(CLASSES)}   # label png value per class


class ContractError(RuntimeError):
    pass


def check_no_forbidden_paths(*paths: str) -> None:
    """Fail closed if any path carries a forbidden component (e.g. 'test')."""
    for p in paths:
        if not p:
            continue
        norm = str(p).replace("\\", "/").lower()
        parts = [q for q in norm.split("/") if q]
        parts = [q.split(".")[0] if "." in q else q for q in parts]
        for bad in FORBIDDEN_PATH_COMPONENTS:
            if bad in parts:
                raise ContractError("forbidden path component %r in %r" % (bad, p))


# ---------------------------------------------------------------------------
# rule engine
# ---------------------------------------------------------------------------

def candidate_seeds(p_ma: np.ndarray, p_he: np.ndarray, t: float) -> np.ndarray:
    """Boolean seed mask at original resolution.  Fixed formula."""
    if p_ma.shape != p_he.shape:
        raise ContractError("p_ma %r vs p_he %r shape mismatch" % (p_ma.shape, p_he.shape))
    return (p_ma > t) | (p_he > t)


def grow_mask(seed: np.ndarray, radius: int) -> np.ndarray:
    """Dilate by `radius` px with a RECT structuring element."""
    import cv2
    if radius <= 0:
        return seed.astype(bool)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * radius + 1, 2 * radius + 1))
    return cv2.dilate(seed.astype(np.uint8), k, iterations=1).astype(bool)


def boxes_from_mask(mask: np.ndarray):
    """8-connected component bboxes (x0, y0, x1, y1) of a boolean mask."""
    import cv2
    n, _lab, stats, _cent = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8)
    return [(int(stats[i, 0]), int(stats[i, 1]),
             int(stats[i, 0] + stats[i, 2]), int(stats[i, 1] + stats[i, 3]))
            for i in range(1, n)]


def tile_patch(box, tile: int = TILE, stride: int = TILE_STRIDE):
    """Split a cluster bbox into <= tile x tile patches with the fixed stride."""
    x0, y0, x1, y1 = (int(v) for v in box)
    w, h = x1 - x0, y1 - y0
    xs = [x0] if w <= tile else _axis(x0, x1, tile, stride)
    ys = [y0] if h <= tile else _axis(y0, y1, tile, stride)
    out = set()
    for py in ys:
        for px in xs:
            out.add((int(px), int(py), int(min(px + tile, x1)), int(min(py + tile, y1))))
    return sorted(out)


def _axis(a0: int, a1: int, tile: int, stride: int):
    vals = list(range(a0, a1 - tile + 1, stride))
    tail = a1 - tile
    if not vals or tail > vals[-1]:
        vals.append(tail)
    return sorted(set(vals))


def candidate_tiles(p_ma: np.ndarray, p_he: np.ndarray, t: float,
                    dilate: int, margin: int = MARGIN,
                    tile: int = TILE, stride: int = TILE_STRIDE):
    """Frozen rule -> (tiles, cluster_boxes)."""
    seeds = candidate_seeds(p_ma, p_he, t)
    grown = grow_mask(seeds, dilate + margin)
    boxes = boxes_from_mask(grown)
    tiles = []
    for b in boxes:
        tiles.extend(tile_patch(b, tile, stride))
    return sorted(set(tiles)), boxes


def mask_from_tiles(tiles, shape) -> np.ndarray:
    """Boolean write-back mask == union of tiles (== the candidate region)."""
    m = np.zeros(shape, dtype=bool)
    for x0, y0, x1, y1 in tiles:
        m[y0:y1, x0:x1] = True
    return m


def window_for(tile, H, W, context: int = PATCH_CONTEXT,
               max_patch: int = 768):
    """Extraction window = tile grown by `context`, clipped to the image, capped.

    Single source of truth shared by the cache builder and the evaluator, so a
    patch cut at build time and a patch cut at inference time are the same
    rectangle.  The window always strictly contains the tile (asserted).
    """
    x0, y0, x1, y1 = (int(v) for v in tile)
    ex0, ey0 = x0 - context, y0 - context
    ex1, ey1 = x1 + context, y1 + context
    if ex1 - ex0 > max_patch:
        mid = (x0 + x1) // 2
        ex0, ex1 = mid - max_patch // 2, mid + max_patch // 2
    if ey1 - ey0 > max_patch:
        mid = (y0 + y1) // 2
        ey0, ey1 = mid - max_patch // 2, mid + max_patch // 2
    ex0 = max(0, ex0); ey0 = max(0, ey0)
    ex1 = min(int(W), ex1); ey1 = min(int(H), ey1)
    if not (ex0 <= x0 and ey0 <= y0 and ex1 >= x1 and ey1 >= y1):
        raise ContractError("window lost the tile: win=%r tile=%r"
                            % ((ex0, ey0, ex1, ey1), (x0, y0, x1, y1)))
    return ex0, ey0, ex1, ey1


def windows_for(tiles, H, W, context: int = PATCH_CONTEXT, max_patch: int = 768):
    """`window_for` for a list of tiles, in the SAME order.

    Kept as a separate function (rather than a list comprehension at the call
    site) so the evaluator cannot accidentally invent a second geometry.
    """
    return [window_for(t, H, W, context, max_patch) for t in tiles]


def bucket_windows_by_shape(windows):
    """Group window indices by their exact (h, w): deterministic, no padding.

    Candidate tiles are clipped to their cluster bbox and windows are clipped
    again at the image border, so a single image routinely yields a dozen
    different patch sizes (measured: 892 distinct candidate window shapes over
    the 76 val images, 2..51 per image).  A batch therefore cannot be stacked
    naively; it is grouped by shape instead, so every forward pass sees only
    tensors of one identical size and NO padding pixel ever reaches a
    convolution or a GroupNorm.  Zero-padding a patch up to the largest window
    of an arbitrary group would change that sample's normalisation statistics
    (and hence its prediction) as a function of which other patches happened to
    share the batch.

    Returns a list of ((h, w), [index, ...]) in order of first appearance.
    """
    buckets = {}
    for i, win in enumerate(windows):
        ex0, ey0, ex1, ey1 = (int(v) for v in win)
        h, w = ey1 - ey0, ex1 - ex0
        if h <= 0 or w <= 0:
            raise ContractError("empty extraction window %r for tile %d"
                                % ((ex0, ey0, ex1, ey1), i))
        buckets.setdefault((h, w), []).append(i)
    return list(buckets.items())


def extract_exact(rgb, ma_he, win):
    """Cut ONE patch out of the full-resolution arrays -- real pixels only.

    Returns (rgb_chw, prob_chw) as float32: (3, h, w) in [0, 1] and (2, h, w).
    The rectangle is the window itself (tile + context, clipped, capped), the
    same one `build_patch_cache.cut` stores, so an inference patch and a cached
    training patch are the same crop by construction.

    rgb   : (H, W, 3) uint8 full image (or (3, H, W))
    ma_he : (2, H, W) float32 champion (MA, HE)
    """
    ex0, ey0, ex1, ey1 = (int(v) for v in win)
    rgb_c = rgb if (rgb.ndim == 3 and rgb.shape[2] == 3) \
        else np.asarray(rgb).transpose(1, 2, 0)
    rr = rgb_c[ey0:ey1, ex0:ex1]
    pp = np.asarray(ma_he)[:, ey0:ey1, ex0:ex1]
    if rr.shape[:2] != (ey1 - ey0, ex1 - ex0) or pp.shape[1:] != rr.shape[:2]:
        raise ContractError("window %r is outside the image (got rgb %r prob %r)"
                            % ((ex0, ey0, ex1, ey1), tuple(rr.shape),
                               tuple(pp.shape)))
    out_rgb = np.ascontiguousarray(rr.transpose(2, 0, 1), dtype=np.float32)
    out_rgb /= 255.0
    return out_rgb, np.ascontiguousarray(pp, dtype=np.float32)


def extract_exact_batch(rgb, ma_he, windows, idxs):
    """Stack the exact (unpadded) patches `idxs` -- all must share one shape.

    Fails closed instead of padding: a caller that mixes window shapes has a
    bucketing bug, and silently padding would change the model's output.
    """
    idxs = [int(i) for i in idxs]
    if not idxs:
        raise ContractError("extract_exact_batch got no indices")
    shapes = {(int(windows[i][3]) - int(windows[i][1]),
               int(windows[i][2]) - int(windows[i][0])) for i in idxs}
    if len(shapes) != 1:
        raise ContractError("extract_exact_batch got mixed window shapes %r"
                            % (sorted(shapes),))
    h, w = shapes.pop()
    n = len(idxs)
    out_rgb = np.empty((n, 3, h, w), np.float32)
    out_p = np.empty((n, 2, h, w), np.float32)
    for k, i in enumerate(idxs):
        out_rgb[k], out_p[k] = extract_exact(rgb, ma_he, windows[i])
    return out_rgb, out_p


# ---------------------------------------------------------------------------
# fixed accumulation weights for overlapping tiles
# ---------------------------------------------------------------------------

def hann2d(h: int, w: int, eps: float = 1e-3) -> np.ndarray:
    """Fixed separable Hann window in [eps, 1]; no trained parameters.

    Used for the fixed weighted average when tiles overlap.  A Hann window
    de-emphasises tile borders, which is the standard way to suppress seam
    artefacts without introducing a learned blend.
    """
    wy = np.hanning(h).astype(np.float64)
    wx = np.hanning(w).astype(np.float64)
    if wy.size == 0:
        wy = np.ones(1)
    if wx.size == 0:
        wx = np.ones(1)
    wy = np.maximum(wy, eps)
    wx = np.maximum(wx, eps)
    return (wy[:, None] * wx[None, :]).astype(np.float32)


def stitch_tiles(tiles, patches, weight_fn=hann2d, shape=None, dtype=np.float32):
    """Fixed weighted average of overlapping tile predictions.

    tiles   : list of (x0, y0, x1, y1)
    patches : array-like of (N, H, W) or (N, C, H, W) aligned with `tiles`
    returns : (acc / wsum) over `shape`, NaN-free (absent pixels are left 0)
    """
    patches = np.asarray(patches)
    if patches.ndim == 3:
        patches = patches[:, None]
    n, c = patches.shape[0], patches.shape[1]
    if n != len(tiles):
        raise ContractError("stitch_tiles: %d tiles vs %d patches" % (len(tiles), n))
    if shape is None:
        H = max(t[3] for t in tiles)
        W = max(t[2] for t in tiles)
        shape = (H, W)
    acc = np.zeros((c,) + tuple(shape), np.float64)
    wsum = np.zeros(tuple(shape), np.float64)
    for i, (x0, y0, x1, y1) in enumerate(tiles):
        wh, ww = y1 - y0, x1 - x0
        if patches[i].shape[-2] < wh or patches[i].shape[-1] < ww:
            raise ContractError(
                "stitch_tiles: patch %d is %r but tile %d needs at least (%d, %d)"
                % (i, tuple(patches[i].shape[-2:]), i, wh, ww))
        w2 = weight_fn(wh, ww).astype(np.float64)
        acc[:, y0:y1, x0:x1] += patches[i][:, :wh, :ww].astype(np.float64) * w2[None]
        wsum[y0:y1, x0:x1] += w2
    out = np.zeros_like(acc, dtype=dtype)
    nz = wsum > 0
    out[:, nz] = (acc[:, nz] / wsum[nz][None, :]).astype(dtype)
    return out
