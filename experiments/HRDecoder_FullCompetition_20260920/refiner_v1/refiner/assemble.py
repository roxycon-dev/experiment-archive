#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Write-back assembly: champion probability -> refined probability.

Hard guarantees enforced here (each has a unit test):

  G1  Pixels outside the candidate tiles are BITWISE the champion, for all
      four classes.
  G2  EX and SE are BITWISE the champion everywhere -- the refiner has no
      code path that can produce them.
  G3  Overlapping tiles are combined with a FIXED weighted average (Hann
      window); the weights are non-learned and sum to 1 in the overlap.
  G4  The final decision is always the FIXED threshold p > 0.5.  No
      per-image or per-class threshold adaptation.
  G5  A patch that was never produced (missing tile) leaves the champion
      value untouched rather than introducing a zero.
  G6  `patches` may be RAGGED (one array per tile): tiles are clipped to their
      cluster bbox and to the image border, so an image normally mixes dozens
      of patch sizes and they must never be stacked into one padded array.

Everything is float32, matching the champion's own precision contract.
"""
from __future__ import annotations

import numpy as np

from . import candidate as C

THRESHOLD = 0.5
REFINED_INDEX = tuple(C.CLS_INDEX[c] for c in C.REFINED_CLASSES)   # (3, 1)


def _patch_list(patches, n_tiles: int):
    """Normalise `patches` to a plain list of (2, h, w) arrays, one per tile.

    Accepts either a stacked (N, 2, h, w) array (every tile the same size) or a
    RAGGED sequence of per-tile arrays.  Ragged is the normal case: candidate
    tiles are clipped to their cluster bbox and to the image border, so an
    image typically mixes dozens of patch sizes (see `tiled_infer`).
    """
    if isinstance(patches, np.ndarray):
        if patches.ndim != 4 or patches.shape[1] != len(C.REFINED_CLASSES):
            raise C.ContractError("patches must be (N,2,h,w), got %r"
                                  % (tuple(patches.shape),))
        if patches.shape[0] != n_tiles:
            raise C.ContractError("patches %d != tiles %d"
                                  % (patches.shape[0], n_tiles))
        return list(patches)
    seq = list(patches)
    if len(seq) != n_tiles:
        raise C.ContractError("patches %d != tiles %d" % (len(seq), n_tiles))
    out = []
    for i, p in enumerate(seq):
        p = np.asarray(p)
        if p.ndim != 3 or p.shape[0] != len(C.REFINED_CLASSES):
            raise C.ContractError("patch %d must be (2,h,w), got %r"
                                  % (i, tuple(p.shape)))
        out.append(p)
    return out


def assemble(champ_prob: np.ndarray, tiles, patches, *,
             weight_fn=C.hann2d) -> np.ndarray:
    """Return the refined (4, H, W) float32 probability map.

    champ_prob : (4, H, W) float32 frozen champion probability at ORIGINAL res
    tiles      : list of (x0, y0, x1, y1) in original coords, patch order
    patches    : one (2, h, w) refined MA/HE patch per tile -- as a stacked
                 (N,2,h,w) array when all tiles have the same size, or as a
                 ragged list/sequence when they do not (the usual case).  Each
                 patch must be at least as large as its tile (patches are
                 extracted with context, so it is normally larger); the tile
                 region is taken from the patch centre.
    """
    if champ_prob.ndim != 3 or champ_prob.shape[0] != len(C.CLASSES):
        raise C.ContractError("champ_prob must be (4,H,W), got %r"
                              % (tuple(champ_prob.shape),))
    out = np.array(champ_prob, dtype=np.float32, copy=True)     # G2/G1 default
    if not tiles:
        return out
    plist = _patch_list(patches, len(tiles))

    H, W = out.shape[1], out.shape[2]
    acc = np.zeros((2, H, W), np.float64)
    wsum = np.zeros((H, W), np.float64)
    for i, (x0, y0, x1, y1) in enumerate(tiles):
        x0 = max(0, int(x0)); y0 = max(0, int(y0))
        x1 = min(W, int(x1)); y1 = min(H, int(y1))
        if x1 <= x0 or y1 <= y0:
            continue                                                   # G5
        p = plist[i]
        ph, pw = p.shape[-2:]
        th, tw = y1 - y0, x1 - x0
        if ph < th or pw < tw:
            raise C.ContractError(
                "tile %d patch %dx%d smaller than tile %dx%d" % (i, ph, pw, th, tw))
        cy, cx = (ph - th) // 2, (pw - tw) // 2
        sub = p[:, cy:cy + th, cx:cx + tw].astype(np.float64)
        w2 = weight_fn(th, tw).astype(np.float64)
        acc[:, y0:y1, x0:x1] += sub * w2[None]
        wsum[y0:y1, x0:x1] += w2

    touched = wsum > 0
    for j, ci in enumerate(REFINED_INDEX):
        ch = out[ci]
        ch[touched] = (acc[j, touched] / wsum[touched]).astype(np.float32)
        out[ci] = ch
    return out


def binarize(prob: np.ndarray, threshold: float = THRESHOLD) -> np.ndarray:
    """Fixed strict threshold, matching the historical export contract."""
    if float(threshold) != THRESHOLD:
        raise C.ContractError("binarize threshold is fixed at %.2f, got %r"
                              % (THRESHOLD, threshold))
    if not np.issubdtype(prob.dtype, np.floating) or prob.dtype != np.float32:
        raise C.ContractError("binarize requires float32, got %r" % prob.dtype)
    return prob > threshold


def assert_bitwise_unchanged(new_prob: np.ndarray, champ_prob: np.ndarray,
                             tiles) -> dict:
    """Verify G1/G2 on an actual assembled map.  Returns a report dict."""
    ch = np.asarray(champ_prob, np.float32)
    nw = np.asarray(new_prob, np.float32)
    if ch.shape != nw.shape:
        raise C.ContractError("shape mismatch %r vs %r" % (ch.shape, nw.shape))
    cand = C.mask_from_tiles(tiles, ch.shape[1:]) if tiles else np.zeros(
        ch.shape[1:], bool)
    rep = {"n_pixels": int(ch.shape[1] * ch.shape[2]),
           "n_candidate_px": int(cand.sum())}
    for ci, name in enumerate(C.CLASSES):
        outside = ~cand
        rep["%s_changed_outside" % name] = int(
            np.count_nonzero(nw[ci][outside] != ch[ci][outside]))
    for name in ("EX", "SE"):
        ci = C.CLS_INDEX[name]
        rep["%s_total_changed" % name] = int(np.count_nonzero(nw[ci] != ch[ci]))
    rep["MA_changed_inside"] = int(np.count_nonzero(
        (nw[REFINED_INDEX[0]] != ch[REFINED_INDEX[0]]) & cand))
    rep["HE_changed_inside"] = int(np.count_nonzero(
        (nw[REFINED_INDEX[1]] != ch[REFINED_INDEX[1]]) & cand))
    return rep
