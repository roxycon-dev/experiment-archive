#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Patch dataset for the MA/HE refiner — reads the frozen patch cache.

The cache is produced ONCE by `build_patch_cache.py` from the frozen champion
(RGB + champion MA/HE probability + GT at original resolution, cropped to
candidate patches).  Nothing here re-runs the champion.

Cache layout (server root = CACHE_ROOT, never contains 'test'):

    <cache_root>/index_{split}.json      per-image manifest + provenance
    <cache_root>/patch/{split}/{stem}.npz   (ragged, no padding)
        tiles   int32   (N,4)      x0,y0,x1,y1 in ORIGINAL image coords
        rects   int32   (N,4)      extraction window actually stored
        shape   int32   (N,2)      per-patch (h, w)
        role    uint8   (N,)       0 candidate-pos 1 candidate-neg-fp
                                   2 candidate-neg-plain 3 background
        rgb     uint8   flat       sum_i 3*h_i*w_i, channel-first per patch
        prob    float16 flat       sum_i 2*h_i*w_i, champion (MA, HE)
        gt      uint8   flat       sum_i 2*h_i*w_i, GT MA/HE binary
        valid   uint8   flat       sum_i 1*h_i*w_i, 1 inside the tile

Patches are stored at their NATURAL size instead of padded to a common grid:
padding every patch up to the largest window in an image inflated the cache by
more than 10x in the measured smoke run, and zero padding also pollutes
normalisation statistics.  The reader rebuilds the per-patch offsets from
`shape`.

Why patches and not full probability maps: a full 4-channel fp32 map for one
1728x2592 fundus image is ~72 MB; 530 images would be ~38 GB.  Candidate
patches cover <5% of pixels (measured), so the whole train+val cache stays in
the low-GB range and every crop is *self-consistent*: rgb, prob and gt are cut
from the same rectangle, so no independent resampling can desynchronise them.

Sampling protocol (frozen, declared in configs/refiner_v1.json):
  * ALL candidate tiles are kept (no cap) -> the audited candidate recall is
    preserved exactly; dropping tiles would silently lower the upper bound.
  * plus `n_background_per_image` extra tiles with no GT and no champion
    activation, drawn deterministically with `seed`.  They are training-only:
    they are not candidates, so inference never writes them back.
"""
from __future__ import annotations

import json
import os
import zlib

import numpy as np
import torch
from torch.utils.data import Dataset

from . import candidate as C

CACHE_ROOT_DEFAULT = ("/root/autodl-tmp/HRDecoder_FullCompetition_20260920/"
                      "refiner_v1/patch_cache")
ROLE_NAMES = {0: "candidate_pos", 1: "candidate_neg_fp",
              2: "candidate_neg_plain", 3: "background"}


def load_index(cache_root: str, split: str) -> dict:
    C.check_no_forbidden_paths(cache_root, split)
    p = os.path.join(cache_root, "index_%s.json" % split)
    if not os.path.isfile(p):
        raise C.ContractError("cache index missing: %s" % p)
    with open(p, "r", encoding="utf-8") as f:
        idx = json.load(f)
    if idx.get("split") != split:
        raise C.ContractError("index split %r != %r" % (idx.get("split"), split))
    return idx


class PatchCacheDataset(Dataset):
    """One sample = one 512-bounded candidate patch (or background tile)."""

    def __init__(self, cache_root: str, split: str, haar_levels: int = 2,
                 train: bool = True, augment: bool = True, seed: int = 0,
                 include_background: bool = True):
        self.cache_root = cache_root
        self.split = split
        C.check_no_forbidden_paths(cache_root, split)
        idx = load_index(cache_root, split)
        self.idx = idx
        self.haar_levels = int(haar_levels)
        self.train = bool(train)
        self.augment = bool(augment) and self.train
        self.seed = int(seed)
        entries = []
        for rec in idx["images"]:
            for k, role in enumerate(rec["roles"]):
                if int(role) == 3 and not include_background:
                    continue
                entries.append((rec["stem"], k, int(role)))
        self.entries = entries
        self._cache = {}
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Change the augmentation seed stream (training loop calls this)."""
        self.epoch = int(epoch)

    # -- helpers -------------------------------------------------------------
    def _npz(self, stem):
        if stem not in self._cache:
            p = os.path.join(self.cache_root, "patch", self.split, stem + ".npz")
            if not os.path.isfile(p):
                raise C.ContractError("patch npz missing: %s" % p)
            self._cache[stem] = np.load(p, allow_pickle=False)
        return self._cache[stem]

    def role_counts(self) -> dict:
        out = {v: 0 for v in ROLE_NAMES.values()}
        for _s, _k, r in self.entries:
            out[ROLE_NAMES[r]] += 1
        return out

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, i):
        stem, k, role = self.entries[i]
        z = self._npz(stem)
        tiles = np.asarray(z["tiles"])
        shapes = np.asarray(z["shape"])
        n_tiles = tiles.shape[0]
        h, w = int(shapes[k, 0]), int(shapes[k, 1])

        # offsets into the flat ragged arrays (channel-first per patch)
        def _offsets(c):
            size = shapes[:, 0].astype(np.int64) * shapes[:, 1].astype(np.int64) * c
            return np.concatenate([[0], np.cumsum(size)])

        o3, o2, o1 = _offsets(3), _offsets(2), _offsets(1)
        rgb_u8 = np.asarray(z["rgb"][o3[k]:o3[k + 1]]).reshape(3, h, w)
        prob = np.asarray(z["prob"][o2[k]:o2[k + 1]]).astype(np.float32).reshape(2, h, w)
        gt = np.asarray(z["gt"][o2[k]:o2[k + 1]]).astype(np.float32).reshape(2, h, w)
        valid = np.asarray(z["valid"][o1[k]:o1[k + 1]]).astype(np.float32).reshape(h, w)
        tile = np.asarray(tiles[k])
        if n_tiles != len(z["role"]):
            raise C.ContractError("corrupt cache for %s" % stem)

        rgb = rgb_u8.astype(np.float32) / 255.0
        if self.augment:
            # stable, process-independent seed (builtin hash() is salted per run)
            key = zlib.crc32(("%s|%d|%d" % (stem, int(k), self.epoch)).encode("utf-8"))
            rng = np.random.default_rng((self.seed, key))
            if rng.random() < 0.5:                      # hflip: SAME transform
                rgb = rgb[:, :, ::-1]; prob = prob[:, :, ::-1]
                gt = gt[:, :, ::-1]; valid = valid[:, ::-1]
            if rng.random() < 0.5:                      # vflip: SAME transform
                rgb = rgb[:, ::-1, :]; prob = prob[:, ::-1, :]
                gt = gt[:, ::-1, :]; valid = valid[::-1, :]
            # integer 90-degree rotation: keeps every array pixel-aligned
            rot = int(rng.integers(0, 4))
            if rot:
                rgb = np.rot90(rgb, rot, (1, 2)); prob = np.rot90(prob, rot, (1, 2))
                gt = np.rot90(gt, rot, (1, 2)); valid = np.rot90(valid, rot, (0, 1))
        # flips/rot90 produce negative or non-contiguous strides; torch refuses
        # negative strides, so every array is made contiguous explicitly.
        rgb = np.ascontiguousarray(rgb, dtype=np.float32)
        prob = np.ascontiguousarray(prob, dtype=np.float32)
        gt = np.ascontiguousarray(gt, dtype=np.float32)
        valid = np.ascontiguousarray(valid, dtype=np.float32)
        return {
            "rgb": torch.from_numpy(rgb),
            "p_champ": torch.from_numpy(prob),
            "gt": torch.from_numpy(gt),
            "valid": torch.from_numpy(valid)[None],
            "role": torch.tensor(role, dtype=torch.long),
            "tile": torch.from_numpy(np.asarray(tile, dtype=np.int64).copy()),
            "stem": stem,
        }


def collate(batch):
    """All patches in a batch are padded to a common (H,W) by zero padding.

    The cache already stores every patch on a common grid per image, but two
    images can have different patch sizes, so the batch is padded here.  The
    `valid` mask is expanded to the padded canvas so padding never contributes
    to the loss and is never written back.
    """
    h = max(b["rgb"].shape[-2] for b in batch)
    w = max(b["rgb"].shape[-1] for b in batch)
    n = len(batch)
    out = {
        "rgb": torch.zeros(n, 3, h, w),
        "p_champ": torch.zeros(n, 2, h, w),
        "gt": torch.zeros(n, 2, h, w),
        "valid": torch.zeros(n, 1, h, w),
        "role": torch.stack([b["role"] for b in batch]),
        "tile": torch.zeros(n, 4, dtype=torch.long),
        "stem": [b["stem"] for b in batch],
    }
    for i, b in enumerate(batch):
        hh, ww = b["rgb"].shape[-2:]
        out["rgb"][i, :, :hh, :ww] = b["rgb"]
        out["p_champ"][i, :, :hh, :ww] = b["p_champ"]
        out["gt"][i, :, :hh, :ww] = b["gt"]
        out["valid"][i, :, :hh, :ww] = b["valid"]
        out["tile"][i] = b["tile"]
    return out
