#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Frozen champion probability provider (inference only, never trained).

Reuses the historical `runtime.ChampionEnsemble` verbatim -- it is not
re-implemented here, so the MA/HE probabilities the refiner consumes are the
same numbers that produced the frozen champion masks (val mIoU 0.426694,
MA .259672 / HE .437583).

Contract:
  * all parameters `requires_grad_(False)`, module in `eval()` mode
  * MA  = sigmoid(logit(pf_MA) + H4(f7))     (post-fusion MA correction head)
  * HE  = pf_HE                              (bitwise identical to BASE_0 HE)
  * EX/SE are never read by the refiner; `assemble.py` keeps them untouched.
"""
from __future__ import annotations

import copy
import importlib
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

from . import candidate as C

DEFAULT_REPO = "/root/autodl-tmp/hr_exec_20260919/repo"
DEFAULT_SRC = "/root/autodl-tmp/hr_exec_20260919/code"
DEFAULT_SETTINGS = ("/root/autodl-tmp/HRDecoder_FullCompetition_20260920/analysis/"
                    "expert_eval_settings.json")
DEFAULT_DATA_ROOT = "/root/autodl-tmp/hr_exec_20260919/data/DDR_split_613"

MA_IDX = C.CLS_INDEX["MA"]
HE_IDX = C.CLS_INDEX["HE"]


class FrozenChampion:
    """Wraps ChampionEnsemble + the mmseg dataset for one split."""

    def __init__(self, settings_path: str = DEFAULT_SETTINGS,
                 data_root: str = DEFAULT_DATA_ROOT,
                 repo: str = DEFAULT_REPO, src: str = DEFAULT_SRC,
                 device: str = "cuda"):
        for d in (src, repo):
            if d not in sys.path:
                sys.path.insert(0, d)
        self.rt = importlib.import_module("runtime")
        self.rp = importlib.import_module("run_protocol")
        self.settings_path = settings_path
        self.data_root = data_root
        self.device = device
        settings = self.rp.validate_settings(settings_path)
        if settings.get("_pending"):
            raise C.ContractError("settings refer to missing assets: %s"
                                  % settings["_pending"])
        cfg = self.rt.load_config(settings["repo"], settings["config"])
        cfg.model["source_checkpoint"] = settings["_abs"]["reference_checkpoint"]
        settings["_cfg"] = cfg
        self.settings = settings
        self.cfg = cfg
        self.rt.inject_repo(settings["repo"])
        self.ensemble = self.rt.ChampionEnsemble(settings, settings["champion"])
        self._freeze()
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    def _freeze(self):
        mods = [m for m, _w, _s in getattr(self.ensemble, "members", [])]
        mods += list(getattr(self.ensemble, "heads", {}).values())
        for m in mods:
            if hasattr(m, "eval"):
                m.eval()
            for p in m.parameters():
                p.requires_grad_(False)

    def dataset(self, split: str):
        """Build the mmseg dataset for `split`.

        The *val* config entry is used as the template for BOTH splits.  The
        historical champion export verified that `cfg.data['val']` +
        test_mode=True reproduces the frozen champion masks exactly, so it is
        the only preprocessing pipeline in this repo with that provenance.
        Reusing it avoids a train/val preprocessing mismatch -- the patches
        are cut at original resolution and must be byte-comparable.
        """
        C.check_no_forbidden_paths(self.data_root, split)
        if split not in ("train", "val"):
            raise C.ContractError("split %r not allowed" % split)
        from mmseg.datasets import build_dataset
        d = copy.deepcopy(self.cfg.data["val"])
        d.test_mode = True
        d.data_root = self.data_root
        d.img_dir = os.path.join(self.data_root, "images", split)
        d.ann_dir = os.path.join(self.data_root, "labels", split)
        return build_dataset(d)

    @torch.no_grad()
    def champion_prob4(self, img: torch.Tensor, meta) -> np.ndarray:
        """(4, H, W) float32 champion probability at ORIGINAL resolution.

        Single `pre_head_probs` pass, then the same channel loop as
        `ChampionEnsemble.predict()` (which is what the frozen contract was
        produced with).  MA goes through the H4 head; EX/HE/SE stay `pf`.
        """
        pf, ps, pr = self.ensemble.pre_head_probs(img, meta)
        pf = pf.float(); ps = ps.float(); pr = pr.float()
        ori = tuple(int(x) for x in meta[0]["ori_shape"][:2])
        img_f = F.interpolate(img, size=ori, mode="bilinear", align_corners=False)[0]
        chans = []
        for c in range(len(C.CLASSES)):
            cname = C.CLASSES[c]
            if cname in self.ensemble.heads:
                chans.append(self.ensemble.corrected_class_prob(
                    c, pf, ps, pr, img_f, self.ensemble.heads[cname]).unsqueeze(0))
            else:
                chans.append(pf[c:c + 1])
        out = torch.cat(chans, dim=0).cpu().numpy().astype(np.float32)
        if tuple(out.shape[1:]) != ori:
            raise C.ContractError("champion prob shape %r != ori %r"
                                  % (tuple(out.shape), ori))
        return out

    @torch.no_grad()
    def champion_ma_he(self, img: torch.Tensor, meta) -> np.ndarray:
        """(2, H, W) champion (MA, HE) at ORIGINAL resolution.

        Derived from `champion_prob4` so the two can never disagree.
        NOTE the contract class order is EX=0, HE=1, SE=2, MA=3: MA and HE are
        NOT adjacent, so `p4[3:5]` would silently yield one channel.
        """
        p4 = self.champion_prob4(img, meta)
        return np.ascontiguousarray(np.stack([p4[MA_IDX], p4[HE_IDX]], axis=0))

    @torch.no_grad()
    def champion_full(self, img: torch.Tensor, meta) -> np.ndarray:
        """Alias of `champion_prob4` kept for readability at call sites."""
        return self.champion_prob4(img, meta)
