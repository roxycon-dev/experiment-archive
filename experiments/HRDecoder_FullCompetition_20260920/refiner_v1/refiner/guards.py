#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Fail-closed guards: no test data, no training of the frozen champion.

These are the same shape of guard the previous round used successfully, kept
deliberately small so they can be read in one screen.
"""
from __future__ import annotations

import os

from . import candidate as C

ALLOWED_SPLITS = ("train", "val")


def assert_no_test_paths(*paths: str) -> None:
    """Raise if any path component is literally 'test' (case-insensitive)."""
    C.check_no_forbidden_paths(*paths)


def assert_split(split: str) -> str:
    if split not in ALLOWED_SPLITS:
        raise C.ContractError("split %r is not allowed (allowed: %r)"
                              % (split, ALLOWED_SPLITS))
    return split


def assert_val_only_inference(split: str, *paths: str) -> None:
    """Inference/eval entry points may only ever see val (or train for cache
    builds), and must never carry a path containing 'test'."""
    assert_split(split)
    assert_no_test_paths(*paths)


def assert_frozen_reference(model) -> None:
    """The champion must be an input provider, not part of the graph."""
    for p in model.parameters():
        if p.requires_grad:
            raise C.ContractError(
                "frozen reference has a trainable parameter (%r)" % (p.shape,))
    if model.training:
        raise C.ContractError("frozen reference must be in eval() mode")


def assert_only_refiner_trainable(refiner, others) -> dict:
    """Exactly the refiner parameters require grad; returns a count report."""
    rn = sum(p.numel() for p in refiner.parameters() if p.requires_grad)
    rt = sum(p.numel() for p in refiner.parameters())
    on = sum(p.numel() for m in others for p in m.parameters() if p.requires_grad)
    if on != 0:
        raise C.ContractError("%d parameters outside the refiner require grad" % on)
    return {"refiner_trainable": rn, "refiner_total": rt,
            "non_refiner_trainable": on}


def add_config_path(args, cfg, key, value):
    """Small helper used by the entry points to keep provenance in one place."""
    setattr(args, key, value)
    return args


def sha256_file(path: str) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_tree(root: str, exts=(".py", ".json")) -> dict:
    out = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in ("__pycache__", ".git")]
        for fn in sorted(filenames):
            if fn.endswith(exts):
                p = os.path.join(dirpath, fn)
                out[os.path.relpath(p, root).replace("\\", "/")] = sha256_file(p)
    return out
