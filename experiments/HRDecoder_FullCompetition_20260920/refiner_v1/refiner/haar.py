#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Fixed (non-learnable) Haar high-frequency features.

Mechanism borrowed from WSRFNet's idea of a *reversible multi-scale Haar
representation*: the 2x2 orthonormal Haar analysis is exactly invertible, so
the high-frequency sub-bands carry the vessel / microaneurysm edge detail that
a 0.5-decision-threshold ensemble throws away -- without needing a second
encoder.

This module intentionally contains NO trainable parameters and NO learned
upsampling.  It is a pure feature extractor; everything it produces is a fixed
deterministic function of the green channel.  That is what lets the refiner
stay small: the multi-scale detail is handed to it, not learned from scratch.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

SUB_BANDS = ("h1", "v1", "d1", "h2", "v2", "d2")
NUM_SUB_BANDS = len(SUB_BANDS)          # -> 6 channels


def _even(x: torch.Tensor) -> torch.Tensor:
    """Pad H/W by 1 on the right/bottom when odd, so 2x2 pools divide evenly."""
    h, w = x.shape[-2], x.shape[-1]
    ph, pw = h % 2, w % 2
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph), mode="replicate")
    return x


def haar_level(x: torch.Tensor):
    """One 2x2 orthonormal Haar analysis level.

    x : (B, C, H, W)  ->  (a, h, v, d) each (B, C, ceil(H/2), ceil(W/2))

    Coefficients are the standard orthonormal 2x2 Haar basis /2:
        a = (p00 + p01 + p10 + p11) / 2
        h = (p00 - p01 + p10 - p11) / 2     (horizontal detail)
        v = (p00 + p01 - p10 - p11) / 2     (vertical detail)
        d = (p00 - p01 - p10 + p11) / 2     (diagonal detail)
    The transform is orthonormal, hence exactly invertible.
    """
    x = _even(x)
    p00 = x[..., 0::2, 0::2]
    p01 = x[..., 0::2, 1::2]
    p10 = x[..., 1::2, 0::2]
    p11 = x[..., 1::2, 1::2]
    a = (p00 + p01 + p10 + p11) * 0.5
    h = (p00 - p01 + p10 - p11) * 0.5
    v = (p00 + p01 - p10 - p11) * 0.5
    d = (p00 - p01 - p10 + p11) * 0.5
    return a, h, v, d


def haar_inverse(a, h, v, d) -> torch.Tensor:
    """Exact inverse of `haar_level` (used by the invertibility unit test)."""
    p00 = (a + h + v + d) * 0.5
    p01 = (a - h + v - d) * 0.5
    p10 = (a + h - v - d) * 0.5
    p11 = (a - h - v + d) * 0.5
    b, c, hh, ww = a.shape
    out = torch.empty(b, c, hh * 2, ww * 2, dtype=a.dtype, device=a.device)
    out[..., 0::2, 0::2] = p00
    out[..., 0::2, 1::2] = p01
    out[..., 1::2, 0::2] = p10
    out[..., 1::2, 1::2] = p11
    return out


def haar_hf_channels(green: torch.Tensor, levels: int = 2) -> torch.Tensor:
    """Stack `levels` levels of high-frequency Haar sub-bands, resampled to the
    input resolution with FIXED nearest-neighbour upsampling.

    green : (B, 1, H, W) in [0, 1]
    returns: (B, 3 * levels, H, W) -- level-major order h1,v1,d1,h2,v2,d2,
             matching refiner.haar.SUB_BANDS.
    """
    if green.ndim != 4 or green.shape[1] != 1:
        raise ValueError("green must be (B,1,H,W), got %r" % (tuple(green.shape),))
    hw = green.shape[-2:]
    outs, cur = [], green
    for _ in range(int(levels)):
        a, h, v, d = haar_level(cur)
        for band in (h, v, d):
            ups = F.interpolate(band, size=hw, mode="nearest")
            outs.append(ups)
        cur = a
    return torch.cat(outs, dim=1)
