#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""HaarHeteroRefiner — a small MA/HE residual refiner for the frozen champion.

Design constraints (from the round decision):
  * Mechanism is *borrowed*, not the whole network re-implemented:
      - WSRFNet  -> a fixed, invertible Haar multi-scale high-frequency
                    representation (see refiner.haar, zero parameters).
      - HACDR-Net -> heterogeneous receptive-field aggregation: parallel
                    depthwise convolutions with different kernel sizes and
                    dilations, fused by 1x1 convolutions.
  * No stacking of several unvalidated large modules: the whole trainable
    network is a stem conv, two heterogeneous depthwise blocks and a 1x1 head.
  * The head is ZERO-INITIALISED and the residual is BOUNDED, so at
    initialisation the model returns the frozen champion probability
    EXACTLY (bitwise), and it can never move a probability by more than
    `residual_scale`.

Residual form (probability space, additive, exactly-zero at init):

    p_refined = clip( p_champ + scale * tanh(r), 0, 1 )

with r the raw head output.  Zero-init the head -> r == 0 -> tanh(0) == 0
-> p_refined == p_champ bitwise.  tanh'(0) == 1, so the branch is NOT dead:
gradients reach the head weights from the first step.  (Contrast with
ReLU-style gating, whose derivative is 0 at the origin.)

scale = 0.5 is a deliberate choice, written into configs/refiner_v1.json:
inside a candidate, champion p > t >= 0.05, so |delta| <= 0.5 is always
sufficient to move that pixel across the 0.5 decision boundary in either
direction, while capping the residual at half a probability unit.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .haar import haar_hf_channels

DEFAULT_IN_CH = 3 + 2 + 6      # RGB + champion MA/HE prob + 6 Haar HF bands


def _groups(want: int, ch: int) -> int:
    g = int(want)
    while g > 1 and ch % g != 0:
        g -= 1
    return max(1, g)


class HeteroDWBlock(nn.Module):
    """Heterogeneous receptive-field aggregation (HACDR-Net style).

    Three depthwise branches with different kernel sizes / dilations run on the
    same feature map; a 1x1 bottleneck fuses them and a residual 1x1 keeps the
    block identity-friendly.
    """

    def __init__(self, ch: int, kernels=(3, 5, 3), dilations=(1, 2, 4),
                 groups: int = 8):
        super().__init__()
        if len(kernels) != len(dilations):
            raise ValueError("kernels/dilations length mismatch")
        self.branches = nn.ModuleList([
            nn.Conv2d(ch, ch, k, padding=d * (k // 2), dilation=d, groups=ch,
                      bias=False)
            for k, d in zip(kernels, dilations)])
        # GroupNorm, not BatchNorm: candidate patches have different spatial
        # sizes and a batch is zero-padded to its maximum, which would leak
        # padding into BatchNorm's running statistics.  GroupNorm normalises
        # per sample and is size-agnostic, so padding can never change a
        # prediction for the pixels that matter.
        self.norm = nn.GroupNorm(_groups(groups, ch * len(kernels)),
                                 ch * len(kernels))
        self.fuse = nn.Conv2d(ch * len(kernels), ch, 1, bias=False)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        y = torch.cat([b(x) for b in self.branches], dim=1)
        y = self.act(self.norm(y))
        return x + self.fuse(y)


class HaarHeteroRefiner(nn.Module):
    """Frozen-champion MA/HE residual refiner.

    Input  (B, 3+2+6, H, W):
        [0:3]   RGB patch, [0,1]
        [3:5]   frozen champion probabilities for (MA, HE)
        [5:11]  fixed Haar high-frequency bands of the green channel
    Output (B, 2, H, W): raw residual logits r for (MA, HE).
        p_refined_c = clip(p_champ_c + scale * tanh(r_c), 0, 1)
    """

    def __init__(self, in_ch: int = DEFAULT_IN_CH, width: int = 48,
                 haar_levels: int = 2, residual_scale: float = 0.5):
        super().__init__()
        self.in_ch = int(in_ch)
        self.width = int(width)
        self.haar_levels = int(haar_levels)
        self.residual_scale = float(residual_scale)
        if self.residual_scale <= 0:
            raise ValueError("residual_scale must be > 0")

        self.stem = nn.Sequential(
            nn.Conv2d(self.in_ch, self.width, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(8, self.width), self.width),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(
            HeteroDWBlock(self.width, (3, 5, 3), (1, 2, 4)),
            HeteroDWBlock(self.width, (5, 3, 7), (1, 3, 2)),
        )
        self.head = nn.Conv2d(self.width, 2, 1)
        self.reset_head_zero()

    # -- construction helpers ------------------------------------------------
    def reset_head_zero(self):
        """Zero-init the head so the model is exactly the identity at start."""
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    @staticmethod
    def build_input(rgb: torch.Tensor, p_champ: torch.Tensor,
                    haar_levels: int = 2) -> torch.Tensor:
        """Assemble the 11-channel input.

        rgb      : (B, 3, H, W) in [0, 1]
        p_champ  : (B, 2, H, W) frozen champion (MA, HE) probabilities
        """
        if rgb.shape[-2:] != p_champ.shape[-2:]:
            raise ValueError("rgb %r vs p_champ %r spatial mismatch"
                             % (tuple(rgb.shape[-2:]), tuple(p_champ.shape[-2:])))
        green = rgb[:, 1:2]
        hf = haar_hf_channels(green, levels=haar_levels)
        return torch.cat([rgb, p_champ, hf], dim=1)

    # -- forward -------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.stem(x)
        y = self.blocks(y)
        return self.head(y)

    def refine(self, p_champ: torch.Tensor, residual_logits: torch.Tensor,
               clamp: bool = True) -> torch.Tensor:
        """Bounded additive residual in probability space.

        At residual_logits == 0 this returns `p_champ` **bitwise**
        (0.0 * tanh(0) is +0.0 and p + 0.0 == p for finite p).
        """
        delta = self.residual_scale * torch.tanh(residual_logits)
        out = p_champ + delta
        if clamp:
            out = out.clamp(0.0, 1.0)
        return out

    def forward_refine(self, rgb: torch.Tensor, p_champ: torch.Tensor):
        """Convenience: x -> (residual_logits, refined_prob)."""
        x = self.build_input(rgb, p_champ, self.haar_levels)
        r = self.forward(x)
        return r, self.refine(p_champ, r)
