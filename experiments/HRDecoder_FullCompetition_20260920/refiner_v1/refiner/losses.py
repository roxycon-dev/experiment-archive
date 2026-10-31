#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Two-term objective for the MA/HE refiner.  Nothing else.

Rationale for exactly two terms (no sum of several papers' losses):

  1. `focal_bce`  -- MA/HE are rare and extremely imbalanced *per pixel*.
     Focal loss (gamma=2) down-weights the easy majority and is the
     standard remedy for hard-example dominance; the per-class alpha gives
     the class balance.  This is the *classification* term: it decides which
     pixels the refiner moves across 0.5.
  2. `soft_dice`  -- IoU-shaped, region-level overlap term.  The evaluation
     metric is per-class IoU, so a region term is directly aligned with the
     objective; it also stops the classifier from over-fragmenting.

The two weights are FROZEN before the first formal step (2026-09-21):

    L = 30.0 * focal_bce(alpha=0.75, gamma=2.0) + 1.0 * soft_dice

Rationale (one-time design decision, from GO_NO_GO.md risk R1): at w_focal=1.0
the measured focal share of the total loss in the 40-step learning smoke was
only ~0.5 %, i.e. the objective was effectively Dice-only and the
hard-example term was never exercised.  With w_focal=30 the focal term is a
visible but non-dominant part of the early objective, which is what the
two-term design intends.  This value was fixed BEFORE any formal epoch and
must never be revised afterwards, and never on val.

`assert_frozen_objective` is called by every training entry point, so a
config that has silently drifted from these numbers fails closed.

Ignored pixels (label == 255 or an explicit ignore mask) are excluded from
BOTH terms by renormalising over the valid mask; a patch with zero valid
pixels returns a zero loss with a valid (non-NaN) graph.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

EPS = 1e-6

# ---- frozen objective (single source of truth, mirrored in the configs) ----
FROZEN_TERMS = ("focal_bce", "soft_dice")
FROZEN_WEIGHTS = {"focal_bce": 30.0, "soft_dice": 1.0}
FROZEN_ALPHA = 0.75
FROZEN_GAMMA = 2.0
FROZEN_OBJECTIVE_STR = "L = 30.0*focal_bce(alpha=0.75,gamma=2) + 1.0*soft_dice"


def assert_frozen_objective(loss_cfg: dict) -> dict:
    """Fail closed unless the config declares exactly the frozen objective."""
    terms = list(loss_cfg.get("terms") or [])
    if terms != list(FROZEN_TERMS):
        raise ValueError("objective terms %r != frozen %r" % (terms, list(FROZEN_TERMS)))
    w = {k: float(v) for k, v in (loss_cfg.get("weights") or {}).items()}
    if w != FROZEN_WEIGHTS:
        raise ValueError("objective weights %r != frozen %r" % (w, FROZEN_WEIGHTS))
    focal = loss_cfg.get("focal") or {}
    if float(focal.get("alpha", -1)) != FROZEN_ALPHA or \
       float(focal.get("gamma", -1)) != FROZEN_GAMMA:
        raise ValueError("focal params %r are not the frozen (alpha=%.2f, gamma=%.1f)"
                         % (loss_cfg.get("focal"), FROZEN_ALPHA, FROZEN_GAMMA))
    return {"terms": terms, "weights": dict(w), "objective": FROZEN_OBJECTIVE_STR}


def focal_bce(prob: torch.Tensor, target: torch.Tensor, valid: torch.Tensor,
              alpha: float = 0.75, gamma: float = 2.0) -> torch.Tensor:
    """Binary focal loss, class-balanced by a fixed alpha.

    prob   : (B, C, H, W) probabilities in [0, 1]
    target : (B, C, H, W) in {0, 1}
    valid  : (B, 1, H, W) or (B, C, H, W) in {0, 1}; 0 = ignore
    """
    if prob.shape != target.shape:
        raise ValueError("prob %r vs target %r" % (tuple(prob.shape), tuple(target.shape)))
    if valid.ndim == 4 and valid.shape[1] == 1 and prob.shape[1] > 1:
        valid = valid.expand_as(target)
    if valid.shape != target.shape:
        raise ValueError("valid %r must broadcast to target %r"
                         % (tuple(valid.shape), tuple(target.shape)))
    p = prob.clamp(EPS, 1.0 - EPS)
    ce = -(target * torch.log(p) + (1.0 - target) * torch.log(1.0 - p))
    pt = target * p + (1.0 - target) * (1.0 - p)
    at = target * alpha + (1.0 - target) * (1.0 - alpha)
    loss = at * (1.0 - pt).pow(gamma) * ce
    denom = valid.sum()
    if float(denom) <= 0.0:
        return prob.sum() * 0.0
    return (loss * valid).sum() / denom


def soft_dice(prob: torch.Tensor, target: torch.Tensor, valid: torch.Tensor,
              reduce: str = "mean") -> torch.Tensor:
    """Soft Dice over valid pixels, reduced over the class dim.

    reduce='mean' -> mean over classes (each class weighted equally, which is
    what we want: MA and HE must both improve, not be dominated by HE).
    """
    if prob.shape != target.shape:
        raise ValueError("prob %r vs target %r" % (tuple(prob.shape), tuple(target.shape)))
    v = valid
    if v.ndim == 4 and v.shape[1] == 1 and prob.shape[1] > 1:
        v = v.expand_as(target)
    p = prob.clamp(0.0, 1.0) * v
    t = target * v
    dims = (0, 2, 3)
    inter = (p * t).sum(dim=dims)
    denom = p.sum(dim=dims) + t.sum(dim=dims)
    dice = (2.0 * inter + EPS) / (denom + EPS)
    loss = 1.0 - dice
    if reduce == "mean":
        return loss.mean()
    return loss


def weighted_parts(parts: dict, w_focal: float, w_dice: float) -> dict:
    """Weighted contributions and the focal share of the total.

    `focal_share` is the number the training log records so that the R1
    concern ("is the hard-example term actually exercised?") can be checked
    from the CSV without recomputing anything.
    """
    fw = float(w_focal) * float(parts["focal"])
    dw = float(w_dice) * float(parts["dice"])
    tot = fw + dw
    return {"focal_weighted": fw, "dice_weighted": dw, "total": tot,
            "focal_share": (fw / tot) if tot > 0.0 else 0.0}


def refiner_losses(prob_ref: torch.Tensor, target: torch.Tensor,
                   valid: torch.Tensor, alpha: float = 0.75, gamma: float = 2.0,
                   w_focal: float = FROZEN_WEIGHTS["focal_bce"],
                   w_dice: float = FROZEN_WEIGHTS["soft_dice"]):
    """The complete objective.  Returns (total, parts) with parts as floats."""
    lf = focal_bce(prob_ref, target, valid, alpha=alpha, gamma=gamma)
    ld = soft_dice(prob_ref, target, valid)
    total = w_focal * lf + w_dice * ld
    parts = {"focal": float(lf.detach()), "dice": float(ld.detach()),
             "w_focal": float(w_focal), "w_dice": float(w_dice)}
    parts.update(weighted_parts(parts, w_focal, w_dice))
    return total, parts


def dice_vs_reference(prob: torch.Tensor, prob_champ: torch.Tensor,
                      target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Diagnostic only (never optimised): Dice of the refined probability
    versus the *binary* GT, next to the champion's.  Used by the eval script
    to report per-class deltas."""
    def d(p):
        v = valid
        if v.ndim == 4 and v.shape[1] == 1 and p.shape[1] > 1:
            v = v.expand_as(target)
        pp = p.clamp(0, 1) * v
        t = target * v
        dims = (0, 2, 3)
        inter = (pp * t).sum(dim=dims)
        denom = pp.sum(dim=dims) + t.sum(dim=dims)
        return (2.0 * inter + EPS) / (denom + EPS)
    return torch.stack([d(prob), d(prob_champ)], dim=0)
