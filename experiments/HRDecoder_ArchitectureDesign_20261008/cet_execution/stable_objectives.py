"""Reference objectives for v4 diagnosis; deliberately not wired to training.

Inputs use four independent foreground Bernoulli channels in EX/HE/SE/MA order.
``label`` is an integer DDR mask (0=background, 1..4=lesion, 255=ignore).
``valid`` is a binary mask with shape [N,H,W], [N,1,H,W], or [N,4,H,W].

All objectives reduce per (sample, class) and average over active valid pairs.
The module computes half/bfloat16 inputs in float32 and preserves float64 for
reference tests.  It contains no optimizer, model, data-loading, or training
code.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F

NUM_CLASSES = 4
SMOOTH = 1.0e-5
DEFAULT_MIX_WEIGHTS = (0.5625, 0.1875, 0.25)


def _compute_dtype(dtype: torch.dtype) -> torch.dtype:
    if not dtype.is_floating_point:
        raise TypeError(f"logits must have a floating dtype, got {dtype}")
    return torch.float64 if dtype == torch.float64 else torch.float32


def _validate_and_expand(
    logits: torch.Tensor,
    label: torch.Tensor,
    valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return labels [N,H,W] and a boolean valid mask [N,4,H,W]."""
    if not isinstance(logits, torch.Tensor) or logits.ndim != 4:
        raise ValueError("logits must have shape [N,4,H,W]")
    if logits.shape[1] != NUM_CLASSES:
        raise ValueError(f"logits must have exactly {NUM_CLASSES} channels")
    if not logits.dtype.is_floating_point:
        raise TypeError(f"logits must have a floating dtype, got {logits.dtype}")

    if not isinstance(label, torch.Tensor):
        raise TypeError("label must be a torch.Tensor")
    if label.ndim == 4 and label.shape[1] == 1:
        labels = label[:, 0]
    elif label.ndim == 3:
        labels = label
    else:
        raise ValueError("label must have shape [N,H,W] or [N,1,H,W]")
    if tuple(labels.shape) != (logits.shape[0], logits.shape[2], logits.shape[3]):
        raise ValueError("label spatial/batch shape must match logits")
    if labels.device != logits.device:
        raise ValueError("label and logits must be on the same device")
    if labels.dtype.is_floating_point and not bool(torch.isfinite(labels).all()):
        raise ValueError("label contains non-finite values")
    allowed = (labels == 0) | (labels == 255)
    for class_id in range(1, NUM_CLASSES + 1):
        allowed = allowed | (labels == class_id)
    if not bool(allowed.all()):
        raise ValueError("label values must be 0, 1..4, or ignore value 255")

    if not isinstance(valid, torch.Tensor):
        raise TypeError("valid must be a torch.Tensor")
    if valid.ndim == 3:
        valid = valid[:, None]
    if valid.ndim != 4 or valid.shape[0] != logits.shape[0] or tuple(valid.shape[2:]) != tuple(logits.shape[2:]):
        raise ValueError("valid must have shape [N,H,W], [N,1,H,W], or [N,4,H,W]")
    if valid.shape[1] not in (1, NUM_CLASSES):
        raise ValueError("valid channel dimension must be 1 or 4")
    if valid.device != logits.device:
        raise ValueError("valid and logits must be on the same device")
    if valid.dtype.is_floating_point and not bool(torch.isfinite(valid).all()):
        raise ValueError("valid contains non-finite values")
    if not bool(((valid == 0) | (valid == 1)).all()):
        raise ValueError("valid must contain only binary 0/1 values")

    if valid.shape[1] == 1:
        valid = valid.expand(-1, NUM_CLASSES, -1, -1)
    # A label of 255 is ignored even if a caller accidentally marks it valid.
    valid_mask = valid.to(torch.bool) & (labels[:, None] != 255)
    return labels, valid_mask


def _macro_active_valid(pair_values: list[torch.Tensor], pair_active: list[torch.Tensor], reference: torch.Tensor) -> torch.Tensor:
    """Average over (N,C) pairs with at least one valid pixel."""
    if not pair_values:
        return reference.sum() * 0.0
    values = torch.stack(pair_values, dim=1)  # [N,C]
    active = torch.stack(pair_active, dim=1)  # [N,C], bool
    if not bool(active.any()):
        # Keep an explicit differentiable zero for an all-invalid batch.
        return reference.sum() * 0.0
    active_f = active.to(values.dtype)
    return (values * active_f).sum() / active_f.sum()


def gt_loss(logits: torch.Tensor, label: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """GT loss: active-pair Dice + balanced positive/negative softplus.

    For each sample/class pair, the Dice term is present only when that pair
    contains at least one valid positive.  Positive and negative BCE-like
    groups are independently averaged by their own pixel counts; an absent
    group contributes zero and does not renormalize the other group.
    Valid, fully annotated empty classes remain active and receive negative
    supervision.  All-invalid input returns a graph-connected zero.
    """
    labels, valid_mask = _validate_and_expand(logits, label, valid)
    work = logits.to(dtype=_compute_dtype(logits.dtype))
    pair_values: list[torch.Tensor] = []
    pair_active: list[torch.Tensor] = []

    for class_index in range(NUM_CLASSES):
        z = work[:, class_index : class_index + 1]
        y = (labels[:, None] == class_index + 1).to(work.dtype)
        m = valid_mask[:, class_index : class_index + 1].to(work.dtype)
        positive = m * y
        negative = m * (1.0 - y)

        dims = (1, 2, 3)
        n_pos = positive.sum(dim=dims)
        n_neg = negative.sum(dim=dims)
        n_valid = m.sum(dim=dims)

        probability = torch.sigmoid(z)
        dice_numerator = 2.0 * (m * probability * y).sum(dim=dims) + SMOOTH
        dice_denominator = (m * (probability.square() + y.square())).sum(dim=dims) + SMOOTH
        dice = 1.0 - dice_numerator / dice_denominator
        dice = dice * (n_pos > 0).to(work.dtype)

        positive_term = (positive * F.softplus(-z)).sum(dim=dims) / n_pos.clamp_min(1.0)
        negative_term = (negative * F.softplus(z)).sum(dim=dims) / n_neg.clamp_min(1.0)
        pair_values.append(dice + 0.25 * (positive_term + negative_term))
        pair_active.append(n_valid > 0)

    return _macro_active_valid(pair_values, pair_active, work)


def trusted_kl(
    logits: torch.Tensor,
    teacher: torch.Tensor,
    label: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    """GT-consistent Bernoulli KL from detached teacher probabilities.

    ``teacher`` is q in probability space, not teacher logits.  A pixel is
    trusted only when q's hard decision agrees with GT at threshold 0.5.
    The numerator uses that trust mask, while each pair is normalized by all
    valid pixels (not by the trusted subset), then macro-averaged over active
    valid sample/class pairs.
    """
    labels, valid_mask = _validate_and_expand(logits, label, valid)
    if not isinstance(teacher, torch.Tensor) or tuple(teacher.shape) != tuple(logits.shape):
        raise ValueError("teacher probabilities must have the same [N,4,H,W] shape as logits")
    if teacher.device != logits.device:
        raise ValueError("teacher and logits must be on the same device")
    q = teacher.detach().to(dtype=_compute_dtype(logits.dtype))
    if not bool(torch.isfinite(q).all()):
        raise ValueError("teacher probabilities contain non-finite values")
    if not bool(((q >= 0.0) & (q <= 1.0)).all()):
        raise ValueError("teacher must contain probabilities in [0,1]")

    work = logits.to(dtype=_compute_dtype(logits.dtype))
    pair_values: list[torch.Tensor] = []
    pair_active: list[torch.Tensor] = []
    dims = (1, 2, 3)
    for class_index in range(NUM_CLASSES):
        z = work[:, class_index : class_index + 1]
        q_c = q[:, class_index : class_index + 1]
        y = (labels[:, None] == class_index + 1)
        m = valid_mask[:, class_index : class_index + 1].to(work.dtype)
        trusted = ((y & (q_c > 0.5)) | ((~y) & (q_c <= 0.5))).to(work.dtype)

        # KL(Bernoulli(q) || Bernoulli(sigmoid(z))) = BCEWithLogits(z,q)-H(q).
        # xlogy defines 0*log(0)=0, including exact teacher probabilities 0/1.
        q_entropy_terms = torch.special.xlogy(q_c, q_c) + torch.special.xlogy(1.0 - q_c, 1.0 - q_c)
        kl = F.binary_cross_entropy_with_logits(z, q_c, reduction="none") + q_entropy_terms
        # Roundoff can make exact-zero KL very slightly negative in float32.
        kl = kl.clamp_min(0.0)

        n_valid = m.sum(dim=dims)
        pair_kl = (m * trusted * kl).sum(dim=dims) / n_valid.clamp_min(1.0)
        pair_values.append(pair_kl)
        pair_active.append(n_valid > 0)

    return _macro_active_valid(pair_values, pair_active, work)


def mixture_logits(
    logits_sequence: Sequence[torch.Tensor],
    weights: Sequence[float] = DEFAULT_MIX_WEIGHTS,
) -> torch.Tensor:
    """Return logit of a weighted probability mixture using log-domain math.

    The implementation evaluates log(sum_i w_i*sigmoid(z_i)) and its
    complement with log-sigmoid/logsumexp, avoiding probability clamps that
    would saturate gradients.  Weights must be nonnegative and sum to one.
    """
    try:
        sources = tuple(logits_sequence)
        weight_values = tuple(float(weight) for weight in weights)
    except (TypeError, ValueError) as exc:
        raise TypeError("logits_sequence and weights must be finite sequences") from exc
    if len(sources) != 3:
        raise ValueError("logits_sequence must contain exactly three tensors (A,U,S)")
    if len(weight_values) != 3:
        raise ValueError("weights must contain exactly three values")
    if any(not math.isfinite(weight) or weight < 0.0 for weight in weight_values):
        raise ValueError("weights must be finite and nonnegative")
    if not math.isclose(sum(weight_values), 1.0, rel_tol=0.0, abs_tol=1.0e-8):
        raise ValueError("weights must sum to one within absolute tolerance 1e-8")
    if any(not isinstance(source, torch.Tensor) or source.ndim != 4 for source in sources):
        raise ValueError("each mixture input must be a [N,4,H,W] tensor")
    if any(source.shape[1] != NUM_CLASSES for source in sources):
        raise ValueError(f"each mixture input must have exactly {NUM_CLASSES} channels")
    if any(tuple(source.shape) != tuple(sources[0].shape) for source in sources[1:]):
        raise ValueError("all mixture logits must have identical shapes")
    if any(source.device != sources[0].device for source in sources[1:]):
        raise ValueError("all mixture logits must be on the same device")
    if any(not source.dtype.is_floating_point for source in sources):
        raise TypeError("all mixture logits must have floating dtypes")
    dtype = torch.float64 if any(source.dtype == torch.float64 for source in sources) else torch.float32
    work = torch.stack([source.to(dtype=dtype) for source in sources], dim=0)
    if not bool(torch.isfinite(work).all()):
        raise ValueError("mixture logits must be finite")

    log_weights = torch.tensor(
        [math.log(weight) if weight > 0.0 else -math.inf for weight in weight_values],
        dtype=dtype,
        device=work.device,
    ).view(3, *([1] * (work.ndim - 1)))
    log_probability = torch.logsumexp(F.logsigmoid(work) + log_weights, dim=0)
    log_complement = torch.logsumexp(F.logsigmoid(-work) + log_weights, dim=0)
    return log_probability - log_complement


__all__ = ["gt_loss", "trusted_kl", "mixture_logits", "DEFAULT_MIX_WEIGHTS"]
