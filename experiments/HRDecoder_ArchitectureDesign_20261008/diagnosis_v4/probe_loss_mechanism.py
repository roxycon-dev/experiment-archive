"""CPU-only loss-mechanism probe for HRDecoder v3.

This script does not import the training runner, load model/data/checkpoints, or
step an optimizer. It re-implements the audited scalar formulas below so it can
run independently, then checks the Dice formula with PyTorch autograd on small
CPU tensors. Full 2048x2048 results are closed-form in pixel count.

Run from any directory:
    python path/to/probe_loss_mechanism.py
The default output is evidence/loss_mechanism_probe.json next to this script.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

EPS = 1e-5
P2048 = 2048 * 2048


def sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))


def dice_closed_form(pixel_count: int, positive_count: int, logit: float) -> dict[str, float]:
    """One image/class with a uniform logit; derivative is for a shared logit shift."""
    p = sigmoid(logit)
    k = float(positive_count)
    n = 2.0 * p * k + EPS
    d = pixel_count * p * p + k + EPS
    loss = 1.0 - n / d
    dn = 2.0 * k * p * (1.0 - p)
    dd = 2.0 * pixel_count * p * p * (1.0 - p)
    grad_shift = -(dn * d - n * dd) / (d * d)
    return {
        "p": p,
        "positive_pixels": k,
        "positive_fraction": k / pixel_count,
        "intersection": p * k,
        "predicted_p2_sum": pixel_count * p * p,
        "numerator": n,
        "denominator": d,
        "dice_loss": loss,
        "grad_wrt_uniform_logit_shift": grad_shift,
    }


def empty_dice_closed_form(pixel_count: int, logit: float) -> dict[str, float]:
    x = pixel_count * sigmoid(logit) ** 2
    d = x + EPS
    return {
        "p": sigmoid(logit),
        "predicted_p2_sum": x,
        "dice_loss": x / d,
        "grad_wrt_uniform_logit_shift": 2.0 * EPS * x * (1.0 - sigmoid(logit)) / (d * d),
    }


def empty_dice_sparse_peak(pixel_count: int, peak_logit: float, rest_logit: float, peak_pixels: int = 1) -> dict[str, float]:
    p_peak = sigmoid(peak_logit)
    p_rest = sigmoid(rest_logit)
    p2_sum = peak_pixels * p_peak * p_peak + (pixel_count - peak_pixels) * p_rest * p_rest
    return {
        "peak_logit": peak_logit,
        "rest_logit": rest_logit,
        "peak_pixels": peak_pixels,
        "maximum_logit": peak_logit,
        "predicted_p2_sum": p2_sum,
        "empty_class_dice_loss": p2_sum / (p2_sum + EPS),
    }


def bernoulli_kl(q: float, p: float) -> float:
    terms = []
    if q > 0.0:
        terms.append(q * math.log(q / p))
    if q < 1.0:
        terms.append((1.0 - q) * math.log((1.0 - q) / (1.0 - p)))
    return sum(terms)


def kl_closed_form(pixel_count: int, positive_fraction: float, positive_teacher_q: float, logit: float) -> dict[str, float]:
    p = sigmoid(logit)
    f = positive_fraction
    mean_q = f * positive_teacher_q
    return {
        "student_p": p,
        "teacher_positive_q": positive_teacher_q,
        "teacher_positive_fraction": f,
        "mean_teacher_probability": mean_q,
        "mean_bernoulli_kl": f * bernoulli_kl(positive_teacher_q, p) + (1.0 - f) * bernoulli_kl(0.0, p),
        "grad_wrt_uniform_logit_shift": p - mean_q,
        "positive_fraction_needed_for_nonpositive_bias_gradient": p / positive_teacher_q if positive_teacher_q > 0 else None,
        "pixel_count_for_normalization": pixel_count,
    }


def signed_aux_closed_form(positive_fraction: float, difference: float, beta: float = 0.1) -> dict[str, float]:
    """One contrast head differs on a sparse support; the second has zero target."""
    f = positive_fraction
    a = abs(difference)
    weight_support = 1.0 + 4.0 * a
    mean_weight = 1.0 + 4.0 * a * f
    if a < beta:
        point_loss = 0.5 * a * a / beta
        point_grad_mag = a / beta
    else:
        point_loss = a - 0.5 * beta
        point_grad_mag = 1.0
    # signed_aux averages two heads; only the first has this target.
    loss = 0.5 * (f * weight_support * point_loss / mean_weight)
    grad_mag = 0.5 * (f * weight_support * point_grad_mag / mean_weight)
    return {
        "difference_magnitude": a,
        "difference_support_fraction": f,
        "weight_on_support": weight_support,
        "weight_mean": mean_weight,
        "signed_aux_loss_at_zero_prediction": loss,
        "grad_magnitude_wrt_uniform_dhat_shift": grad_mag,
        "grad_magnitude_after_lambda_d_0_1": 0.1 * grad_mag,
    }


def current_masked_dice_small(logits: torch.Tensor, label: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Minimal copy of objectives.py masked_dice for a single binary class."""
    logits = logits.float()
    probability = logits.sigmoid()
    target = label.to(probability.dtype)
    mask = valid.float().expand_as(probability)
    numerator = 2.0 * (probability * target * mask).sum((-2, -1)) + EPS
    denominator = ((probability.square() + target.square()) * mask).sum((-2, -1)) + EPS
    return (1.0 - numerator / denominator).mean()


def original_binary_dice_small(logits: torch.Tensor, label: torch.Tensor, smooth: float = EPS) -> torch.Tensor:
    """Minimal copy of binary_dice_loss (one image/class) from HRDecoder."""
    pred = logits.sigmoid().contiguous().view(logits.shape[0], -1).float()
    target = label.contiguous().view(label.shape[0], -1).float()
    numerator = 2.0 * (pred * target).sum(1) + smooth
    denominator = (pred.pow(2) + target.pow(2)).sum(1) + smooth
    return (1.0 - numerator / denominator).mean()


def autograd_probe() -> dict:
    h = w = 8
    valid = torch.ones((1, 1, h, w), dtype=torch.float32, device="cpu")
    result = {}
    for positive_count in (0, 1, 4):
        label = torch.zeros((1, 1, h, w), dtype=torch.float32, device="cpu")
        if positive_count:
            label.reshape(-1)[:positive_count] = 1.0
        z = torch.full((1, 1, h, w), -10.55, dtype=torch.float32, device="cpu", requires_grad=True)
        loss = current_masked_dice_small(z, label, valid)
        grad = torch.autograd.grad(loss, z)[0]
        form = dice_closed_form(h * w, positive_count, -10.55)
        original = original_binary_dice_small(z.detach(), label)
        result[str(positive_count)] = {
            "shape": [1, 1, h, w],
            "logit": -10.55,
            "masked_dice_loss_torch": float(loss.item()),
            "masked_dice_grad_sum_torch": float(grad.sum().item()),
            "masked_dice_loss_closed_form": form["dice_loss"],
            "masked_dice_grad_sum_closed_form": form["grad_wrt_uniform_logit_shift"],
            "original_hrdecoder_dice_loss_torch": float(original.item()),
            "difference_current_vs_original": float((loss.detach() - original).item()),
        }

    # Verify the old implementation does not mask a label=255 pixel: target is
    # all-zero there, but its predicted p^2 remains in the old denominator.
    logits = torch.full((1, 1, 2, 2), -2.0, dtype=torch.float32)
    labels_with_void = torch.tensor([[[[1.0, 0.0], [0.0, 0.0]]]])
    logits[0, 0, 1, 1] = 2.0
    old_unmasked = original_binary_dice_small(logits, labels_with_void)
    valid_void = torch.ones((1, 1, 2, 2), dtype=torch.float32)
    valid_void[0, 0, 1, 1] = 0.0
    new_masked = current_masked_dice_small(logits, labels_with_void, valid_void)
    result["conditional_ignore_pixel_demo"] = {
        "description": "A high-probability pixel is voided by valid=0 for current masked Dice; old Dice has only a zero target and still counts p^2.",
        "old_dice_loss_no_ignore_mask": float(old_unmasked.item()),
        "current_dice_loss_with_valid_mask": float(new_masked.item()),
        "label_255_presence_in_real_training_data_was_not_checked": True,
    }
    return result


def main() -> None:
    logits = [-10.55, -12.34, -15.0, -20.0]
    fractions = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    prevalence_rates = [0.01, 0.1, 0.5, 0.9, 0.99]
    cases = []
    for z in logits:
        empty = empty_dice_closed_form(P2048, z)
        positive = []
        for f in fractions:
            k = max(1, int(round(P2048 * f)))
            stats = dice_closed_form(P2048, k, z)
            g_empty = empty["grad_wrt_uniform_logit_shift"]
            g_positive = stats["grad_wrt_uniform_logit_shift"]
            r_critical = g_empty / (g_empty - g_positive) if g_positive < 0 else None
            positive.append({**stats, "positive_case_fraction_for_net_raise_threshold": r_critical})
        g_empty = empty["grad_wrt_uniform_logit_shift"]
        representative_k = max(1, int(round(P2048 * 1e-3)))
        rep = dice_closed_form(P2048, representative_k, z)
        mixed_rates = []
        for r in prevalence_rates:
            mean_dice_loss = (1.0 - r) * empty["dice_loss"] + r * rep["dice_loss"]
            mixed_rates.append({
                "positive_case_rate": r,
                "mean_dice_loss_at_positive_fraction_0_001": mean_dice_loss,
                "mean_grad_shift_from_dice": (1.0 - r) * g_empty + r * rep["grad_wrt_uniform_logit_shift"],
                "direction": "lower_logits" if (1.0 - r) * g_empty + r * rep["grad_wrt_uniform_logit_shift"] > 0 else "raise_logits",
            })
        cases.append({
            "uniform_logit": z,
            "empty_class": empty,
            "positive_class_by_pixel_fraction": positive,
            "mixed_empty_positive_case_rate_at_fraction_0_001": mixed_rates,
        })

    kl = []
    for z in (-10.55, -12.34):
        for f in fractions:
            kl.append({"uniform_logit": z, **kl_closed_form(P2048, f, 0.9, z)})
    # Teacher q=0 on every valid pixel is included to expose the empty-teacher direction.
    for z in (-10.55, -12.34):
        kl.append({"uniform_logit": z, **kl_closed_form(P2048, 0.0, 0.9, z)})

    signed_aux = []
    for f in (1e-6, 1e-5, 1e-4, 1e-3, 1e-2):
        for d in (0.01, 0.1, 1.0):
            signed_aux.append(signed_aux_closed_form(f, d))

    valid_area_sensitivity = []
    for area_fraction in (0.25, 0.5, 1.0):
        valid_pixels = int(round(P2048 * area_fraction))
        for z in (-10.55, -12.34, -20.0):
            k = max(1, int(round(valid_pixels * 1e-3)))
            valid_area_sensitivity.append({
                "valid_area_fraction_of_2048_canvas": area_fraction,
                "valid_pixels": valid_pixels,
                "uniform_logit": z,
                "empty_class": empty_dice_closed_form(valid_pixels, z),
                "positive_class_fraction": 1e-3,
                "positive_pixels": k,
                "positive_class": dice_closed_form(valid_pixels, k, z),
            })

    maxlogit_only_examples = []
    for peak in (-10.55, -12.34):
        maxlogit_only_examples.append({
            "maximum_logit_is_peak_and_all_other_pixels_are_at_minus20": empty_dice_sparse_peak(P2048, peak, -20.0, 1),
            "uniform_at_maximum": empty_dice_closed_form(P2048, peak),
        })

    output = {
        "probe": "loss_mechanism_audit_v1",
        "device": "cpu",
        "torch_version": torch.__version__,
        "training_or_model_inference": False,
        "optimizer_steps": 0,
        "real_images_or_checkpoints_loaded": False,
        "source_formulas": {
            "current": "training_execution/objectives.py: masked_dice, masked_kl, signed_aux",
            "historical": "autodl100/work/HRDecoder/mmseg/models/losses/binary_loss.py: binary_dice_loss, binary_loss",
            "dice_epsilon_numerator_and_denominator": EPS,
            "dice_probability_denominator_exponent": 2,
            "full_canvas": [2048, 2048],
            "full_canvas_pixels": P2048,
            "logits_are_uniform_synthetic_probes": True,
        },
        "small_tensor_autograd_crosscheck": autograd_probe(),
        "full_canvas_dice": cases,
        "full_canvas_kl_teacher_q_0_9_on_lesion_support": kl,
        "signed_aux_sparse_teacher_difference": signed_aux,
        "valid_area_sensitivity": valid_area_sensitivity,
        "same_max_logit_different_pixel_distribution": maxlogit_only_examples,
        "interpretation_limits": [
            "Uniform-logit calculations isolate the loss mechanism; real model logits are spatially varying.",
            "The reported EX/SE values are maxima supplied for this audit; maxima alone do not specify the pixel distribution or mean gradients.",
            "Synthetic lesion fractions and positive-case rates are examples, not measured dataset prevalence.",
            "This establishes a plausible objective-level mechanism, not causal attribution of the 20k validation miss.",
        ],
    }
    output_path = Path(__file__).resolve().parent / "evidence" / "loss_mechanism_probe.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(output_path)
    print(json.dumps({
        "autograd": output["small_tensor_autograd_crosscheck"],
        "dice_summary": [
            {"z": x["uniform_logit"], "empty_loss": x["empty_class"]["dice_loss"], "empty_grad": x["empty_class"]["grad_wrt_uniform_logit_shift"],
             "positive_loss_f_1e-3": next(y["dice_loss"] for y in x["positive_class_by_pixel_fraction"] if abs(y["positive_fraction"] - 1e-3) < 1e-5),
             "positive_grad_f_1e-3": next(y["grad_wrt_uniform_logit_shift"] for y in x["positive_class_by_pixel_fraction"] if abs(y["positive_fraction"] - 1e-3) < 1e-5),
             "raise_threshold_f_1e-3": next(y["positive_case_fraction_for_net_raise_threshold"] for y in x["positive_class_by_pixel_fraction"] if abs(y["positive_fraction"] - 1e-3) < 1e-5)}
            for x in cases
        ],
        "json": str(output_path),
    }, indent=2))


if __name__ == "__main__":
    torch.set_num_threads(1)
    main()
