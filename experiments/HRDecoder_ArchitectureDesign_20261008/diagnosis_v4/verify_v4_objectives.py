"""CPU-only synthetic checks for the isolated v4 objective reference.

Run from any directory with the workspace Python:
    python verify_v4_objectives.py

This script creates no model, dataset, optimizer, checkpoint, or training run.
It writes only evidence/v4_objective_checks.json beside this file.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys
from typing import Any

import torch
import torch.nn.functional as F

from v4_objectives import DEFAULT_MIX_WEIGHTS, gt_loss, mixture_logits, trusted_kl


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "evidence" / "v4_objective_checks.json"
ATOL = 1.0e-6


def _check(name: str, condition: bool, details: dict[str, Any]) -> dict[str, Any]:
    row = {"name": name, "status": "PASS" if condition else "FAIL", **details}
    if not condition:
        raise AssertionError(f"{name}: {details}")
    return row


def _logit(probability: torch.Tensor) -> torch.Tensor:
    return torch.log(probability) - torch.log1p(-probability)


def _old_dice_only(logits: torch.Tensor, label: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Historical-style per-pair Dice-only surrogate used for gradient contrast."""
    terms = []
    for c in range(4):
        y = (label[:, None] == c + 1).to(logits.dtype)
        m = valid[:, None].to(logits.dtype)
        p = torch.sigmoid(logits[:, c : c + 1])
        npos = (m * y).sum(dim=(1, 2, 3))
        d = 1.0 - (2.0 * (m * p * y).sum(dim=(1, 2, 3)) + 1.0e-5) / (
            (m * (p.square() + y.square())).sum(dim=(1, 2, 3)) + 1.0e-5
        )
        active = m.sum(dim=(1, 2, 3)) > 0
        d = d * (npos > 0).to(d.dtype)
        terms.append((d * active).sum() / active.sum().clamp_min(1))
    return torch.stack(terms).mean()


def run_checks() -> list[dict[str, Any]]:
    if torch.cuda.is_initialized():
        raise RuntimeError("verifier must remain CPU-only; CUDA was initialized")
    torch.manual_seed(20261009)
    results: list[dict[str, Any]] = []

    # Extreme but finite logits exercise softplus, BCEWithLogits, and the
    # log-domain mixture without allowing probability clamps to hide gradients.
    label = torch.tensor([[[1, 0]]], dtype=torch.long)
    valid = torch.ones_like(label, dtype=torch.bool)
    z_extreme = torch.tensor(
        [[[[1000.0, -1000.0]], [[-1000.0, 1000.0]], [[1000.0, -1000.0]], [[-1000.0, 1000.0]]]],
        dtype=torch.float32,
        requires_grad=True,
    )
    loss_extreme = gt_loss(z_extreme, label, valid)
    loss_extreme.backward()
    gt_extreme_finite = bool(torch.isfinite(loss_extreme) and torch.isfinite(z_extreme.grad).all())

    z_kd_extreme = z_extreme.detach().clone().requires_grad_(True)
    q_extreme = torch.full_like(z_kd_extreme, 0.9)
    kd_extreme = trusted_kl(z_kd_extreme, q_extreme, label, valid)
    kd_extreme.backward()
    kd_extreme_finite = bool(torch.isfinite(kd_extreme) and torch.isfinite(z_kd_extreme.grad).all())

    mix_extreme_inputs = [
        z_extreme.detach().clone().requires_grad_(True),
        (-z_extreme.detach()).clone().requires_grad_(True),
        torch.zeros_like(z_extreme.detach(), requires_grad=True),
    ]
    mix_extreme = mixture_logits(mix_extreme_inputs)
    mix_extreme.sum().backward()
    mix_extreme_finite = bool(
        torch.isfinite(mix_extreme).all()
        and all(source.grad is not None and torch.isfinite(source.grad).all() for source in mix_extreme_inputs)
    )
    results.append(
        _check(
            "extreme_logits_finite_and_differentiable",
            gt_extreme_finite and kd_extreme_finite and mix_extreme_finite,
            {
                "gt_loss_at_logits_pm1000": float(loss_extreme.detach()),
                "trusted_kl_at_logits_pm1000": float(kd_extreme.detach()),
                "mixture_logits_range": [float(mix_extreme.min()), float(mix_extreme.max())],
                "all_losses_and_gradients_finite": True,
            },
        )
    )

    # A missed positive at z=-20 must receive a useful upward gradient; pure
    # Dice is nearly flat there because sigmoid(z) is saturated near zero.
    z_missed = torch.full((1, 4, 1, 2), -20.0, requires_grad=True)
    missed_label = torch.tensor([[[1, 0]]], dtype=torch.long)
    missed_valid = torch.ones_like(missed_label, dtype=torch.bool)
    missed_loss = gt_loss(z_missed, missed_label, missed_valid)
    missed_grad = torch.autograd.grad(missed_loss, z_missed)[0][0, 0, 0, 0]
    z_old = z_missed.detach().clone().requires_grad_(True)
    old_loss = _old_dice_only(z_old, missed_label, missed_valid)
    old_grad = torch.autograd.grad(old_loss, z_old)[0][0, 0, 0, 0]
    gradient_ratio = abs(float(missed_grad)) / max(abs(float(old_grad)), 1.0e-30)
    results.append(
        _check(
            "missed_positive_gets_recovery_gradient",
            float(missed_grad) < 0.0 and gradient_ratio > 1.0e6,
            {
                "new_loss_gradient_at_positive_z_minus20": float(missed_grad),
                "dice_only_gradient_at_positive_z_minus20": float(old_grad),
                "absolute_gradient_ratio": gradient_ratio,
                "gradient_descent_direction": "increase positive logit",
            },
        )
    )

    # Empty but fully annotated classes retain negative BCE-like supervision.
    z_empty = torch.full((1, 4, 1, 2), 2.0, requires_grad=True)
    empty_label = torch.zeros((1, 1, 2), dtype=torch.long)
    empty_valid = torch.ones_like(empty_label, dtype=torch.bool)
    empty_loss = gt_loss(z_empty, empty_label, empty_valid)
    empty_grad = torch.autograd.grad(empty_loss, z_empty)[0]
    expected_empty_loss = 0.25 * float(F.softplus(torch.tensor(2.0)))
    results.append(
        _check(
            "fully_labeled_empty_class_has_negative_supervision",
            float(empty_loss) > 0.0
            and bool((empty_grad > 0.0).all())
            and math.isclose(float(empty_loss), expected_empty_loss, rel_tol=1e-6, abs_tol=1e-6),
            {
                "loss": float(empty_loss),
                "expected_0_25_softplus_2": expected_empty_loss,
                "all_logit_gradients_positive": bool((empty_grad > 0.0).all()),
            },
        )
    )

    # All-invalid batches are a differentiable zero for both objectives.
    z_invalid = torch.randn((1, 4, 1, 3), requires_grad=True)
    invalid_label = torch.tensor([[[1, 2, 0]]], dtype=torch.long)
    invalid_mask = torch.zeros_like(invalid_label, dtype=torch.bool)
    invalid_gt = gt_loss(z_invalid, invalid_label, invalid_mask)
    invalid_gt.backward()
    gt_zero_ok = float(invalid_gt) == 0.0 and z_invalid.grad is not None and bool((z_invalid.grad == 0).all())
    z_invalid_kd = torch.randn((1, 4, 1, 3), requires_grad=True)
    invalid_kd = trusted_kl(z_invalid_kd, torch.full_like(z_invalid_kd, 0.5), invalid_label, invalid_mask)
    invalid_kd.backward()
    kd_zero_ok = float(invalid_kd) == 0.0 and z_invalid_kd.grad is not None and bool((z_invalid_kd.grad == 0).all())
    results.append(
        _check(
            "all_invalid_returns_backwardable_graph_zero",
            gt_zero_ok and kd_zero_ok,
            {"gt_loss": float(invalid_gt), "trusted_kl": float(invalid_kd), "zero_gradients": True},
        )
    )

    # Ignore value 255 and valid=0 both remove a pixel from every term.
    logits_a = torch.tensor([[[[0.7, -2.0]], [[-0.4, 1.2]], [[0.1, 3.0]], [[-1.0, -0.2]]]])
    logits_b = logits_a.clone()
    logits_b[0, :, 0, 1] = torch.tensor([-900.0, 800.0, -700.0, 600.0])
    labels_ignore = torch.tensor([[[1, 255]]], dtype=torch.long)
    valid_ignore = torch.ones_like(labels_ignore, dtype=torch.bool)
    labels_masked = torch.tensor([[[1, 0]]], dtype=torch.long)
    valid_masked = torch.tensor([[[1, 0]]], dtype=torch.bool)
    gt_ignore = gt_loss(logits_a, labels_ignore, valid_ignore)
    gt_masked = gt_loss(logits_b, labels_masked, valid_masked)
    teacher_ignore = torch.full_like(logits_a, 0.7)
    teacher_masked = teacher_ignore.clone()
    teacher_masked[..., 1] = 0.99
    kd_ignore = trusted_kl(logits_a, teacher_ignore, labels_ignore, valid_ignore)
    kd_masked = trusted_kl(logits_b, teacher_masked, labels_masked, valid_masked)
    ignore_delta_gt = abs(float(gt_ignore - gt_masked))
    ignore_delta_kd = abs(float(kd_ignore - kd_masked))
    results.append(
        _check(
            "ignore_and_zero_valid_pixels_have_no_effect",
            ignore_delta_gt < 1.0e-7 and ignore_delta_kd < 1.0e-7,
            {"gt_loss_abs_delta": ignore_delta_gt, "trusted_kl_abs_delta": ignore_delta_kd},
        )
    )

    # If every teacher hard decision contradicts GT, no pixel is trusted.
    wrong_label = torch.tensor([[[1, 0]]], dtype=torch.long)
    wrong_valid = torch.ones_like(wrong_label, dtype=torch.bool)
    wrong_teacher = torch.full((1, 4, 1, 2), 0.5)
    wrong_teacher[:, 0, :, 0] = 0.49  # positive GT, teacher predicts negative
    wrong_teacher[:, 0, :, 1] = 0.51  # negative GT, teacher predicts positive
    wrong_teacher[:, 1:, :, :] = 0.51  # negative GT, teacher predicts positive
    wrong_kd_logits = torch.randn_like(wrong_teacher, requires_grad=True)
    wrong_kd = trusted_kl(wrong_kd_logits, wrong_teacher, wrong_label, wrong_valid)
    wrong_kd.backward()
    results.append(
        _check(
            "wrong_teacher_hard_decisions_are_excluded",
            float(wrong_kd) == 0.0 and bool((wrong_kd_logits.grad == 0).all()),
            {"loss": float(wrong_kd), "all_teacher_decisions_wrong": True, "student_gradient_zero": True},
        )
    )

    # Only one of two class-1 pixels is trusted. Its KL is divided by both
    # valid pixels, then by four active classes in the macro average.
    trusted_label = torch.tensor([[[1, 1]]], dtype=torch.long)
    trusted_valid = torch.ones_like(trusted_label, dtype=torch.bool)
    q = torch.full((1, 4, 1, 2), 0.1)
    q[:, 0, 0, 0] = 0.9
    q[:, 0, 0, 1] = 0.1  # wrong hard decision, excluded
    student = _logit(q).detach()
    student[:, 0, 0, 0] = -4.0  # teacher is right, student is wrong
    student.requires_grad_(True)
    q.requires_grad_(True)
    trusted_loss = trusted_kl(student, q, trusted_label, trusted_valid)
    trusted_grad = torch.autograd.grad(trusted_loss, student, retain_graph=True)[0]
    teacher_grad = torch.autograd.grad(trusted_loss, q, allow_unused=True)[0]
    per_pixel_kl = F.binary_cross_entropy_with_logits(
        torch.tensor(-4.0), torch.tensor(0.9), reduction="none"
    ) + torch.special.xlogy(torch.tensor(0.9), torch.tensor(0.9)) + torch.special.xlogy(
        torch.tensor(0.1), torch.tensor(0.1)
    )
    expected_trusted_loss = float(per_pixel_kl) / 2.0 / 4.0
    results.append(
        _check(
            "correct_teacher_wrong_student_receives_detached_kl",
            float(trusted_loss) > 0.0
            and math.isclose(float(trusted_loss), expected_trusted_loss, rel_tol=1e-6, abs_tol=1e-6)
            and float(trusted_grad[0, 0, 0, 0]) < 0.0
            and teacher_grad is None,
            {
                "loss": float(trusted_loss),
                "expected_one_trusted_pixel_over_two_valid_then_four_classes": expected_trusted_loss,
                "gradient_at_wrong_student_positive_gt": float(trusted_grad[0, 0, 0, 0]),
                "teacher_receives_gradient": teacher_grad is not None,
            },
        )
    )

    # Verify exact probability-space mixing against direct arithmetic at
    # moderate logits, and confirm the anchor contribution is a lower bound.
    mix_sources = [torch.randn((1, 4, 2, 3), dtype=torch.float64) * 3.0 for _ in range(3)]
    mix_out = mixture_logits(mix_sources)
    weights = torch.tensor(DEFAULT_MIX_WEIGHTS, dtype=torch.float64).view(3, 1, 1, 1, 1)
    direct_probability = (weights * torch.stack([torch.sigmoid(z) for z in mix_sources])).sum(dim=0)
    direct_logit = torch.log(direct_probability) - torch.log1p(-direct_probability)
    mix_error = float((mix_out - direct_logit).abs().max())
    anchor_gap = float((torch.sigmoid(mix_out) - DEFAULT_MIX_WEIGHTS[0] * torch.sigmoid(mix_sources[0])).min())
    results.append(
        _check(
            "mixture_matches_probability_algebra_and_anchor_lower_bound",
            mix_error < 1.0e-12 and anchor_gap >= -1.0e-12,
            {"max_abs_logit_error_vs_direct": mix_error, "minimum_anchor_lower_bound_gap": anchor_gap},
        )
    )

    # Equal-source initialization is invariant. If auxiliary logits are zero,
    # however, fixed weights inject 0.4375 * 0.5 probability and generally do
    # not reproduce the anchor; that non-equivalence is an explicit check.
    anchor = torch.tensor([[[[-2.0, 1.0]], [[0.4, -0.8]], [[1.5, -3.0]], [[0.2, 2.0]]]])
    equal_sources = mixture_logits([anchor, anchor, anchor])
    equal_error = float((equal_sources - anchor).abs().max())
    zero_aux_mix = mixture_logits([anchor, torch.zeros_like(anchor), torch.zeros_like(anchor)])
    expected_b0 = torch.logit(
        DEFAULT_MIX_WEIGHTS[0] * torch.sigmoid(anchor)
        + (DEFAULT_MIX_WEIGHTS[1] + DEFAULT_MIX_WEIGHTS[2]) * 0.5
    )
    b0_error = float((zero_aux_mix - expected_b0).abs().max())
    non_equivalence = float((zero_aux_mix - anchor).abs().max())
    results.append(
        _check(
            "mixture_initialization_equivalence_and_zero_aux_non_equivalence",
            equal_error < 1.0e-6 and b0_error < 1.0e-6 and non_equivalence > 0.1,
            {
                "equal_source_max_abs_error": equal_error,
                "zero_aux_matches_probability_formula_max_abs_error": b0_error,
                "zero_aux_vs_anchor_max_abs_difference": non_equivalence,
                "fixed_weights": list(DEFAULT_MIX_WEIGHTS),
            },
        )
    )

    zero_weight_result = mixture_logits([anchor, torch.zeros_like(anchor), torch.zeros_like(anchor)], (1.0, 0.0, 0.0))
    bad_weight_checks = []
    for bad_weights in ((0.5, 0.25, 0.1), (-0.1, 0.6, 0.5)):
        try:
            mixture_logits([anchor, anchor, anchor], bad_weights)
        except ValueError:
            bad_weight_checks.append(True)
        else:
            bad_weight_checks.append(False)
    results.append(
        _check(
            "custom_mixture_weights_are_validated",
            float((zero_weight_result - anchor).abs().max()) < 1.0e-6 and all(bad_weight_checks),
            {
                "unit_anchor_weight_max_abs_error": float((zero_weight_result - anchor).abs().max()),
                "rejects_nonunit_sum_and_negative_weight": all(bad_weight_checks),
            },
        )
    )

    return results


def main() -> int:
    results = run_checks()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "title": "v4 objective CPU synthetic checks",
        "device": "cpu",
        "torch_version": torch.__version__,
        "python_version": sys.version.split()[0],
        "training_started": False,
        "checkpoint_or_dataset_loaded": False,
        "checks_passed": sum(row["status"] == "PASS" for row in results),
        "checks_total": len(results),
        "results": results,
    }
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "checks_passed": report["checks_passed"], "checks_total": report["checks_total"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
