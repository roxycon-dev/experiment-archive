"""CPU-only synthetic comparison of gradient accumulation and pooled-batch Dice.

Calls the exact masked_bce_dice_loss implementation from the current project.
Case 1 has different valid-class patterns and shows the full objective difference.
Case 2 keeps all four classes valid in both microbatches so the BCE averaging
matches pooled BCE while nonlinear micro-Dice averaging differs from pooled Dice.
No model, dataset, GPU, server, or optimizer is used.
Run: python audit_dice_accumulation.py
Writes audit_dice_accumulation.json beside this script.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(r"D:\dataset\experiment\HRDecoder_Ready_20261002")
OUTPUT_PATH = Path(__file__).resolve().with_name("audit_dice_accumulation.json")

if not (PROJECT_ROOT / "trainer" / "loss.py").is_file():
    raise FileNotFoundError(f"Expected loss source not found under {PROJECT_ROOT}")
sys.path.insert(0, str(PROJECT_ROOT))
from trainer.loss import masked_bce_dice_loss  # noqa: E402

torch.set_num_threads(1)


def compare(logits_values, target_values, valid_values):
    """Compare mean microbatch losses/gradients with one pooled-batch loss."""
    logits = torch.tensor(
        logits_values, dtype=torch.float32, device="cpu", requires_grad=True
    )
    target = torch.tensor(target_values, dtype=torch.float32, device="cpu")
    valid = torch.tensor(valid_values, dtype=torch.float32, device="cpu")
    batch_size = int(logits.shape[0])

    micro_losses, micro_bce, micro_dice = [], [], []
    for i in range(batch_size):
        loss_i, parts_i = masked_bce_dice_loss(
            logits[i:i + 1], target[i:i + 1], valid[i:i + 1]
        )
        micro_losses.append(float(loss_i.detach()))
        micro_bce.append(float(parts_i["loss_bce"]))
        micro_dice.append(float(parts_i["loss_dice"]))
        (loss_i / float(batch_size)).backward()
    micro_gradient = logits.grad.detach().clone()

    logits.grad.zero_()
    pooled_loss, pooled_parts = masked_bce_dice_loss(logits, target, valid)
    pooled_loss.backward()
    pooled_gradient = logits.grad.detach().clone()

    return {
        "shape": list(logits.shape),
        "grad_accum": batch_size,
        "inputs": {
            "logits": logits_values,
            "target": target_values,
            "valid": valid_values,
        },
        "microbatch_losses": micro_losses,
        "microbatch_mean_loss": sum(micro_losses) / batch_size,
        "microbatch_bce_parts": micro_bce,
        "microbatch_mean_bce": sum(micro_bce) / batch_size,
        "microbatch_dice_parts": micro_dice,
        "microbatch_mean_dice": sum(micro_dice) / batch_size,
        "pooled_batch_loss": float(pooled_loss.detach()),
        "pooled_batch_bce": float(pooled_parts["loss_bce"]),
        "pooled_batch_dice": float(pooled_parts["loss_dice"]),
        "bce_abs_difference": abs(
            sum(micro_bce) / batch_size - float(pooled_parts["loss_bce"])
        ),
        "dice_abs_difference": abs(
            sum(micro_dice) / batch_size - float(pooled_parts["loss_dice"])
        ),
        "max_abs_gradient_difference": float(
            (micro_gradient - pooled_gradient).abs().max().item()
        ),
        "microbatch_accumulated_gradient": micro_gradient.tolist(),
        "pooled_batch_gradient": pooled_gradient.tolist(),
    }


# Case 1: EX is valid in both samples, HE only in sample 2; SE/MA are invalid.
# This represents variation in valid-class patterns, so its total-loss difference
# combines Dice nonlinearity and per-microbatch active-class weighting.
case1_logits = [
    [
        [[0.8, -0.4]],
        [[0.0, 0.0]],
        [[0.1, -0.1]],
        [[-0.2, 0.2]],
    ],
    [
        [[-0.2, 1.1]],
        [[0.3, -0.7]],
        [[0.0, 0.0]],
        [[0.0, 0.0]],
    ],
]
case1_target = [
    [
        [[1.0, 0.0]],
        [[0.0, 0.0]],
        [[0.0, 0.0]],
        [[0.0, 0.0]],
    ],
    [
        [[0.0, 1.0]],
        [[1.0, 0.0]],
        [[0.0, 0.0]],
        [[0.0, 0.0]],
    ],
]
case1_valid = [
    [
        [[1.0, 1.0]],
        [[0.0, 0.0]],
        [[0.0, 0.0]],
        [[0.0, 0.0]],
    ],
    [
        [[1.0, 1.0]],
        [[1.0, 1.0]],
        [[0.0, 0.0]],
        [[0.0, 0.0]],
    ],
]
case1 = compare(case1_logits, case1_target, case1_valid)
case1["purpose"] = (
    "Full loss comparison with different active-class patterns; do not attribute "
    "the total difference to Dice alone."
)

# Case 2: all pixels of all four classes are valid in both microbatches.
# Positive pixel density differs: 2/4 in sample 1 and 1/4 in sample 2.
# Every class has the same pixel count in each microbatch, so mean micro-BCE
# equals pooled BCE; only the nonlinear Dice ratio is grouped differently.
class_offsets = [0.0, 0.15, -0.1, 0.25]
case2_logits = [
    [
        [[0.8 + off, -0.3 + off, 0.5 + off, -0.7 + off]]
        for off in class_offsets
    ],
    [
        [[0.4 + off, 0.1 + off, -0.6 + off, -1.0 + off]]
        for off in class_offsets
    ],
]
case2_target = [
    [[[1.0, 1.0, 0.0, 0.0]] for _ in range(4)],
    [[[1.0, 0.0, 0.0, 0.0]] for _ in range(4)],
]
case2_valid = [
    [[[1.0, 1.0, 1.0, 1.0]] for _ in range(4)],
    [[[1.0, 1.0, 1.0, 1.0]] for _ in range(4)],
]
case2 = compare(case2_logits, case2_target, case2_valid)
case2["purpose"] = (
    "Dice-only grouping isolation: every class is valid in both microbatches, "
    "with foreground density 2/4 versus 1/4; BCE is pooled-equivalent."
)
case2["all_classes_valid_in_both_microbatches"] = True
case2["foreground_density_per_microbatch_per_class"] = [0.5, 0.25]
if case2["bce_abs_difference"] > 1e-7:
    raise AssertionError("BCE should match when every class has equal valid-pixel counts")
if case2["dice_abs_difference"] <= 1e-5:
    raise AssertionError("Chosen tensors did not produce a meaningful Dice difference")

result = {
    "device": "cpu",
    "torch_version": str(torch.__version__),
    "source_loss_file": str(PROJECT_ROOT / "trainer" / "loss.py"),
    "source_function": "masked_bce_dice_loss",
    "comparison": (
        "mean of independent microbatch losses/gradients versus one loss "
        "computed on the pooled batch"
    ),
    "case_1_varying_active_classes": case1,
    "case_2_all_valid_dice_isolation": case2,
}

OUTPUT_PATH.write_text(
    json.dumps(result, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
)
print(json.dumps(result, indent=2, ensure_ascii=False))

