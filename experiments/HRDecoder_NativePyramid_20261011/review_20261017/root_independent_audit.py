"""Root-authored CPU-only arithmetic/log audit; no model or pixel evaluation."""
from pathlib import Path
import hashlib
import json
import math
import statistics


REVIEW = Path(__file__).resolve().parent
RUN = REVIEW.parent
WORKSPACE = RUN.parent
CLASSES = ["EX", "HE", "SE", "MA"]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def counts_sum(rows):
    return [[sum(row[c][k] for row in rows) for k in range(3)] for c in range(4)]


def ious(counts):
    return [100 * tp / (tp + fp + fn) for tp, fp, fn in counts]


def main():
    final_path = RUN / "server_execution_20261016/completion_20261017/final_validation.json"
    fusion_path = WORKSPACE / "HRDecoder_FinalDecision_20261005/execution/server_results/runs/S_seed42/val_u20000/metrics.json"
    baseline_path = WORKSPACE / "HRDecoder_FinalDecision_20261005/结题报告/evidence/original_HRDecoder_baseline_evidence.json"
    log_path = REVIEW / "recovered_logs/train_metrics_all40000.jsonl"
    manifest_path = REVIEW / "recovered_logs/train_metrics_all40000.manifest.json"
    final, fusion, baseline, manifest = map(read_json, [final_path, fusion_path, baseline_path, manifest_path])
    current = {r["sid"]: r["heads"]["Z"] for r in final["per_image"]}
    target = {r["sid"]: r["fixed_fusion"] for r in fusion["per_image"]}
    assert len(current) == len(target) == 76 and current.keys() == target.keys()
    for sid in current:
        for c in range(4):
            assert current[sid][c][0] + current[sid][c][2] == target[sid][c][0] + target[sid][c][2]
    current_sum, target_sum = counts_sum(current.values()), counts_sum(target.values())
    assert current_sum == final["heads"]["Z"]["counts"]
    assert target_sum == fusion["fixed_fusion"]["counts"]
    current_iou, target_iou = ious(current_sum), ious(target_sum)
    assert math.isclose(statistics.mean(current_iou), final["mIoU_percent"], abs_tol=1e-10)
    assert math.isclose(statistics.mean(target_iou), fusion["fixed_fusion"]["miou_pct"], abs_tol=1e-10)
    per_class = {}
    for c, name in enumerate(CLASSES):
        tp, fp, fn = current_sum[c]
        target_tp, target_fp, _ = target_sum[c]
        threshold = target_iou[c] / 100
        gt = tp + fn
        absent = [sid for sid in current if current[sid][c][0] + current[sid][c][2] == 0]
        absent_fp = sum(current[sid][c][1] for sid in absent)
        target_absent_fp = sum(target[sid][c][1] for sid in absent)
        per_class[name] = {
            "v3_iou_percent": current_iou[c], "target_iou_percent": target_iou[c],
            "gap_pp": target_iou[c] - current_iou[c],
            "v3_counts": current_sum[c], "target_counts": target_sum[c],
            "target_minus_v3_tp": target_tp - tp, "target_minus_v3_fp": target_fp - fp,
            "required_delta_tp_minus_target_iou_times_delta_fp": threshold * (gt + fp) - tp,
            "required_fp_removal_if_tp_unchanged": fp - (tp / threshold - gt),
            "absent_class_images": len(absent),
            "v3_fp_on_absent_images": absent_fp,
            "target_fp_on_absent_images": target_absent_fp,
            "v3_fp_on_positive_images": fp - absent_fp,
            "target_fp_on_positive_images": target_fp - target_absent_fp,
            "v3_fp_absent_share": absent_fp / fp,
        }

    losses, clipping = [], []
    positives, zero_positive = [0] * 4, [0] * 4
    direct_norms = [[[] for _ in range(4)] for _ in range(4)]
    sequence_ok = True
    finite = True
    digest = hashlib.sha256()
    with log_path.open("rb") as stream:
        for line_number, raw in enumerate(stream, 1):
            digest.update(raw)
            row = json.loads(raw)
            sequence_ok &= row["update"] == row["executed_updates"] == line_number
            metric = row["metrics"]
            losses.append(metric["loss"])
            clipping.append(metric["clip_coefficient"])
            finite &= all(math.isfinite(metric[k]) for k in ("loss", "main_loss", "aux_loss", "grad_norm_before_clip", "clip_coefficient"))
            for c in range(4):
                positive = metric["supervision_positive_pixels"][c]
                positives[c] += positive
                zero_positive[c] += positive == 0
                for scale in range(4):
                    direct_norms[scale][c].append(metric["gradients_before_clip"]["direct_scale_class"][scale][c])
    assert len(losses) == 40000 and sequence_ok and finite
    assert digest.hexdigest() == manifest["projected_jsonl_sha256"]
    log_summary = {
        "rows": len(losses), "continuous_executed_updates_1_to_40000": sequence_ok,
        "finite_logged_loss_and_global_gradient_values": finite,
        "projected_sha256": digest.hexdigest(),
        "first_2000_mean_loss": statistics.mean(losses[:2000]),
        "last_2000_mean_loss": statistics.mean(losses[-2000:]),
        "clipping_trigger_rate": sum(x < 1 for x in clipping) / len(clipping),
        "clip_coefficient_mean": statistics.mean(clipping),
        "clip_coefficient_median": statistics.median(clipping),
        "mean_positive_pixels_by_class": {c: p / len(losses) for c, p in zip(CLASSES, positives)},
        "zero_positive_update_rate_by_class": {c: n / len(losses) for c, n in zip(CLASSES, zero_positive)},
        "direct_weight_gradient_median_scale_by_class": [[statistics.median(v) for v in row] for row in direct_norms],
        "scope": "all 40000 local projected rows; omitted per-parameter maps not reviewed; remote raw not reread in this audit",
    }
    eps = 1e-5
    def dice(p, y):
        return 1 - (2 * sum(a * b for a, b in zip(p, y)) + eps) / (sum(a * a for a in p) + sum(b * b for b in y) + eps)
    n, p = 2048 ** 2, .5
    empty_grad = 2 * eps * p * p * (1 - p) / (n * p * p + eps) ** 2
    arithmetic = {
        "dice_lower_iou_worse_counterexample": {
            "labels": [1, 0], "before_p": [.51, .49], "after_p": [.49, .01],
            "before_loss": dice([.51, .49], [1, 0]), "after_loss": dice([.49, .01], [1, 0]),
            "before_iou": 1, "after_iou": 0,
        },
        "empty_class_constant_p_half_2048_all_valid": {
            "per_logit_derivative_before_image_class_averaging": empty_grad,
            "shared_bias_derivative_before_image_class_averaging": n * empty_grad,
            "scope": "illustrative algebra, not measured logits or training gradient",
        },
        "empty_aux_crop_construction": {
            "canvas_hw": [2048, 2048], "valid_support_yx": [0, 768, 0, 1024],
            "permitted_crop_yx": [1024, 1792, 1024, 1792],
            "crop_size": 768, "crop_is_all_ignore": True,
            "scope": "admissible padding/crop geometry, not a measured frequency",
        },
    }
    baseline_nodes = []
    def scan(node):
        if isinstance(node, dict):
            if node.get("completed_training_updates") == 40000 and "miou_pct" in node:
                baseline_nodes.append(node["miou_pct"])
            for value in node.values():
                scan(value)
        elif isinstance(node, list):
            for value in node:
                scan(value)
    scan(baseline)
    absent_only_fixed_iou = [100 * current_sum[c][0] / (
        current_sum[c][0] + current_sum[c][2] + per_class[name]["v3_fp_on_positive_images"])
        for c, name in enumerate(CLASSES)]
    report = {
        "reviewer": "root independent source, log and arithmetic review", "date": "2026-10-17",
        "no_model_execution_no_training_no_new_pixel_evaluation": True,
        "matched_validation_ids": 76, "per_image_gt_totals_match": True,
        "v3_miou_percent": statistics.mean(current_iou),
        "target_miou_percent": statistics.mean(target_iou),
        "gap_pp": statistics.mean(target_iou) - statistics.mean(current_iou),
        "baseline_reported_40k_miou_percent": sorted(set(baseline_nodes)),
        "remove_all_fp_on_absent_class_images_only": {
            "iou_percent": dict(zip(CLASSES, absent_only_fixed_iou)),
            "miou_percent": statistics.mean(absent_only_fixed_iou),
            "scope": "conditional algebraic upper bound when predictions on all class-positive images stay unchanged; no model prediction edited",
        },
        "per_class": per_class, "full_log_scan": log_summary, "algebra": arithmetic,
        "saved_input_hashes": {str(p): sha256(p) for p in (final_path, fusion_path, baseline_path, manifest_path)},
    }
    output = REVIEW / "root_independent_audit.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(output), "gap_pp": report["gap_pp"], "full_log_scan": log_summary,
                      "absent_image_fp": {c: {k: v for k, v in row.items() if "absent" in k or "positive_images" in k} for c, row in per_class.items()},
                      "baseline_40k": report["baseline_reported_40k_miou_percent"],
                      "absent_only_fp_fix_bound": report["remove_all_fp_on_absent_class_images_only"],
                      "algebra": arithmetic}, ensure_ascii=False))


if __name__ == "__main__":
    main()
