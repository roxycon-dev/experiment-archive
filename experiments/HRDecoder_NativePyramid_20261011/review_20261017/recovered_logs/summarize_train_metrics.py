#!/usr/bin/env python3
"""Stream a projected train.jsonl(/.gz) into fixed 2k-update summaries."""
import gzip
import heapq
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

CLASSES = ("EX", "HE", "SE", "MA")
SCALES = ("F1", "F2", "F3", "F4")
MODULES = (
    "backbone.net.bn1", "backbone.net.bn2", "backbone.net.conv1", "backbone.net.conv2",
    "backbone.net.layer1", "backbone.net.stage2", "backbone.net.stage3", "backbone.net.stage4",
    "backbone.net.transition1", "backbone.net.transition2", "backbone.net.transition3",
    "decoder.auxiliary", "decoder.deep", "decoder.detail", "decoder.direct_bias",
    "decoder.direct_classifiers", "decoder.fusions", "decoder.laterals", "decoder.readout", "decoder.stem",
)
LOSSES = ("positive_bce", "negative_bce", "squared_dice", "dense_bce", "main_loss", "aux_loss", "loss")


def open_text(path):
    if str(path).lower().endswith((".gz", ".gz.partial")):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "rt", encoding="utf-8")


def quantile(values, q):
    if not values:
        return None
    x = sorted(values)
    pos = (len(x) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(x) - 1)
    return x[lo] + (x[hi] - x[lo]) * (pos - lo)


def add_num(dst, key, value):
    if value is None:
        return
    if isinstance(value, bool):
        return
    try:
        v = float(value)
    except (TypeError, ValueError):
        return
    if math.isfinite(v):
        dst[key].append(v)


def add_gradient_row(dst, row, suffix):
    metrics = row.get("metrics", {})
    grad = metrics.get("gradients_" + suffix, {})
    modules = grad.get("modules", {})
    for name in MODULES:
        add_num(dst, f"{suffix}.module.{name}", modules.get(name))
    # Disjoint named module groups make root-sum-square an exact aggregate norm.
    backbone_names = [n for n in MODULES if n.startswith("backbone.")]
    decoder_names = [n for n in MODULES if n.startswith("decoder.")]
    for label, names in (("backbone_all", backbone_names), ("decoder_all", decoder_names)):
        vals = [modules.get(n, 0.0) for n in names]
        if vals and all(isinstance(v, (int, float)) for v in vals):
            add_num(dst, f"{suffix}.aggregate.{label}", math.sqrt(sum(float(v) ** 2 for v in vals)))
    # C and D endpoints and upstream groups; semantic trunk is shared by C and D.
    for label, names in (
        ("C_endpoint_auxiliary", ["decoder.auxiliary"]),
        ("D_endpoint_readout", ["decoder.readout"]),
        ("D_stem_detail", ["decoder.stem", "decoder.detail"]),
        ("shared_semantic_trunk", ["decoder.deep", "decoder.laterals", "decoder.fusions"]),
    ):
        vals = [modules.get(n, 0.0) for n in names]
        if vals and all(isinstance(v, (int, float)) for v in vals):
            add_num(dst, f"{suffix}.aggregate.{label}", math.sqrt(sum(float(v) ** 2 for v in vals)))
    matrix = grad.get("direct_scale_class")
    if matrix is not None:
        for si, scale in enumerate(SCALES):
            if si >= len(matrix) or matrix[si] is None:
                continue
            for ci, cls in enumerate(CLASSES):
                if ci < len(matrix[si]):
                    add_num(dst, f"{suffix}.direct.{scale}.{cls}", matrix[si][ci])
    # Check that global clipping scales every nonzero module/direct norm uniformly.
    before = metrics.get("gradients_before_clip", {}).get("modules", {})
    after = metrics.get("gradients_after_clip", {}).get("modules", {})
    coeff = metrics.get("clip_coefficient")
    if isinstance(coeff, (int, float)):
        for name in MODULES:
            b, a = before.get(name), after.get(name)
            if isinstance(b, (int, float)) and isinstance(a, (int, float)) and b > 0:
                add_num(dst, f"clip_module_ratio_error.{name}", abs(a / b - coeff))


def summarize(values):
    out = {}
    for k, vals in values.items():
        if vals:
            out[k] = {
                "mean": sum(vals) / len(vals),
                "p10": quantile(vals, .10),
                "p50": quantile(vals, .50),
                "p90": quantile(vals, .90),
                "min": min(vals),
                "max": max(vals),
                "n": len(vals),
            }
    return out


def main():
    if len(sys.argv) < 2:
        raise SystemExit("usage: summarize_train_metrics.py INPUT.jsonl[.gz] [OUTPUT_PREFIX]")
    source = Path(sys.argv[1])
    prefix = Path(sys.argv[2]) if len(sys.argv) > 2 else source.with_suffix("")
    bins = defaultdict(lambda: defaultdict(list))
    overall = defaultdict(list)
    first, last, count, bad_lines = None, None, 0, 0
    gaps, duplicates = [], []
    seen = set()
    relation_error = []
    top_norm_rows = []
    min_clip_row = None
    with open_text(source) as stream:
        for line_no, line in enumerate(stream, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                bad_lines += 1
                continue
            update = int(row.get("update", row.get("executed_updates", -1)))
            if update < 1:
                bad_lines += 1
                continue
            if first is None:
                first = update
            if last is not None:
                if update == last:
                    duplicates.append(update)
                elif update != last + 1:
                    gaps.append([last, update])
            if update in seen and update != last:
                duplicates.append(update)
            seen.add(update)
            last = update
            count += 1
            b = ((update - 1) // 2000) * 2000
            dst = bins[b]
            m = row.get("metrics", {})
            for key in LOSSES:
                for target in (dst, overall):
                    add_num(target, key, m.get(key))
            for target in (dst, overall):
                add_num(target, "lr", row.get("lr"))
                add_num(target, "grad_norm_before_clip", m.get("grad_norm_before_clip"))
                add_num(target, "clip_coefficient", m.get("clip_coefficient"))
            lr = row.get("lr")
            clip = m.get("clip_coefficient")
            if isinstance(lr, (int, float)) and isinstance(clip, (int, float)):
                for target in (dst, overall):
                    add_num(target, "lr_times_clip", lr * clip)
            if isinstance(clip, (int, float)):
                for target in (dst, overall):
                    add_num(target, "clip_active", int(clip < 1.0 - 1e-12))
                    add_num(target, "clip_strong", int(clip < .5))
                if min_clip_row is None or clip < min_clip_row["clip_coefficient"]:
                    min_clip_row = {"update": update, "clip_coefficient": clip,
                                    "grad_norm_before_clip": m.get("grad_norm_before_clip"),
                                    "lr": lr, "loss": m.get("loss")}
            norm = m.get("grad_norm_before_clip")
            if isinstance(norm, (int, float)):
                item = (float(norm), update, {"update": update, "grad_norm_before_clip": norm,
                                              "clip_coefficient": clip, "lr": lr,
                                              "loss": m.get("loss")})
                if len(top_norm_rows) < 10:
                    heapq.heappush(top_norm_rows, item)
                elif norm > top_norm_rows[0][0]:
                    heapq.heapreplace(top_norm_rows, item)
            if all(isinstance(m.get(k), (int, float)) for k in ("main_loss", "aux_loss", "loss")):
                relation_error.append(abs(m["loss"] - (m["main_loss"] + .2 * m["aux_loss"])))
            for field, labels in (("supervision_positive_pixels", CLASSES), ("supervision_positive_images", CLASSES)):
                arr = m.get(field, [])
                for i, cls in enumerate(labels):
                    if i < len(arr):
                        for target in (dst, overall):
                            add_num(target, f"{field}.{cls}", arr[i])
                            add_num(target, f"{field}.{cls}.zero", int(arr[i] == 0))
            add_gradient_row(dst, row, "before_clip")
            add_gradient_row(dst, row, "after_clip")
            add_gradient_row(overall, row, "before_clip")
            add_gradient_row(overall, row, "after_clip")

    records = []
    for b in sorted(bins):
        low, high = b + 1, min(b + 2000, last or b + 2000)
        records.append({"updates": f"{low}-{high}", "n": len(next(iter(bins[b].values()))) if bins[b] else 0,
                        **summarize(bins[b])})
    payload = {"source": str(source), "first_update": first, "last_update": last,
               "rows": count, "bad_or_incomplete_lines": bad_lines, "gaps": gaps[:50],
               "gap_count": len(gaps), "duplicate_updates": duplicates[:50],
               "duplicate_count": len(duplicates), "loss_relation_abs_error_max": max(relation_error, default=None),
               "overall": summarize(overall), "bins_2k": records,
               "top_global_grad_norm_rows": [item[2] for item in sorted(top_norm_rows, reverse=True)],
               "minimum_clip_coefficient_row": min_clip_row}
    out_json = prefix.with_name(prefix.name + "_2k_summary.json")
    out_csv = prefix.with_name(prefix.name + "_2k_summary.csv")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    # Long-form CSV keeps the schema stable as the selected module metrics evolve.
    with out_csv.open("w", encoding="utf-8-sig", newline="") as f:
        import csv
        writer = csv.writer(f)
        writer.writerow(["updates", "metric", "mean", "p10", "p50", "p90", "max", "n"])
        for record in records:
            label = record["updates"]
            for key, stat in record.items():
                if isinstance(stat, dict) and "mean" in stat:
                    writer.writerow([label, key, stat["mean"], stat["p10"], stat["p50"], stat["p90"], stat["max"], stat["n"]])
    compact_keys = [*LOSSES, "lr", "lr_times_clip", "grad_norm_before_clip", "clip_coefficient", "clip_active", "clip_strong"]
    compact_keys += [f"{phase}.aggregate.{name}" for phase in ("before_clip", "after_clip")
                     for name in ("backbone_all", "decoder_all", "C_endpoint_auxiliary", "D_endpoint_readout",
                                  "D_stem_detail", "shared_semantic_trunk")]
    compact_keys += [f"{phase}.module.{name}" for phase in ("before_clip", "after_clip")
                     for name in ("backbone.net.stage2", "backbone.net.stage3", "backbone.net.stage4",
                                  "decoder.auxiliary", "decoder.direct_classifiers", "decoder.readout",
                                  "decoder.detail")]
    compact_keys += [f"before_clip.direct.{scale}.MA" for scale in SCALES]
    compact_bins = []
    for record in records:
        compact = {"updates": record["updates"], "n": record["n"]}
        for key in compact_keys:
            if key in record:
                stat = record[key]
                compact[key] = {part: stat[part] for part in ("mean", "p50", "p90")}
        compact_bins.append(compact)
    overall_compact = {k: {part: v[part] for part in ("mean", "p10", "p50", "p90", "max", "n")}
                       for k, v in payload["overall"].items()
                       if k in set(compact_keys) | {"supervision_positive_pixels.MA", "supervision_positive_pixels.MA.zero"}}
    print(json.dumps({k: payload[k] for k in ("source", "first_update", "last_update", "rows", "bad_or_incomplete_lines", "gap_count", "duplicate_count", "loss_relation_abs_error_max", "top_global_grad_norm_rows", "minimum_clip_coefficient_row")}
                     | {"overall_compact": overall_compact, "bins_2k_compact": compact_bins}, ensure_ascii=False))
    print(f"json={out_json}")
    print(f"csv={out_csv}")


if __name__ == "__main__":
    main()
