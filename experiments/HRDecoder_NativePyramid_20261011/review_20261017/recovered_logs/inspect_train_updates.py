#!/usr/bin/env python3
import json
import sys

path = sys.argv[1]
targets = {int(x) for x in sys.argv[2:]}
found = {}
with open(path, "rt", encoding="utf-8") as stream:
    for line in stream:
        row = json.loads(line)
        update = int(row.get("update", -1))
        if update not in targets:
            continue
        m = row["metrics"]
        before = m["gradients_before_clip"]
        after = m["gradients_after_clip"]
        found[update] = {
            "update": update,
            "lr": row.get("lr"),
            "loss": m.get("loss"),
            "main_loss": m.get("main_loss"),
            "aux_loss": m.get("aux_loss"),
            "positive_bce": m.get("positive_bce"),
            "negative_bce": m.get("negative_bce"),
            "squared_dice": m.get("squared_dice"),
            "dense_bce": m.get("dense_bce"),
            "grad_norm_before_clip": m.get("grad_norm_before_clip"),
            "clip_coefficient": m.get("clip_coefficient"),
            "positive_pixels": m.get("supervision_positive_pixels"),
            "positive_images": m.get("supervision_positive_images"),
            "module_grad_norms_before": before.get("modules"),
            "module_grad_norms_after": after.get("modules"),
            "direct_scale_class_before": before.get("direct_scale_class"),
            "direct_scale_class_after": after.get("direct_scale_class"),
        }
        if len(found) == len(targets):
            break
print(json.dumps([found[k] for k in sorted(found)], indent=2))
