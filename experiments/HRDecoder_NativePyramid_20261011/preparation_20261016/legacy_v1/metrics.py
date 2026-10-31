"""Cumulative counts, fixed threshold and explicit collapse predicates."""
import numpy as np


def accumulate(counts, prediction, label):
    if prediction.shape != (4, *label.shape):
        raise ValueError("Predictions must be aligned to native labels")
    valid = label != 255
    for c in range(4):
        gt = (label == c + 1) & valid
        pred = prediction[c].astype(bool) & valid
        counts[c] += [np.count_nonzero(pred & gt), np.count_nonzero(pred & ~gt),
                      np.count_nonzero(~pred & gt)]


def summarize(counts, images):
    counts = np.asarray(counts, dtype=np.int64)
    if counts.shape != (4, 3) or (counts < 0).any():
        raise ValueError("Expected four TP/FP/FN rows")
    tp, fp, fn = counts.T.astype(np.float64)
    def divide(a, b):
        return np.divide(a, b, out=np.zeros(4), where=b != 0)
    iou = divide(tp, tp + fp + fn)
    return {"mIoU_percent": float(iou.mean() * 100), "IoU_percent": (iou * 100).tolist(),
            "precision_percent": (divide(tp, tp + fp) * 100).tolist(),
            "recall_percent": (divide(tp, tp + fn) * 100).tolist(),
            "counts": counts.tolist(), "images": images, "classes": ["EX", "HE", "SE", "MA"],
            "threshold": "strictly > 0.5", "undefined_precision_classes": np.where(tp + fp == 0)[0].tolist()}


def validation_collapse(current, previous, protocol):
    rule = protocol["validation_collapse"]
    if not previous or current["update"] < rule["minimum_update"]:
        return []
    best = max(previous, key=lambda x: x["mIoU_percent"])
    reasons = []
    if best["mIoU_percent"] - current["mIoU_percent"] >= rule["miou_drop_pp"]:
        reasons.append("large_validation_miou_drop")
    for c in range(4):
        established = any(v["IoU_percent"][c] >= rule["class_iou_previous_min"] for v in previous)
        if established and current["recall_percent"][c] / 100 <= rule["class_recall_max"]:
            reasons.append("lost_class_" + protocol["classes"][c])
    return reasons


class TrainProbeGuard:
    def __init__(self, protocol):
        self.rule = protocol["train_probe_collapse"]
        self.best_recall = [0.] * 4
        self.streak = [0] * 4

    def observe(self, result, update):
        reasons = []
        for c, r_percent in enumerate(result["recall_percent"]):
            r = r_percent / 100
            median = result["positive_logit_median"][c]
            if median is None:
                raise ValueError("Fixed training probe must contain every class")
            bad = (update >= self.rule["minimum_update"] and
                   ((self.best_recall[c] >= self.rule["previous_recall_min"] and
                     r < self.rule["recall_fraction"] * self.best_recall[c]) or
                    (update >= self.rule["initial_stall_minimum_update"] and
                     r < self.rule["initial_stall_recall_max"])) and
                   median < self.rule["positive_logit_median_max"])
            self.streak[c] = self.streak[c] + 1 if bad else 0
            self.best_recall[c] = max(self.best_recall[c], r)
            if self.streak[c] >= self.rule["consecutive"]:
                reasons.append("saturated_training_class_" + str(c))
        return reasons

    def state_dict(self):
        return {"best_recall": self.best_recall, "streak": self.streak}

    def load_state_dict(self, state):
        self.best_recall = list(state["best_recall"]); self.streak = list(state["streak"])

