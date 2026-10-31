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

