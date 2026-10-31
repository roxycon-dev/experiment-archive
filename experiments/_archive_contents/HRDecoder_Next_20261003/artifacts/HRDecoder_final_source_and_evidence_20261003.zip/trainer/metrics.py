# -*- coding: utf-8 -*-
"""trainer.metrics — 整图累计 TP/FP/FN 指标（口径对齐旧 lesion_metric 的累计语义）。

协议（configs metrics.protocol = whole_image_accumulated_counts）：
- 每类独立：TP=Σ[p>τ ∧ y=1 ∧ v=1]，FP=Σ[p>τ ∧ y=0 ∧ v=1]，FN=Σ[p≤τ ∧ y=1 ∧ v=1]。
  跨整张验证集图片逐像素累计（不是逐图平均），v=0 的像素（未标注）不计入任何桶。
- τ=0.5；概率由 logits 的单次 sigmoid 得到（本模块只接收概率）。
- IoU_c = TP/(TP+FP+FN)；F1_c = 2TP/(2TP+FP+FN)；PPV_c = TP/(TP+FP)；Recall_c = TP/(TP+FN)。
- 主指标 mIoU 只对四类（EX/HE/SE/MA）平均，背景不参与；分母为 0 的类记 NaN
  并从均值中剔除（显式记录 excluded，不冒充 0）。
- AUPR：每类 256-bin 概率直方图（正/负像素分开累计），阈值扫描 + 阶梯插值；
  是对连续 PR 曲线的分箱近似，文档与输出中如实标注。

本模块只做计数与汇总；整图前向由 `trainer/eval.py:WholeImageEvaluator` 负责
（禁止 crop 指标冒充整图指标，禁止静默 resize —— 见该模块的 fail-closed 断言）。
落盘路径统一经 trainer.paths 校验（拒绝 `..` 路径穿越），临时文件用
tempfile 在同目录生成后原子替换，不做任何字符串路径拼接。
"""

from __future__ import annotations

import csv
import json
import os
import tempfile
from datetime import datetime, timezone

import numpy as np

from .paths import ensure_parent_dir

LESION_CLASSES = ("EX", "HE", "SE", "MA")
DEFAULT_THRESHOLD = 0.5
AUPR_BINS = 256


class SegMetricsAccumulator:
    """跨样本累计每类 TP/FP/FN 与 AUPR 直方图。"""

    def __init__(self, threshold: float = DEFAULT_THRESHOLD, aupr_bins: int = AUPR_BINS):
        self.threshold = float(threshold)
        self.aupr_bins = int(aupr_bins)
        self.reset()

    def reset(self) -> None:
        self.tp = np.zeros(4, dtype=np.int64)
        self.fp = np.zeros(4, dtype=np.int64)
        self.fn = np.zeros(4, dtype=np.int64)
        # AUPR 直方图：[4, bins]（正样本像素 / 负样本有效像素分开计）
        self.hist_pos = np.zeros((4, self.aupr_bins), dtype=np.int64)
        self.hist_neg = np.zeros((4, self.aupr_bins), dtype=np.int64)
        self.n_images = 0

    def update(self, probs, target, valid=None) -> None:
        """累计一张（或一批）整图。

        Args:
            probs:  [B,4,H,W] numpy/tensor，sigmoid 后概率（只 sigmoid 一次的产物）。
            target: [B,4,H,W] 同型 0/1。
            valid:  [B,4,H,W] 同型 0/1，v=0 像素完全不计入；None 表示全图有效
                    （等价于全 1，仅用于"整图已标注"场景，不改变语义）。
        """
        import torch  # 延迟导入，保持 numpy-only 场景可单测

        to_np = lambda t: (t.detach().float().cpu().numpy() if torch.is_tensor(t) else np.asarray(t))
        probs = to_np(probs)
        target = to_np(target)
        if probs.ndim == 3:
            probs = probs[None]
            target = target[None]
            if valid is not None:
                valid = valid[None]
        if valid is None:
            valid = np.ones_like(probs)
        else:
            valid = to_np(valid)
        if probs.shape != target.shape or probs.shape != valid.shape:
            raise ValueError(
                f"metrics.update 形状不一致: probs={probs.shape} target={target.shape} valid={valid.shape}"
            )
        if probs.shape[1] != 4:
            raise ValueError(f"metrics.update 期望 4 通道（EX/HE/SE/MA），得到 {probs.shape[1]}")
        b = probs.shape[0]
        # 阈值语义与旧 lesion_metric 一致：严格大于 τ 才算阳性（p > 0.5）。
        v = valid > 0.5
        pos = (target > 0.5) & v
        neg = (target <= 0.5) & v
        pred = (probs > self.threshold) & v          # 未标注像素永不计入 FP
        for c in range(4):
            self.tp[c] += int(np.logical_and(pred[:, c], pos[:, c]).sum())
            self.fp[c] += int(np.logical_and(pred[:, c], neg[:, c]).sum())
            self.fn[c] += int(np.logical_and(~pred[:, c], pos[:, c]).sum())
            pb = np.clip((probs[:, c] * self.aupr_bins).astype(np.int64), 0, self.aupr_bins - 1)
            self.hist_pos[c] += np.bincount(pb[pos[:, c]], minlength=self.aupr_bins)
            self.hist_neg[c] += np.bincount(pb[neg[:, c]], minlength=self.aupr_bins)
        self.n_images += b

    # ---- 汇总 ----

    def compute(self) -> dict:
        per_class = {}
        ious, f1s, pps, rcs, aps = [], [], [], [], []
        excluded = {"iou": [], "aupr": []}
        for c, name in enumerate(LESION_CLASSES):
            tp, fp, fn = int(self.tp[c]), int(self.fp[c]), int(self.fn[c])
            denom_iou = tp + fp + fn
            iou = tp / denom_iou if denom_iou > 0 else float("nan")
            denom_f1 = 2 * tp + fp + fn
            f1 = 2 * tp / denom_f1 if denom_f1 > 0 else float("nan")
            ppv = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
            rec = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
            ap = self._aupr(c)
            entry = {
                "iou": iou, "f1": f1, "ppv": ppv, "recall": rec, "aupr": ap,
                "tp": tp, "fp": fp, "fn": fn,
                "gt_positive_pixels": tp + fn,
            }
            per_class[name] = entry
            if np.isfinite(iou):
                ious.append(iou)
            else:
                excluded["iou"].append(name)
            if np.isfinite(f1):
                f1s.append(f1)
            if np.isfinite(ppv):
                pps.append(ppv)
            if np.isfinite(rec):
                rcs.append(rec)
            if np.isfinite(ap):
                aps.append(ap)
            else:
                excluded["aupr"].append(name)
        return {
            "protocol": "whole_image_accumulated_counts",
            "threshold": self.threshold,
            "n_images": self.n_images,
            "per_class": per_class,
            "mIoU_lesion4": float(np.mean(ious)) if ious else float("nan"),
            "mF1": float(np.mean(f1s)) if f1s else float("nan"),
            "mPPV": float(np.mean(pps)) if pps else float("nan"),
            "mRecall": float(np.mean(rcs)) if rcs else float("nan"),
            "mAUPR": float(np.mean(aps)) if aps else float("nan"),
            "excluded_classes": excluded,
            "aupr_note": f"{self.aupr_bins}-bin histogram approximation of PR curve",
        }

    def _aupr(self, c: int) -> float:
        """阈值自高到低扫描的阶梯 AP；无正样本返回 NaN。

        正确的阶梯插值（right-Riemann）AP = Σ_i (R_i − R_{i−1}) · P_i，
        必须在 **recall 递增** 方向累加。旧实现在 recall 递减方向累加，
        完美分类器只能得到约 0.2（真值为 1.0）——已修正，见
        tests/run_training_smoke.py 的手算一致性用例。
        """
        total_pos = int(self.hist_pos[c].sum())
        if total_pos == 0:
            return float("nan")
        # bin 索引越大 = 概率越高；反转后 cumsum 即"阈值由高到低"的累计，
        # 对应 recall 由低到高递增。
        tp_cum = np.cumsum(self.hist_pos[c][::-1]).astype(np.float64)  # [bins]
        fp_cum = np.cumsum(self.hist_neg[c][::-1]).astype(np.float64)
        precision = tp_cum / np.maximum(tp_cum + fp_cum, 1.0)
        recall = tp_cum / float(total_pos)
        d_recall = np.diff(np.concatenate([[0.0], recall]))  # 非负、递增方向
        return float(np.sum(d_recall * precision))


def _atomic_write_text(path: str, text: str) -> None:
    """同目录临时文件 + os.replace 原子写（路径经校验，临时文件由 tempfile 生成）。"""
    path = ensure_parent_dir(path)
    parent = os.path.dirname(path)
    fd, tmp = tempfile.mkstemp(prefix=".tmp_metrics_", dir=parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        os.replace(tmp, path)
    except Exception:
        if os.path.isfile(tmp):
            os.remove(tmp)
        raise


# ---------------------------------------------------------------------------
# 持久化（JSON + CSV 历史追加）
# ---------------------------------------------------------------------------

CSV_FIELDS = [
    "timestamp", "split", "update", "stage", "mode",
    "mIoU_lesion4", "mF1", "mPPV", "mRecall", "mAUPR", "n_images",
]


def save_metrics_json(report: dict, out_path: str) -> str:
    payload = dict(report)
    payload["generated_at"] = datetime.now(timezone.utc).isoformat()
    _atomic_write_text(out_path, json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True) + "\n")
    return out_path


def append_metrics_csv(report: dict, csv_path: str) -> str:
    """把一次评估摘要追加进 eval_history.csv（幂等表头）。"""
    csv_path = ensure_parent_dir(csv_path)
    new_file = not os.path.isfile(csv_path)
    with open(csv_path, "a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if new_file:
            w.writeheader()
        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "split": report.get("split"),
            "update": report.get("update"),
            "stage": report.get("stage"),
            "mode": report.get("mode"),
            "mIoU_lesion4": report.get("mIoU_lesion4"),
            "mF1": report.get("mF1"),
            "mPPV": report.get("mPPV"),
            "mRecall": report.get("mRecall"),
            "mAUPR": report.get("mAUPR"),
            "n_images": report.get("n_images"),
        }
        w.writerow(row)
    return csv_path
