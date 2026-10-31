# -*- coding: utf-8 -*-
"""trainer.loss — 四类独立 logits 的 masked BCE + masked soft Dice（数学定义见
docs/TRAINING_OBJECTIVE.md，本文件实现与文档逐条对应）。

关键语义（与旧 BinaryLoss 的差异，旧实现不进入新模型）：
1. 每类独立二分类 logits（4 通道，无背景通道），阈值前只做一次 sigmoid
   （BCE 用 BCEWithLogits 内部实现，Dice 复用同一次 sigmoid 的概率）。
2. valid=0 的像素 *整类屏蔽*：既不当正样本也不当负样本（缺类绝不当阴性）。
3. 先按每类有效像素数归一（per-class normalization），再对有效类别等权平均。
4. 全无效类别不产生梯度且不引入 NaN：该类从损失与分母中同时剔除；
   整 batch 全无效时返回 0·Σlogits（保持计算图、梯度为 0）。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

NUM_LESION_CLASSES = 4  # EX/HE/SE/MA，通道序由 tools/contract_common.py 权威定义


def masked_bce_dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    smooth: float = 1.0e-5,
    ce_weight: float = 1.0,
    dice_weight: float = 1.0,
) -> tuple:
    """masked BCE + masked soft Dice。

    Args:
        logits: [B, 4, H, W] float，未过 sigmoid 的原始 logits。
        target: [B, 4, H, W] float，0/1（可微逼近的硬标签）。
        valid:  [B, 4, H, W] float，1=该像素该类参与监督；0=屏蔽（未标注）。
        smooth: Dice 平滑项 ε（沿用旧已验证定义 1e-5）。

    Returns:
        (loss, parts)：loss 为标量张量；parts 为逐分量标量（float，日志用），
        全无效类别对应的 per-class 值记 NaN（仅日志语义，不参与反向）。
    """
    if logits.dim() != 4 or logits.shape[1] != NUM_LESION_CLASSES:
        raise ValueError(
            f"logits 必须为 [B,{NUM_LESION_CLASSES},H,W]，当前 {tuple(logits.shape)}"
        )
    for name, t in (("target", target), ("valid", valid)):
        if t.shape != logits.shape:
            raise ValueError(f"{name} 形状 {tuple(t.shape)} 与 logits {tuple(logits.shape)} 不一致")
    # 统一到 float32 计算域（valid/target 可能来自 uint8/bool 的 numpy 转换）
    target = target.to(logits.dtype)
    valid = valid.to(logits.dtype)
    if bool((valid < 0).any()) or bool((valid > 1).any()):
        raise ValueError("valid 掩码必须是 0/1（含其他值会破坏 per-class 归一语义）")

    # ---- 单次 sigmoid（Dice 用）；BCE 走 BCEWithLogits 数值稳定路径 ----
    prob = torch.sigmoid(logits)

    # ---- 每类有效像素数 n_c = Σ valid_c ----
    n_valid = valid.sum(dim=(0, 2, 3))  # [4]
    class_has_valid = (n_valid > 0).to(logits.dtype)  # [4]
    n_active = class_has_valid.sum()

    # ---- masked BCE（per-class 归一）----
    # BCEWithLogits 逐像素；valid=0 位置乘 0 剔除，且分母只除该类的有效像素数。
    # 全无效类：分子/分母同时为 0，再乘 class_has_valid=0 ⇒ 贡献恒为 0，
    # 反传梯度也恒为 0（不会把"未标注"当成阴性）。
    bce_map = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    bce_per_class = (bce_map * valid).sum(dim=(0, 2, 3)) / torch.clamp(n_valid, min=1.0)
    bce_per_class = bce_per_class * class_has_valid

    # ---- masked soft Dice（per-class 归一，valid 只作用一次）----
    p = prob * valid
    t = target * valid
    dims = (0, 2, 3)
    inter = (p * t).sum(dim=dims)
    denom = p.sum(dim=dims) + t.sum(dim=dims)
    dice_per_class = 1.0 - (2.0 * inter + smooth) / (denom + smooth)
    dice_per_class = dice_per_class * class_has_valid

    # ---- 有效类别等权平均（分母 = 有效类别数，不是 4）----
    safe_n = torch.clamp(n_active, min=1.0)
    loss = (ce_weight * bce_per_class.sum() + dice_weight * dice_per_class.sum()) / safe_n

    if float(n_active.item()) == 0.0:
        # 整个 micro-batch 无任何有效类别：显式 0 梯度（保持计算图、绝不静默丢弃）
        loss = (logits * 0.0).sum()

    # 日志分量：只在 **有效类别** 上取均值（全无效类的 0 不参与，避免稀释读数）
    if float(n_active.item()) > 0.0:
        denom_active = torch.clamp(n_active, min=1.0)
        parts = {
            "loss_bce": float((bce_per_class.sum() / denom_active).item()),
            "loss_dice": float((dice_per_class.sum() / denom_active).item()),
            "loss_total": float(loss.item()),
            "active_classes": int(n_active.item()),
            "valid_pixels_per_class": [int(v) for v in n_valid.tolist()],
        }
    else:
        parts = {
            "loss_bce": float("nan"), "loss_dice": float("nan"),
            "loss_total": float(loss.item()), "active_classes": 0,
            "valid_pixels_per_class": [0, 0, 0, 0],
        }
    return loss, parts


class MaskedBinaryCEDice(torch.nn.Module):
    """可调用包装（StageRunner 使用）；参数与配置 loss 节对齐。"""

    def __init__(self, smooth: float = 1.0e-5, ce_weight: float = 1.0, dice_weight: float = 1.0):
        super().__init__()
        self.smooth = float(smooth)
        self.ce_weight = float(ce_weight)
        self.dice_weight = float(dice_weight)

    def forward(self, logits, target, valid):
        return masked_bce_dice_loss(
            logits, target, valid,
            smooth=self.smooth, ce_weight=self.ce_weight, dice_weight=self.dice_weight,
        )
