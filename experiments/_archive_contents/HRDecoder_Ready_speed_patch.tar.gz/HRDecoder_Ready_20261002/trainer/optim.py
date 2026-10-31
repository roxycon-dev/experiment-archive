# -*- coding: utf-8 -*-
"""trainer.optim — 参数组、按 optimizer update 计数的调度器、梯度裁剪。

规则（与 configs 对齐）：
- 参数组区分 backbone / decoder 两组 LR（H：HRNet 主干 vs FCN 头+上采样；
  U：ConvNeXt 编码器 vs U-Net 解码器）。
- HRNet BN 统计冻结但 affine 可训练（norm_eval=True + requires_grad 保留）；
  ConvNeXt LayerNorm 正常训练（不做任何统计冻结）。
- 梯度累积以 optimizer update 计数：每 grad_accum 个 micro-step 才
  optimizer.step() 一次；scheduler 也按 update 步进（warmup+cosine 手工实现，
  不依赖 PyTorch Lightning/mmengine 语义）。
"""

from __future__ import annotations

import math

import torch

from .gate import GateError


# ---------------------------------------------------------------------------
# 参数组
# ---------------------------------------------------------------------------

def classify_parameter(name: str, backbone_prefixes: tuple) -> str:
    """按参数名把参数归入 'backbone' 或 'decoder'。

    前缀按"点分段"匹配：`encoder` 只匹配 `encoder` 本身或 `encoder.xxx`，
    绝不匹配 `encoder2`（旧实现用裸 startswith，会把同前缀兄弟模块误并组）。
    """
    for prefix in backbone_prefixes:
        p = str(prefix).rstrip(".")
        if name == p or name.startswith(p + "."):
            return "backbone"
    return "decoder"


def build_param_groups(model, *, backbone_lr: float, decoder_lr: float,
                       weight_decay: float, backbone_prefixes: tuple) -> list:
    """构造 AdamW 参数组（两组，LR 不同；weight_decay 按配置统一施加）。"""
    if not backbone_prefixes:
        raise GateError("backbone_prefixes 为空：必须显式声明 backbone 参数名前缀")
    groups = {"backbone": [], "decoder": []}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        groups[classify_parameter(name, tuple(backbone_prefixes))].append(p)
    n_back = sum(p.numel() for p in groups["backbone"])
    n_dec = sum(p.numel() for p in groups["decoder"])
    if not groups["backbone"]:
        raise GateError("backbone 参数组为空：前缀配置与模型命名不匹配（fail closed）")
    if not groups["decoder"]:
        raise GateError("decoder 参数组为空：前缀配置与模型命名不匹配（fail closed）")
    return [
        {"params": groups["backbone"], "lr": float(backbone_lr),
         "weight_decay": float(weight_decay), "name": "backbone",
         "_n_params": n_back},
        {"params": groups["decoder"], "lr": float(decoder_lr),
         "weight_decay": float(weight_decay), "name": "decoder",
         "_n_params": n_dec},
    ]


def build_optimizer(stage_cfg: dict, param_groups: list) -> torch.optim.Optimizer:
    opt_type = str(stage_cfg.get("optimizer", "adamw")).lower()
    if opt_type != "adamw":
        raise GateError(f"仅实现 adamw（配置 optimizer={opt_type!r}）——其他优化器需另行验证")
    wd = float(stage_cfg.get("weight_decay", 1e-2))
    return torch.optim.AdamW(param_groups, lr=1.0, weight_decay=wd)


def audit_param_groups(model, optimizer) -> dict:
    """校验参数组"互斥且完整"：每个可训练参数恰好属于一个组，无遗漏无重复；
    冻结参数（requires_grad=False）不得出现在任何组里。

    返回 {"n_trainable_params":n, "n_frozen_params":n, "n_tensors":n, "groups":{...}}；
    违反时抛 GateError（fail closed，而不是训练到一半才发现有参数没被优化）。
    """
    trainable = {id(p): n for n, p in model.named_parameters() if p.requires_grad}
    frozen = [n for n, p in model.named_parameters() if not p.requires_grad]

    seen = {}
    dupes = []
    groups = {}
    for gi, g in enumerate(optimizer.param_groups):
        gname = g.get("name", f"group{gi}")
        groups[gname] = {"lr": float(g["lr"]), "n_tensors": 0, "n_params": 0}
        for p in g["params"]:
            if id(p) in seen:
                dupes.append((seen[id(p)], gname))
            seen[id(p)] = gname
            groups[gname]["n_tensors"] += 1
            groups[gname]["n_params"] += int(p.numel())
            if not p.requires_grad:
                raise GateError(f"参数组 {gname} 含 requires_grad=False 的参数（破坏了冻结语义）")
    missing = sorted(n for pid, n in trainable.items() if pid not in seen)
    if dupes:
        raise GateError(f"参数被多个组重复持有（互斥性破坏）: {dupes[:5]}")
    if missing:
        raise GateError(
            f"{len(missing)} 个可训练参数未被任何参数组覆盖（完整性破坏），例如 {missing[:5]}"
        )
    return {
        "n_trainable_tensors": len(trainable),
        "n_frozen_tensors": len(frozen),
        "frozen_examples": sorted(frozen)[:5],
        "groups": groups,
    }


# ---------------------------------------------------------------------------
# 调度：linear warmup + cosine（以 optimizer update 计数）
# ---------------------------------------------------------------------------

def lr_multiplier(update: int, *, total_updates: int, warmup_updates: int,
                  cosine_min_ratio: float = 0.01) -> float:
    """返回第 update 次 update（0-based）的全局 LR 乘子。"""
    if total_updates <= 0:
        raise GateError("total_updates 必须 > 0")
    warmup = max(int(warmup_updates), 0)
    if warmup > 0 and update < warmup:
        return (update + 1) / warmup
    progress = (update - warmup) / max(total_updates - warmup, 1)
    progress = min(max(progress, 0.0), 1.0)
    min_r = float(cosine_min_ratio)
    return min_r + (1.0 - min_r) * 0.5 * (1.0 + math.cos(math.pi * progress))


def apply_lr(optimizer: torch.optim.Optimizer, base_lrs: dict, mult: float) -> dict:
    """把 base_lr * mult 写入各参数组，返回实际 LR（日志用）。

    用 enumerate 取组序号——旧实现用 list.index(dict) 反查，遇到两个内容相同的
    参数组会返回第一个下标，导致 LR 写错组。
    """
    applied = {}
    for gi, group in enumerate(optimizer.param_groups):
        name = group.get("name", f"group{gi}")
        if name not in base_lrs:
            raise GateError(f"参数组 {name!r} 没有对应 base_lr（{sorted(base_lrs)}）")
        group["lr"] = float(base_lrs[name]) * float(mult)
        applied[name] = group["lr"]
    return applied


def base_lrs_of(param_groups: list) -> dict:
    """从参数组读出各组基准 LR（优化器构造后立即调用，锁定基准）。"""
    out = {}
    for gi, g in enumerate(param_groups):
        out[g.get("name", f"group{gi}")] = float(g["lr"])
    return out


def scheduler_state(update: int, base_lrs: dict, *, total_updates: int,
                    warmup_updates: int, cosine_min_ratio: float) -> dict:
    return {
        "type": "warmup_cosine_by_update",
        "update": int(update),
        "base_lrs": dict(base_lrs),
        "total_updates": int(total_updates),
        "warmup_updates": int(warmup_updates),
        "cosine_min_ratio": float(cosine_min_ratio),
    }


# ---------------------------------------------------------------------------
# 梯度裁剪
# ---------------------------------------------------------------------------

def clip_gradients(model, max_norm) -> float | None:
    """按配置裁剪梯度；max_norm 为 None/<=0 时不裁剪。返回裁剪前总范数。"""
    if max_norm is None or float(max_norm) <= 0:
        return None
    total_norm = torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad], float(max_norm)
    )
    return float(total_norm)


# ---------------------------------------------------------------------------
# BN / Norm 策略审计
# ---------------------------------------------------------------------------

def freeze_bn_stats(model, *, backbone_prefixes: tuple) -> int:
    """把 backbone 内的 BatchNorm 统计量冻结（模块转 eval），affine 仍可训练。

    必须在每次 `model.train()` **之后** 调用（train() 会把所有子模块切回
    training=True）。返回被冻结的 BN 模块数；层数 0 时不做任何事也不报错
    （ConvNeXt 臂没有 BN，属正常）。
    """
    import torch.nn as nn

    n = 0
    for name, m in model.named_modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)):
            if classify_parameter(name, tuple(backbone_prefixes)) == "backbone":
                m.eval()
                n += 1
    return n


def audit_norm_policy(model, *, backbone_prefixes: tuple, backbone_norm_eval: bool,
                      apply_policy: bool = True) -> dict:
    """审计归一化层策略（preflight/训练启动时输出，防止策略名不副实）。

    - 统计 BatchNorm（统计量冻结目标：running stats 不更新）与 LayerNorm 数量；
    - 检查 BN affine 参数 requires_grad（应 True = 可训练）；
    - backbone_norm_eval=True 时，先按策略执行 `model.train()` + `freeze_bn_stats()`
      （apply_policy=True），再验证 backbone 内 BN 确实处于 eval。
      注意：本函数会调用 model.train()。旧实现只 train() 不施加冻结，
      于是"模型级 train() 会把所有子模块切回 training=True"必然导致该检查
      永远失败（检查项与实现策略自相矛盾），这里把策略施加一并做掉。

    返回 dict；发现策略名不副实时把问题写进 issues（调用方决定是否 fail）。
    """
    import torch.nn as nn

    report = {
        "batchnorm_total": 0, "batchnorm_backbone": 0,
        "batchnorm_affine_trainable": True,
        "batchnorm_in_eval_after_train": True,
        "layernorm_total": 0,
        "groupnorm_total": 0,
        "policy_applied": bool(apply_policy and backbone_norm_eval),
        "issues": [],
    }
    was_training = model.training
    model.train()
    if apply_policy and backbone_norm_eval:
        freeze_bn_stats(model, backbone_prefixes=tuple(backbone_prefixes))
    for name, m in model.named_modules():
        is_backbone = classify_parameter(name, tuple(backbone_prefixes)) == "backbone"
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)):
            report["batchnorm_total"] += 1
            if is_backbone:
                report["batchnorm_backbone"] += 1
            if m.weight is not None and not m.weight.requires_grad:
                report["batchnorm_affine_trainable"] = False
                report["issues"].append(f"BN affine 不可训练: {name}")
            if backbone_norm_eval and is_backbone and m.training:
                report["batchnorm_in_eval_after_train"] = False
                report["issues"].append(f"backbone BN 未处于 eval（统计未冻结）: {name}")
        elif isinstance(m, (nn.LayerNorm,)) or "LayerNorm" in type(m).__name__:
            report["layernorm_total"] += 1
        elif isinstance(m, nn.GroupNorm):
            report["groupnorm_total"] += 1
    if not was_training:
        model.eval()
    return report
