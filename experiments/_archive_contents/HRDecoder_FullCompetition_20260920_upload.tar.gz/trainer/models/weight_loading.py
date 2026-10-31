# -*- coding: utf-8 -*-
"""trainer.models.weight_loading — 预训练权重加载（strict / 显式 missing-unexpected 报告）。

规则（禁止静默部分加载）：
1. 文件 SHA256 必须与配置一致（配置提供时）。
2. 支持 mmcv 风格 {meta,state_dict} 与纯 state_dict 两种容器；自动剥
   "module."（DataParallel）与单一顶层 "backbone."（分割整模 checkpoint）前缀。
3. load_state_dict(strict=False) 后，missing/unexpected 全部显式报告：
   - missing 命中 allow_missing 之外的任何键 → 抛错；
   - unexpected 命中 allow_unexpected 之外的任何键 → 抛错。
4. 报告永远完整打印（不截断不静默），并在返回值中带回。
"""

from __future__ import annotations

import os
import re

import torch

from ..gate import GateError
from ..data.common import file_sha256


def load_pretrained_subset(
    module: torch.nn.Module,
    *,
    ckpt_path: str,
    label: str,
    allow_missing=(),
    allow_unexpected=(),
    sha256_expected: str | None = None,
) -> dict:
    """把 checkpoint 加载进 module（通常是其 backbone/encoder 子模块）。

    Args:
        allow_missing / allow_unexpected: 正则白名单（re.search）；
            白名单外的 missing/unexpected 一律抛 GateError（fail closed）。
    Returns:
        {"missing": [...], "unexpected": [...], "loaded_keys": n, "remapped_keys": {...}}
    """
    if not os.path.isfile(ckpt_path):
        raise GateError(f"{label}: 权重文件不存在: {ckpt_path}")
    if sha256_expected and str(sha256_expected).strip().lower() not in ("", "none"):
        actual = file_sha256(ckpt_path)
        if actual != str(sha256_expected).strip().lower():
            raise GateError(
                f"{label}: SHA256 不一致（配置 {str(sha256_expected)[:12]}… != 实际 {actual[:12]}…）"
            )

    try:
        obj = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except Exception as e:
        # 外部官方权重可能含 numpy 标量等非白名单对象：weights_only 拒绝属预期
        raise GateError(
            f"{label}: 权重无法以 weights_only 安全模式加载（{type(e).__name__}: {e}）。"
            "本实现不回退到不安全反序列化；请先转换权重格式（例如只保留 tensor 的 state_dict）。"
        ) from e

    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        sd = dict(obj["state_dict"])
    elif isinstance(obj, dict) and "model" in obj and isinstance(obj["model"], dict):
        sd = dict(obj["model"])
    elif isinstance(obj, dict):
        sd = dict(obj)
    else:
        raise GateError(f"{label}: 不是 state_dict 容器（{type(obj).__name__}）")

    sd, remapped = _strip_prefixes(sd)
    sd, key_alias = _apply_key_aliases(sd)
    if key_alias:
        remapped["key_alias"] = key_alias

    target_keys = set(module.state_dict().keys())
    compatible = {k: v for k, v in sd.items() if k in target_keys and hasattr(v, "shape")}
    shape_bad = [
        k for k, v in compatible.items()
        if tuple(v.shape) != tuple(module.state_dict()[k].shape)
    ]
    if shape_bad:
        raise GateError(
            f"{label}: 权重形状与模型不匹配（{len(shape_bad)} 个），例如 {shape_bad[:5]}。"
            "结构不一致时绝不静默跳过（会得到随机初始化的子模块）。"
        )
    incompatible = module.load_state_dict(sd, strict=False)
    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)

    miss_bad = [k for k in missing if not _matches(k, allow_missing)]
    unexp_bad = [k for k in unexpected if not _matches(k, allow_unexpected)]
    if miss_bad or unexp_bad:
        raise GateError(
            f"{label}: 静默部分加载被拒绝。\n"
            f"  missing(白名单外 {len(miss_bad)}/{len(missing)}): {miss_bad[:20]}\n"
            f"  unexpected(白名单外 {len(unexp_bad)}/{len(unexpected)}): {unexp_bad[:20]}\n"
            f"  请核对权重文件与结构是否匹配；显式扩白名单需在配置中声明"
            f"（accept_missing_patterns / accept_unexpected_patterns）。"
        )
    return {
        "label": label,
        "missing": missing,
        "unexpected": unexpected,
        "loaded_keys": len(compatible),
        "total_ckpt_keys": len(sd),
        "remapped": remapped,
    }


# 旧键 -> 官方键（历史实现把 dwconv 包了一层容器，键变成 dwconv.dwconv.weight）
KEY_ALIAS_PATTERNS = (
    (re.compile(r"^(.*\.dwconv)\.dwconv\.(weight|bias)$"), r"\1.\2"),
)


def _apply_key_aliases(sd: dict) -> tuple:
    """把历史键名归一化到官方键名（只做显式列出的别名，不猜）。"""
    out = {}
    hits = {}
    for k, v in sd.items():
        new = k
        for pat, repl in KEY_ALIAS_PATTERNS:
            cand = pat.sub(repl, k)
            if cand != k:
                new = cand
                hits.setdefault(k, cand)
                break
        out[new] = v
    return out, hits


def _strip_prefixes(sd: dict) -> tuple:
    """剥 module. 前缀；若绝大多数键带同一顶层 backbone. 前缀则一并剥除。

    返回 (sd, remapped)；任何键名改写都显式记录，绝不静默重命名。
    """
    remapped = {}
    if sd and all(k.startswith("module.") for k in sd):
        remapped["prefix_stripped"] = "module."
        sd = {k[len("module."):]: v for k, v in sd.items()}
    if sd:
        n_back = sum(1 for k in sd if k.startswith("backbone."))
        if n_back > 0.9 * len(sd):
            remapped["prefix_stripped"] = "backbone."
            sd = {k[len("backbone."):] if k.startswith("backbone.") else k: v
                  for k, v in sd.items()}
    return sd, remapped


def _matches(key: str, patterns) -> bool:
    return any(re.search(p, key) for p in patterns)
