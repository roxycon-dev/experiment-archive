# -*- coding: utf-8 -*-
"""trainer.gate — 配置占位符的两层闸门分类（单一权威定义）。

把 REQUIRED/TBD 占位符按"清空条件"分成两层：

- readiness（LOCAL_READY）：本地即可核验的结构/契约/数据完整性。
  该层存在 blocker 时连校准开机也不允许。
- calibration（GPU_REQUIRED）：只能在服务器 GPU 校准/资产到位后回填的字段
  （micro_batch/grad_accum/gradient_checkpointing、权重文件、外部 manifest 等）。
  该层未清空 *不阻止短时校准开机*，但 *绝不允许正式训练*（train.py formal 模式
  与 preflight_contract --layer calibration 共同强制）。

tools/preflight_contract.py 与 train.py 都从这里导入分类表，禁止各自维护。
"""

from __future__ import annotations

import hashlib
import os

PLACEHOLDER_TOKENS = ("REQUIRED", "TBD")

# ---------------------------------------------------------------------------
# 占位符分类：路径前缀 -> 所属层
# 精确到字段（前缀匹配），未列入的占位符一律视为 readiness blocker（fail closed）。
# ---------------------------------------------------------------------------

CALIBRATION_FIELD_PREFIXES = (
    # GPU 显存测速后定档
    "ddr_finetune_stage.micro_batch",
    "ddr_finetune_stage.grad_accum",
    "ddr_finetune_stage.gradient_checkpointing",
    "pretrain_stage.micro_batch",
    "pretrain_stage.grad_accum",
    "pretrain_stage.overfit_monitor",
    # 权重/资产文件（到位后回填 file+sha256）
    "initialization.imagenet_weights",
    "initialization.external_pretrain_checkpoint",
    "structure.encoder_weights_file",
    "structure.encoder_weights_sha256",
    "structure.encoder_weights_source",
    "structure.library_version",
    # 外部预训练资产
    "pretrain_stage.external_manifest",
    "pretrain_stage.manifest_base_dir",
    "pretrain_stage.init_checkpoint_output",
    # 部署路径（上传服务器后按实际路径回填）
    "structure.reference_root",
)


def is_placeholder(value) -> bool:
    """字符串值等于或包含 REQUIRED/TBD 即为占位符（与 preflight_contract 一致）。"""
    return isinstance(value, str) and (
        value.strip() in PLACEHOLDER_TOKENS
        or any(tok in value for tok in PLACEHOLDER_TOKENS)
    )


def scan_placeholders(obj, path: str = "$"):
    """递归收集占位符 (路径, 值) 列表。"""
    hits = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            hits += scan_placeholders(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            hits += scan_placeholders(v, f"{path}[{i}]")
    elif isinstance(obj, str):
        if is_placeholder(obj):
            hits.append((path, obj.strip()))
    return hits


def placeholder_layer(path: str) -> str:
    """占位符路径 -> 'readiness' | 'calibration'。未列入分类表的归 readiness。

    `scan_placeholders` 产出的路径带 `$.` 根前缀（如
    `$.ddr_finetune_stage.micro_batch`），分类表用的是去掉根前缀的点路径。
    旧实现直接拿带前缀的字符串去比，**永远匹配不到**，于是所有 calibration
    字段都被误判成 readiness —— 两层闸门形同失效。此处显式剥离 `$` 根。
    """
    norm = path[2:] if path.startswith("$.") else ("$" if path == "$" else path)
    for prefix in CALIBRATION_FIELD_PREFIXES:
        if norm == prefix or norm.startswith(prefix + "."):
            return "calibration"
    return "readiness"


def split_blockers(cfg) -> dict:
    """把配置中的全部占位符按层分组。

    返回 {"readiness": [(path, value)...], "calibration": [(path, value)...]}。
    """
    out = {"readiness": [], "calibration": []}
    for path, value in scan_placeholders(cfg):
        out[placeholder_layer(path)].append((path, value))
    return out


# ---------------------------------------------------------------------------
# 通用配置工具
# ---------------------------------------------------------------------------

def sha256_file(path: str, chunk_size: int = 1 << 20) -> str:
    """流式计算文件 SHA256；文件不存在抛 FileNotFoundError。"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def deep_get(obj, dotted: str, default=None):
    cur = obj
    for part in dotted.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return default
    return cur


def config_sha256(config_path: str) -> str:
    """配置文件字节级 SHA256（写入 checkpoint，用于断点恢复一致性校验）。"""
    return sha256_file(config_path)


def repo_root() -> str:
    """工作区根目录（trainer/ 的上一级）。"""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class GateError(Exception):
    """配置闸门失败（fail closed）。"""
