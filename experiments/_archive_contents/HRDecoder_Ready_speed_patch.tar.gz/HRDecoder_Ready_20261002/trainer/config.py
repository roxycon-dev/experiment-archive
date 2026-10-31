# -*- coding: utf-8 -*-
"""trainer.config — 单臂配置加载与闸门（train.py / preflight_training 共用）。

职责（全部 fail closed）：
1. 解析 YAML；校验 schema_version / arm_id / status / 必需节。
2. 阶段解析：`external_pretrain` ↔ `pretrain_stage`、`ddr_finetune` ↔
   `ddr_finetune_stage`；控制臂（H0/U0）**没有** pretrain_stage，请求该阶段直接报错。
3. split 闸门：训练与选点 split 只能是 train/val；任何位置出现 test 一律硬拒绝
   （CLI 层 `assert_no_test_split`）。
4. 两层闸门（trainer.gate 单一权威分类）：
   - readiness（LOCAL_READY）：占位符全部清空才允许进入短时 GPU 校准；
   - calibration（GPU_REQUIRED）：可在 `--mode calibration` 下由 CLI 覆盖，
     但 `--mode formal` 要求**两层都为空**（正式训练绝不放宽）。
5. 生效 schema 解析：把占位符/CLI 覆盖合并成 `EffectiveStage`（供 engine 使用），
   并把最终生效值回写进 config dict（使 config SHA256 覆盖 CLI 覆盖）。

本模块不读图像像素、不联网、不导入 torch 之外的深度依赖。
"""

from __future__ import annotations

import copy
import os

from .gate import (
    config_sha256,
    deep_get,
    is_placeholder,
    split_blockers,
)
from .paths import repo_root
from .gate import GateError

STAGES = ("external_pretrain", "ddr_finetune")
STAGE_SECTION = {
    "external_pretrain": "pretrain_stage",
    "ddr_finetune": "ddr_finetune_stage",
}
# 控制臂不允许有外部预训练段
CONTROL_ARMS = ("H0", "U0")
MODES = ("formal", "calibration")

# CLI 可覆盖的"校准层"字段（仅 calibration 模式；formal 模式禁止）
CALIBRATION_OVERRIDES = ("micro_batch", "grad_accum", "gradient_checkpointing")

DEFAULT_NORMALIZATION = {
    "imagenet": {"mean": [123.675, 116.28, 103.53], "std": [58.395, 57.12, 57.375]},
}


def load_yaml(path: str) -> dict:
    try:
        import yaml
    except Exception as e:  # 显式失败，绝不静默
        raise GateError(
            f"PyYAML 未安装（import yaml 失败: {e}）；配置加载必须解析 YAML。"
            "请 `python -m pip install pyyaml`。"
        ) from e
    if not os.path.isfile(path):
        raise GateError(f"配置文件不存在: {path}")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise GateError(f"配置根必须是映射: {path}")
    return cfg


def validate_config(cfg: dict, path: str = "<cfg>") -> None:
    """结构校验（不涉及占位符闸门）。"""
    if cfg.get("schema_version") != "1.0":
        raise GateError(f"{path}: schema_version 必须为 '1.0'，当前 {cfg.get('schema_version')!r}")
    arm = cfg.get("arm_id")
    if arm not in ("H0", "H1", "U0", "U1", 'A0'):
        raise GateError(f"{path}: arm_id 非法: {arm!r}")
    for key in ("structure", "initialization", "data", "ddr_finetune_stage",
                "loss", "metrics", "checkpoint_selection", "test_usage"):
        if not isinstance(cfg.get(key), dict):
            raise GateError(f"{path}: 缺少或非映射的顶层节 {key!r}")
    if arm in CONTROL_ARMS and isinstance(cfg.get("pretrain_stage"), dict):
        raise GateError(f"{path}: 控制臂 {arm} 不允许存在 pretrain_stage 节")
    if arm not in CONTROL_ARMS + ('A0',) and not isinstance(cfg.get("pretrain_stage"), dict):
        raise GateError(f"{path}: 预训练臂 {arm} 必须提供 pretrain_stage 节")
    fam = cfg["structure"].get("family")
    if fam not in ("hrdecoder", "convnext_unet", 'hacdr'):
        raise GateError(f"{path}: structure.family 非法: {fam!r}")
    return None


def assert_no_test_split(cfg: dict, *, split: str, role: str) -> None:
    """硬拒绝 test 作为训练/选点 split（CLI 层最后一道闸）。

    只按**数值语义**判断（split 名等于 "test"），不做路径字符串模糊匹配——
    路径级的 test 引用由 tools/preflight_contract.py 的 F/G 项负责。
    """
    if str(split).strip().lower() == "test":
        raise GateError(
            f"{role} 不得使用 test split（当前 {split!r}）。"
            "test 仅允许在四项实验全部锁定后做一次最终评估，"
            "绝不参与训练或 checkpoint 选型。"
        )
    if str(split).strip().lower() not in ("train", "val"):
        raise GateError(f"{role} 的 split 只能是 train/val，收到 {split!r}")


def checkpoint_selection_splits(cfg: dict) -> tuple:
    cs = cfg.get("checkpoint_selection") or {}
    allowed = cs.get("allowed_splits")
    if allowed != ["val"]:
        raise GateError(
            f"checkpoint_selection.allowed_splits 必须恰为 ['val']（禁止 test 参与选型），当前 {allowed!r}"
        )
    if cs.get("test_forbidden") is not True:
        raise GateError("checkpoint_selection.test_forbidden 必须为 true")
    tu = cfg.get("test_usage") or {}
    if tu.get("train_on_test") != "forbidden" or tu.get("select_checkpoint_on_test") != "forbidden":
        raise GateError("test_usage 的 train_on_test / select_checkpoint_on_test 必须均为 forbidden")
    return ("val",)


def resolve_stage_section(cfg: dict, stage: str) -> dict:
    """阶段名 → 配置节（含控制臂拒绝 pretrain_stage）。"""
    if stage not in STAGES:
        raise GateError(f"未知阶段 {stage!r}（支持 {STAGES}）")
    section = STAGE_SECTION[stage]
    arm = cfg.get("arm_id")
    if stage == "external_pretrain" and arm in CONTROL_ARMS:
        raise GateError(
            f"控制臂 {arm} 没有 external_pretrain 阶段（配置显式禁止外部预训练权重）"
        )
    sec = cfg.get(section)
    if not isinstance(sec, dict):
        raise GateError(f"配置缺少 {section} 节（stage={stage}）")
    return sec


def placeholder_report(cfg: dict) -> dict:
    """两层占位符报告（readiness / calibration）。"""
    blocks = split_blockers(cfg)
    return {
        "readiness": [p for p, _ in blocks["readiness"]],
        "calibration": [p for p, _ in blocks["calibration"]],
        "readiness_detail": blocks["readiness"],
        "calibration_detail": blocks["calibration"],
        "readiness_state": "LOCAL_READY" if not blocks["readiness"] else "BLOCKED",
        "calibration_state": "READY" if not blocks["calibration"] else "GPU_REQUIRED",
    }


def gate_config(cfg: dict, *, mode: str, stage: str, path: str = "<cfg>") -> dict:
    """闸门：按 mode 决定哪一层占位符可以放行。

    - mode="formal"：readiness 与 calibration 都必须清空；
    - mode="calibration"：只要求 readiness 清空；calibration 里与本阶段相关的
      字段必须由 CLI 覆盖（见 apply_calibration_overrides），其余记录为待校准。
    绝不因为 calibration 层未清空而放宽 formal。
    """
    if mode not in MODES:
        raise GateError(f"未知 mode {mode!r}（支持 {MODES}）")
    # A future external checkpoint is an output of pretraining, not an input.
    # Requiring it before the first pretrain update creates a circular gate.
    active_cfg = copy.deepcopy(cfg)
    if stage == 'external_pretrain':
        active_cfg['initialization']['external_pretrain_checkpoint'] = None
    rep = placeholder_report(active_cfg)
    if rep["readiness"]:
        raise GateError(
            f"{path}: readiness 层仍有 {len(rep['readiness'])} 个占位符，连校准都不允许："
            f"{rep['readiness'][:10]}"
        )
    if mode == "formal" and rep["calibration"]:
        raise GateError(
            f"{path}: 正式训练（mode=formal）要求配置无任何占位符，"
            f"仍有 {len(rep['calibration'])} 个 calibration 层占位符："
            f"{rep['calibration'][:10]}。"
            "先用 --mode calibration 完成显存校准并回填，再用 formal 开训。"
        )
    return rep


def apply_calibration_overrides(cfg: dict, *, stage: str, overrides: dict) -> dict:
    """把 CLI 校准值写进配置（深度拷贝，不改调用方对象）。

    formal 模式下禁止覆盖：正式训练的批量/累积/重计算必须来自配置文件本身，
    否则 config SHA256 就不再是"实际跑的那份配置"。
    """
    cfg = copy.deepcopy(cfg)
    sec = resolve_stage_section(cfg, stage)
    applied = {}
    for key, value in (overrides or {}).items():
        if value is None:
            continue
        if key not in CALIBRATION_OVERRIDES:
            raise GateError(f"{key!r} 不是可用的校准覆盖字段（{CALIBRATION_OVERRIDES}）")
        sec[key] = value
        applied[key] = value
    return cfg, applied


def effective_batch_check(stage_cfg: dict, *, stage: str) -> dict:
    """effective_batch == micro_batch * grad_accum（占位符时报错，绝不猜）。"""
    eb = stage_cfg.get("effective_batch")
    mb = stage_cfg.get("micro_batch")
    ga = stage_cfg.get("grad_accum")
    for name, v in (("effective_batch", eb), ("micro_batch", mb), ("grad_accum", ga)):
        if is_placeholder(v) or not isinstance(v, int) or isinstance(v, bool) or v <= 0:
            raise GateError(
                f"{stage}: {name} 必须是正整数（当前 {v!r}）——"
                "正式训练不接受占位符/推断值，请先完成 GPU 校准回填。"
            )
    if eb != mb * ga:
        raise GateError(
            f"{stage}: effective_batch={eb} != micro_batch={mb} × grad_accum={ga}（批量守恒破坏）"
        )
    return {"effective_batch": eb, "micro_batch": mb, "grad_accum": ga}


def ddr_data_config(cfg: dict) -> dict:
    data = cfg["data"]
    root = data.get("ddr_root")
    if is_placeholder(root) or not isinstance(root, str) or not root:
        raise GateError(f"data.ddr_root 未填写: {root!r}")
    if not os.path.isdir(root):
        raise GateError(f"data.ddr_root 目录不存在: {root}")
    train_split = data.get("ddr_train_split")
    val_split = data.get("ddr_val_split")
    assert_no_test_split(cfg, split=train_split, role="data.ddr_train_split")
    assert_no_test_split(cfg, split=val_split, role="data.ddr_val_split")
    if train_split == val_split:
        raise GateError(f"train/val split 不得相同（{train_split!r}）")
    norm = data.get("ddr_normalization") or {}
    mean, std = norm.get("mean"), norm.get("std")
    if is_placeholder(mean) or is_placeholder(std):
        raise GateError(
            "data.ddr_normalization.mean/std 未填写——禁止用 ImageNet 默认值冒充（会改变输入分布）"
        )
    if not (isinstance(mean, list) and isinstance(std, list) and len(mean) == 3 and len(std) == 3):
        raise GateError(f"data.ddr_normalization 必须各 3 个数，当前 mean={mean!r} std={std!r}")
    if any(float(s) <= 0 for s in std):
        raise GateError(f"data.ddr_normalization.std 必须为正: {std!r}")
    channel_order = data.get("channel_order")
    from .data.common import LESION_CHANNEL_ORDER
    if list(channel_order or []) != list(LESION_CHANNEL_ORDER):
        raise GateError(
            f"data.channel_order={channel_order!r} 必须等于 {list(LESION_CHANNEL_ORDER)}（契约通道序）"
        )
    return {
        "ddr_root": root,
        "train_split": train_split,
        "val_split": val_split,
        "mean": [float(x) for x in mean],
        "std": [float(x) for x in std],
        "train_crop": int(data.get("train_crop") or 0) or None,
        "channel_order": list(channel_order),
    }


def metrics_config(cfg: dict) -> dict:
    m = cfg.get("metrics") or {}
    protocol = m.get("protocol")
    if protocol != "whole_image_accumulated_counts":
        raise GateError(
            f"metrics.protocol={protocol!r} 必须是 whole_image_accumulated_counts"
            "（禁止 crop 指标冒充 val）"
        )
    thr = m.get("threshold", 0.5)
    if is_placeholder(thr):
        raise GateError("metrics.threshold 为占位符")
    return {
        "threshold": float(thr),
        "primary": m.get("primary"),
        "reported": list(m.get("reported") or []),
        "eval_mode": str(m.get("eval_mode", "whole_image")),
        "window_size": m.get("window_size"),
        "window_stride": m.get("window_stride"),
    }


def require_reference_root(cfg: dict) -> str:
    """H 臂的旧仓库只读根目录；占位符/缺失时 fail closed（不猜路径）。"""
    st = cfg.get("structure") or {}
    ref = st.get("reference_root")
    if is_placeholder(ref) or not isinstance(ref, str) or not ref:
        # 允许配置未声明时回退到环境变量，但绝不猜一个"看起来像"的默认路径
        env = os.environ.get("HRDECODER_REFERENCE_ROOT")
        if env and os.path.isdir(env):
            return env
        raise GateError(
            "structure.reference_root 未填写（H 臂 adapter 需要旧仓库只读根目录）。"
            "请设置配置字段或环境变量 HRDECODER_REFERENCE_ROOT；本工具不猜路径。"
        )
    if not os.path.isdir(ref):
        raise GateError(f"structure.reference_root 目录不存在: {ref}")
    return ref


def config_digest(cfg_path: str) -> str:
    return config_sha256(cfg_path)


def describe(cfg: dict) -> dict:
    """给日志用的配置摘要（不含敏感/资产细节）。"""
    return {
        "arm_id": cfg.get("arm_id"),
        "family": deep_get(cfg, "structure.family"),
        "paired_group": deep_get(cfg, "structure.paired_group"),
        "status": cfg.get("status"),
        "training_entry": deep_get(cfg, "structure.training_entry"),
        "repo_root": repo_root(),
    }
