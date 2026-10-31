# -*- coding: utf-8 -*-
"""trainer.checkpoint — 训练状态 checkpoint（原子写、断点恢复、双哈希一致性）。

checkpoint 记录：模型 / 优化器 / scheduler / GradScaler / optimizer update 计数 /
micro-step / seed / config SHA256 / 数据 manifest SHA256 / 全部 RNG 态 /
torch 版本 / 是否校准产物。恢复时校验 arm/stage/config SHA256，不一致 fail closed
（除非显式 --allow-config-mismatch，且该豁免本身写入恢复日志）。

反序列化安全：checkpoint 载荷由本包构造，RNG/指标等全部归一化为纯原始类型
（见 trainer.seed.capture_rng），因此加载固定使用 torch.load(weights_only=True)，
拒绝执行任意 pickle 对象。这要求 torch>=1.13（旧服务器栈 mmcv-full 1.7.2
搭配的 torch 1.13.x 满足）；更低版本直接报错，绝不回退到不安全加载。
"""

from __future__ import annotations

import os
import tempfile
from datetime import datetime, timezone

import torch

from .paths import ensure_parent_dir

CHECKPOINT_FORMAT = "hrdecoder-training/v1"
_MIN_TORCH_FOR_SAFE_LOAD = (1, 13)


def _torch_load(path: str) -> dict:
    """仅允许 weights_only 安全加载；torch 版本不足时 fail closed。"""
    version = tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2])
    if version < _MIN_TORCH_FOR_SAFE_LOAD:
        raise RuntimeError(
            f"checkpoint 安全加载需要 torch>={_MIN_TORCH_FOR_SAFE_LOAD[0]}."
            f"{_MIN_TORCH_FOR_SAFE_LOAD[1]}（weights_only 支持），"
            f"当前 {torch.__version__}。拒绝回退到不安全的 pickle 加载。"
        )
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
    except pickle_errors() as e:
        raise ValueError(
            f"checkpoint 无法以 weights_only 安全模式加载（{e}）。"
            "只允许加载本 trainer 产出的 checkpoint；外部权重请走模型初始化路径。"
        ) from e
    return ckpt


def pickle_errors():
    """weights_only 模式下常见失败异常集合（版本间 API 有差异，集中定义）。"""
    return (ValueError, RuntimeError, TypeError, KeyError, AttributeError)


def _plain(obj):
    """把 numpy/张量/str 子类标量递归归一化为 weights_only 兼容的纯原始类型。

    torch.load(weights_only=True) 只接受 dict/list/tuple/str/int/float/bool/None
    与 Tensor；numpy.float64、以及 **torch.__version__ 返回的 TorchVersion**
    （torch>=2.5 起是 str 子类，不在安全全局名单里）都会让加载直接失败。
    写入前必须归一，且失败要显式抛错（禁止静默丢字段）。
    """
    if obj is None or isinstance(obj, bool):
        return obj
    if isinstance(obj, int):
        return int(obj)
    if isinstance(obj, float):
        return float(obj)
    if isinstance(obj, str):          # 含 TorchVersion 等 str 子类
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    if hasattr(obj, "item") and not torch.is_tensor(obj):  # numpy 标量
        return _plain(obj.item())
    raise TypeError(
        f"checkpoint 载荷含不可安全序列化的类型 {type(obj).__name__}；"
        "请先转换为原始类型（禁止静默丢弃字段）。"
    )


def save_checkpoint(path: str, payload: dict) -> str:
    """原子保存 checkpoint（同目录临时文件 + os.replace）。

    临时文件必须带 `.pth` 后缀：torch.save 依据**文件扩展名**选择序列化后端，
    `tempfile.mkstemp(prefix=".tmp_ckpt_")` 生成的名字没有扩展名，会直接抛
    `RuntimeError: invalid file name`（历史实现的 checkpoint 保存 100% 失败）。
    """
    path = ensure_parent_dir(path)
    parent = os.path.dirname(path)
    fd, tmp = tempfile.mkstemp(prefix=".tmp_ckpt_", suffix=".pth", dir=parent)
    os.close(fd)
    try:
        torch.save(payload, tmp)
        # 自检：立刻用 weights_only=True 回读一次。宁可在保存时就失败，
        # 也不要等几小时后恢复时才发现 checkpoint 根本读不出来。
        _verify_safely_loadable(tmp)
        os.replace(tmp, path)
    except Exception:
        if os.path.isfile(tmp):
            os.remove(tmp)
        raise
    return path


def _verify_safely_loadable(path: str) -> None:
    try:
        torch.load(path, map_location="cpu", weights_only=True)
    except Exception as e:
        raise ValueError(
            "checkpoint 写入自检失败：产物无法用 weights_only=True 安全加载"
            f"（{type(e).__name__}: {e}）。"
            "通常是载荷里混入了未归一化的对象（如 TorchVersion / numpy 标量）。"
        ) from e


def build_checkpoint_payload(
    *,
    model,
    optimizer,
    scheduler_state: dict,
    scaler,
    update: int,
    micro_step: int,
    total_updates: int,
    seed: int,
    config_sha256: str,
    data_manifest_sha256: str,
    manifest_path,
    arm_id: str,
    family: str,
    stage: str,
    mode: str,
    is_calibration: bool,
    metrics_at_save,
    extra=None,
) -> dict:
    from .seed import capture_rng

    payload = {
        "format": CHECKPOINT_FORMAT,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "arm_id": arm_id,
        "family": family,
        "stage": stage,
        "mode": mode,
        "is_calibration": bool(is_calibration),
        "update": int(update),
        "micro_step": int(micro_step),
        "total_updates": int(total_updates),
        "seed": int(seed),
        "config_sha256": config_sha256,
        "data_manifest_sha256": None if data_manifest_sha256 is None else str(data_manifest_sha256),
        "manifest_path": None if manifest_path is None else str(manifest_path),
        "torch_version": str(torch.__version__),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": _plain(dict(scheduler_state)),
        "scaler_state": None if scaler is None else _plain(scaler.state_dict()),
        "rng": capture_rng(),
        "metrics_at_save": _plain(metrics_at_save or {}),
    }
    if extra:
        for k, v in extra.items():
            if k in payload:
                raise ValueError(f"checkpoint extra 字段 {k!r} 与保留字段重名")
            payload[k] = _plain(v)
    return payload


def load_checkpoint(path: str) -> dict:
    """读取 checkpoint 并做格式校验（不恢复状态）。"""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"checkpoint 不存在: {path}")
    ckpt = _torch_load(path)
    if not isinstance(ckpt, dict) or ckpt.get("format") != CHECKPOINT_FORMAT:
        got = ckpt.get("format") if isinstance(ckpt, dict) else type(ckpt).__name__
        raise ValueError(
            f"checkpoint 格式不符：期望 {CHECKPOINT_FORMAT}，实际 {got}（{path}）。"
            "外部 ImageNet/官方权重请走模型初始化路径，不用本函数。"
        )
    return ckpt


def verify_resume_compatibility(
    ckpt: dict,
    *,
    arm_id: str,
    stage: str,
    config_sha256: str,
    data_manifest_sha256: str | None = None,
    allow_config_mismatch: bool = False,
    allow_manifest_mismatch: bool = False,
    allow_calibration_checkpoint: bool = False,
) -> list:
    """断点恢复前的一致性校验，返回警告列表；致命不一致抛异常。

    校准 checkpoint 不允许作为正式训练恢复源（is_calibration 标记隔离）；
    数据 manifest 指纹变化意味着训练数据统计环境已变，默认拒绝恢复。
    """
    if ckpt.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"checkpoint format={ckpt.get('format')!r} 不是 {CHECKPOINT_FORMAT}")
    warnings = []
    if ckpt.get("arm_id") != arm_id:
        raise ValueError(f"checkpoint arm_id={ckpt.get('arm_id')!r} 与当前 {arm_id!r} 不一致")
    if ckpt.get("stage") != stage:
        raise ValueError(f"checkpoint stage={ckpt.get('stage')!r} 与当前 {stage!r} 不一致")
    if ckpt.get("is_calibration") and not allow_calibration_checkpoint:
        raise ValueError(
            "checkpoint 标记为校准产物（is_calibration=true），禁止用于正式训练恢复"
        )
    if ckpt.get("config_sha256") != config_sha256:
        if not allow_config_mismatch:
            raise ValueError(
                "config SHA256 与 checkpoint 记录不一致"
                f"（ckpt={ckpt.get('config_sha256')!r}, current={config_sha256!r}）。"
                "如确认配置变更可接受，用 --allow-config-mismatch 显式豁免（会记录在日志）。"
            )
        warnings.append("config SHA256 不一致，已按 --allow-config-mismatch 显式豁免")
    cur_man = None if data_manifest_sha256 is None else str(data_manifest_sha256)
    ck_man = ckpt.get("data_manifest_sha256")
    if cur_man is not None and ck_man is not None and cur_man != ck_man:
        if not allow_manifest_mismatch:
            raise ValueError(
                "数据 manifest SHA256 与 checkpoint 记录不一致"
                f"（ckpt={ck_man!r}, current={cur_man!r}）——训练数据已变，"
                "用 --allow-manifest-mismatch 显式豁免（会记录在日志）。"
            )
        warnings.append("数据 manifest SHA256 不一致，已按 --allow-manifest-mismatch 显式豁免")
    if ckpt.get("mode") == "calibration" and not allow_calibration_checkpoint:
        raise ValueError("checkpoint mode=calibration，禁止作为正式训练恢复源")
    return warnings


def resume_state(ckpt: dict, *, model, optimizer, scaler, restore_rng_fn=None) -> dict:
    """把 checkpoint 状态恢复到 model/optimizer/scaler/RNG，返回恢复出的计数。

    scheduler 为手工实现的 warmup+cosine（trainer.optim.lr_multiplier），
    其"状态"就是 update 计数，因此以返回值中的 update 为准，
    绝不重放/累加（否则断点后 update 会跳变）。
    """
    model.load_state_dict(ckpt["model_state"], strict=True)
    if ckpt.get("optimizer_state") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    if scaler is not None and ckpt.get("scaler_state") is not None:
        scaler.load_state_dict(ckpt["scaler_state"])
    if restore_rng_fn is None:
        from .seed import restore_rng as restore_rng_fn  # noqa: PLC0415
    rng = ckpt.get("rng")
    if not isinstance(rng, dict):
        raise ValueError("checkpoint 缺少 RNG 状态，拒绝在不一致的随机流下继续训练")
    restore_rng_fn(rng)
    sched = ckpt.get("scheduler_state") or {}
    return {
        "update": int(ckpt.get("update", 0)),
        "micro_step": int(ckpt.get("micro_step", 0)),
        "total_updates": int(ckpt.get("total_updates", sched.get("total_updates", 0))),
        "scheduler_update": int(sched.get("update", ckpt.get("update", 0))),
        "seed": int(ckpt.get("seed", 0)),
    }
