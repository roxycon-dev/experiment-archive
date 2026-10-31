# -*- coding: utf-8 -*-
"""trainer.models — 模型构建入口。

- family=hrdecoder：H 臂。以 adapter 方式复用旧仓库（只读 reference_root）中的
  HRNet-W48 主干与 FCNHead，组装为"单个 HRDecoder-W48 学生"：HRNet 四尺度特征
  → FCN 头 → 4 通道 logits → 上采样回输入分辨率。旧融合包装（HRDecoder
  segmentor 的多裁剪融合 / learnable 融合）、旧 MA residual head
  （detail_branch/detail_head/detail_scale）、旧 frozen B 副本一律不进入新模型
  （见 trainer/models/hrdecoder_adapter.py 顶部说明）。
- family=convnext_unet：U 臂。builtin 纯 torch 实现（ConvNeXt-S 编码器 +
  逐级 U-Net decoder，stride-1 全分辨率输出），不依赖 segmentation_models_pytorch
  （依赖不可离线核实 ⇒ 消除依赖；权重格式锁定 facebookresearch 官方 checkpoint）。

依赖缺失时抛 DependencyBlocked（preflight/smoke 显式分类，不算 PASS 也不算 FAIL）。
"""

from __future__ import annotations


class DependencyBlocked(Exception):
    """真实依赖（torch/mmcv/mmengine/mmseg/权重文件）缺失，无法在本机执行该检查。

    语义：不算 PASS 也不算 FAIL；报告单列 DEPENDENCY_BLOCKED，不计入通过数。
    """


def build_model(cfg: dict, *, stage: str, gradient_checkpointing: bool | None = None):
    """按配置 structure.family 构建模型（未加载权重）。

    Returns:
        (model, meta)：meta 含 backbone_prefixes / backbone_norm_eval / 权重报告。
    """
    family = (cfg.get("structure") or {}).get("family")
    if family == "hrdecoder":
        from .hrdecoder_adapter import build_hr_decoder_student

        model, meta = build_hr_decoder_student(cfg, gradient_checkpointing=gradient_checkpointing)
    elif family == "convnext_unet":
        from .convnext_unet import build_convnext_unet

        model, meta = build_convnext_unet(cfg, gradient_checkpointing=gradient_checkpointing)
    elif family == "hacdr":
        from .hacdr_adapter import build_hacdr
        model, meta = build_hacdr(cfg, gradient_checkpointing=gradient_checkpointing)
    else:
        raise ValueError(f"未知 structure.family={family!r}")
    # Transfer is an initialization, not a resume: load the complete encoder
    # AND decoder, but start a fresh optimizer/scheduler for DDR adaptation.
    if stage == "ddr_finetune" and cfg['initialization'].get('external_pretrain_checkpoint') is not None:
        from .weight_loading import load_pretrained_subset
        from ..checkpoint import load_checkpoint
        from ..gate import GateError
        ext = cfg['initialization']['external_pretrain_checkpoint']
        if not isinstance(ext, dict) or not ext.get('file'):
            raise GateError('DDR transfer requires a concrete external checkpoint')
        ckpt = load_checkpoint(ext['file'])
        if ckpt['stage'] != 'external_pretrain' or ckpt['is_calibration'] or ckpt['family'] != family:
            raise GateError('External initialization must be a formal same-family external_pretrain checkpoint')
        meta['external_weight_report'] = load_pretrained_subset(
            model, ckpt_path=ext['file'], label='external_complete_model',
            sha256_expected=ext.get('sha256'))
        del ckpt
    warm = cfg['initialization'].get('ddr_warm_start_checkpoint')
    if stage == 'ddr_finetune' and warm is not None:
        from .weight_loading import load_pretrained_subset
        from ..checkpoint import load_checkpoint
        from ..gate import GateError
        ckpt = load_checkpoint(warm['file'])
        if ckpt['stage']!='ddr_finetune' or ckpt['is_calibration'] or ckpt['family']!=family:
            raise GateError('Continuation requires a formal same-family DDR checkpoint')
        meta['warm_start_report']=load_pretrained_subset(model,ckpt_path=warm['file'],
            label='ddr_continuation_complete_model',sha256_expected=warm.get('sha256'))
        del ckpt
    return model, meta


def probe_family(family: str, cfg: dict) -> dict:
    """不构建模型的依赖探测（preflight 用）：返回 {"status", "detail"}。"""
    if family == "hrdecoder":
        from .hrdecoder_adapter import probe_reference
        from ..config import require_reference_root
        from ..gate import GateError

        try:
            root = require_reference_root(cfg)
        except GateError as e:
            # reference_root 属于 calibration 层（部署路径，上传服务器后回填）。
            # 未回填 ⇒ 本机无法验证 H 臂结构 ⇒ DEPENDENCY_BLOCKED
            # （不是 READINESS_BLOCKED：它不该阻止先开一次短时 GPU 校准）。
            return {"status": "DEPENDENCY_BLOCKED",
                    "detail": f"structure.reference_root 未回填（calibration 层）: {e}"}
        rep = probe_reference(root)
        return {
            "status": "OK" if rep["importable"] else "DEPENDENCY_BLOCKED",
            "detail": rep,
        }
    if family == "convnext_unet":
        st = cfg.get("structure") or {}
        lib = str(st.get("library", "")).lower()
        if lib not in ("builtin", "builtin_convnext_unet"):
            return {
                "status": "READINESS_BLOCKED",
                "detail": (
                    f"structure.library={st.get('library')!r}：v1 只实现 builtin 纯 torch "
                    "ConvNeXt-S；声明第三方库时结构来源不可离线核实。"
                ),
            }
        return {"status": "OK", "detail": {"library": "builtin"}}
    return {"status": "READINESS_BLOCKED", "detail": f"未知 family={family!r}"}
