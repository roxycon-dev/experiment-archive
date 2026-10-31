# -*- coding: utf-8 -*-
"""trainer.models.hrdecoder_adapter — H 臂：单个 HRDecoder-W48 学生。

复用范围（只读 `structure.reference_root` 指向的旧仓库，绝不修改它）：
- `mmseg.models.backbones.hrnet.HRNet`（HRNetV2-W48，stage4 通道 48/96/192/384）
- `mmseg.models.decode_heads.fcn_head.FCNHead`（in_channels=720, num_classes=4,
  use_sigmoid 口径的 4 通道 logits）

**明确绕开**（这些旧组件不进入新模型，见 docs/ASSET_AND_INTERFACE_AUDIT.md）：
- frozen reference 副本（`keep_frozen_b_copy: false`）；
- 旧多裁剪融合包装 `HRDecoder` segmentor（`_fuse_logits` / `hr_settings.crop_num`
  / SE 加权滑窗 / learnable 融合权重）；
- 旧 MA residual head（`detail_branch` + `detail_head` + `detail_scale`）。

学生前向：`image → HRNet(4 尺度特征) → FCNHead → 4 logits → 双线性上采样到输入分辨率`。
上采样只在"低分辨率 logits → 输入尺寸"这一步发生，属于解码器自身的恢复行为，
不是对输入图像 resize（输入尺寸约束由数据层 pad 保证）。

依赖缺失（Windows 上无 mmcv/mmengine，或旧仓库路径不存在）时抛
`trainer.models.DependencyBlocked`：**不允许用自造 HRNet 冒充旧仓库结构**。
"""

from __future__ import annotations

import os
import sys

from ..gate import GateError, deep_get, is_placeholder
from ..models import DependencyBlocked
from .weight_loading import load_pretrained_subset

NUM_CLASSES = 4  # EX/HE/SE/MA

# HRNetV2-W48 的 stage4 通道（与旧仓库 configs/_base_/models/hrdecoder_fcn_hr48.py 一致）
HRNET_W48_EXTRA = dict(
    stage1=dict(num_modules=1, num_branches=1, block="BOTTLENECK",
                num_blocks=(4,), num_channels=(64,)),
    stage2=dict(num_modules=1, num_branches=2, block="BASIC",
                num_blocks=(4, 4), num_channels=(48, 96)),
    stage3=dict(num_modules=4, num_branches=3, block="BASIC",
                num_blocks=(4, 4, 4), num_channels=(48, 96, 192)),
    stage4=dict(num_modules=3, num_branches=4, block="BASIC",
                num_blocks=(4, 4, 4, 4), num_channels=(48, 96, 192, 384)),
)
HRNET_W48_IN_CHANNELS = 48 + 96 + 192 + 384  # 720

# 旧仓库分类权重的非 backbone 键（backbone-only 加载时允许出现的 unexpected）
# OpenMMLab's official HRNetV2 ImageNet checkpoint contains the classification
# aggregation path in addition to the shared HRNet stem/stages.  These modules
# do not exist in the segmentation backbone and are the only expected extras.
DEFAULT_ALLOW_UNEXPECTED = (
    r"^classifier\.", r"^head\.", r"^fc\.", r"^last_layer\.",
    r"^incre_modules\.", r"^downsamp_modules\.", r"^final_layer\.",
)
DEFAULT_ALLOW_MISSING: tuple = ()


def probe_reference(reference_root: str) -> dict:
    """检查旧仓库能否在当前解释器里导入 HRNet/FCNHead（只读，不改动仓库）。

    先把 reference_root 插到 sys.path 首位，保证 `mmseg` 解析到的是**这份旧仓库**
    （而不是环境里可能存在的另一份 mmseg）。
    """
    report = {"reference_root": reference_root, "importable": False, "reason": None,
              "missing_modules": [], "hrnet_class": None, "sys_path_inserted": False}
    if not reference_root or not os.path.isdir(reference_root):
        report["reason"] = f"旧仓库根目录不存在: {reference_root!r}"
        return report
    if not os.path.isfile(os.path.join(reference_root, "mmseg", "models", "backbones", "hrnet.py")):
        report["reason"] = (
            f"{reference_root} 下缺少 mmseg/models/backbones/hrnet.py —— "
            "不是预期的 HRDecoder 旧仓库"
        )
        return report
    if reference_root not in sys.path:
        sys.path.insert(0, reference_root)
        report["sys_path_inserted"] = True
    for mod in ("torch", "mmcv", "mmengine", "mmseg"):
        try:
            __import__(mod)
        except Exception as e:  # noqa: BLE001 — 依赖探测需要区分"缺哪个"
            report["missing_modules"].append(f"{mod}: {type(e).__name__}: {e}")
    if report["missing_modules"]:
        report["reason"] = (
            "旧仓库依赖在本机不可导入（" + "; ".join(report["missing_modules"]) +
            "）。HRDecoder/HRNet 的结构必须来自旧仓库，禁止用自造实现冒充。"
        )
        return report
    try:
        from mmseg.models.backbones.hrnet import HRNet  # noqa: PLC0415
        report["hrnet_class"] = f"{HRNet.__module__}.{HRNet.__name__}"
        report["importable"] = True
    except Exception as e:  # noqa: BLE001
        report["reason"] = f"导入 mmseg.models.backbones.hrnet 失败: {type(e).__name__}: {e}"
    return report


def require_reference(reference_root: str) -> dict:
    rep = probe_reference(reference_root)
    if not rep["importable"]:
        raise DependencyBlocked(
            f"H 臂 adapter 依赖阻塞（DEPENDENCY_BLOCKED，不是 FAIL 也不是 PASS）："
            f"{rep['reason']}"
        )
    return rep


def build_reference_modules(reference_root: str):
    """导入旧仓库的 HRNet / FCNHead 类（导入失败 → DependencyBlocked）。"""
    require_reference(reference_root)
    if reference_root not in sys.path:
        sys.path.insert(0, reference_root)
    try:
        from mmseg.models.backbones.hrnet import HRNet  # noqa: PLC0415
        from mmseg.models.decode_heads.fcn_head import FCNHead  # noqa: PLC0415
    except Exception as e:  # noqa: BLE001
        raise DependencyBlocked(
            f"H 臂 adapter 依赖阻塞：旧仓库模块导入失败 {type(e).__name__}: {e}"
        ) from e
    return HRNet, FCNHead


class HRDecoderStudent:
    """单个学生模型：旧仓库 HRNet-W48 + FCNHead。

    以"组装函数 + 轻量容器"的形式给出，避免在本工作区固化旧仓库的类继承
    （那会把 mmcv 注册表机制拖进来）。容器本身是 `torch.nn.Module` 子类，
    只在依赖可用时动态创建。
    """

    @staticmethod
    def build(reference_root: str, *, norm_eval: bool):
        import torch.nn as nn

        HRNet, FCNHead = build_reference_modules(reference_root)

        backbone = HRNet(
            extra=dict(HRNET_W48_EXTRA),
            norm_cfg=dict(type="BN", requires_grad=True),
            norm_eval=bool(norm_eval),
        )
        decode_head = FCNHead(
            in_channels=HRNET_W48_IN_CHANNELS,
            in_index=0,
            channels=64,
            kernel_size=7,
            num_convs=1,
            concat_input=False,
            dropout_ratio=-1,
            num_classes=NUM_CLASSES,
            norm_cfg=dict(type="BN", requires_grad=True),
            align_corners=False,
        )

        class _Student(nn.Module):
            """image → backbone → decode_head → 上采样到输入尺寸的 4 通道 logits。"""

            def __init__(self):
                super().__init__()
                self.backbone = backbone
                self.decode_head = decode_head
                self.gradient_checkpointing = False

            def forward(self, image):
                h, w = image.shape[-2:]
                # The original HRDecoder uses ``hr_settings.divisible=8``.
                # Reject incompatible shapes explicitly instead of relying on
                # accidental branch rounding inside HRNet.
                if h % 8 or w % 8:
                    raise GateError(
                        f"HRDecoder 输入尺寸必须是 8 的倍数，收到 ({h},{w})"
                    )
                for block in self.backbone.modules():
                    if hasattr(block, 'with_cp'):
                        block.with_cp = bool(self.gradient_checkpointing)
                feats = self.backbone(image)
                # The original HRDecoder does not ask FCNHead to transform the
                # four HRNet branches.  Its segmentor first resizes every branch
                # to branch-0 resolution and concatenates 48+96+192+384=720
                # channels, then calls ``decode_head([fused])``.  Passing the raw
                # list here makes FCNHead select only in_index=0 (48 channels)
                # while its convolution expects 720 channels.
                import torch
                import torch.nn.functional as F  # noqa: PLC0415
                ref_size = feats[0].shape[-2:]
                fused = torch.cat([
                    feat if feat.shape[-2:] == ref_size else F.interpolate(
                        feat, size=ref_size, mode="bilinear", align_corners=False
                    )
                    for feat in feats
                ], dim=1)
                if fused.shape[1] != HRNET_W48_IN_CHANNELS:
                    raise RuntimeError(
                        f"HRNet 四尺度拼接后为 {fused.shape[1]} 通道，"
                        f"期望 {HRNET_W48_IN_CHANNELS}"
                    )
                logits = self.decode_head([fused])
                if logits.shape[1] != NUM_CLASSES:
                    raise RuntimeError(
                        f"decode_head 输出 {logits.shape[1]} 通道，期望 {NUM_CLASSES}"
                    )
                if logits.shape[-2:] != (h, w):
                    logits = F.interpolate(logits, size=(h, w), mode="bilinear",
                                           align_corners=False)
                if logits.shape[-2:] != (h, w):
                    raise RuntimeError(
                        f"学生输出 {tuple(logits.shape[-2:])} != 输入 ({h},{w})"
                    )
                return logits

            eval_forward = forward

        return _Student()


def _weight_acceptance(init_cfg: dict) -> tuple:
    """从配置读 missing/unexpected 接受白名单（缺省用保守默认，不是"全接受"）。"""
    iw = (init_cfg or {}).get("imagenet_weights") or {}
    allow_missing = tuple(iw.get("accept_missing_patterns") or DEFAULT_ALLOW_MISSING)
    allow_unexpected = tuple(
        iw.get("accept_unexpected_patterns") or DEFAULT_ALLOW_UNEXPECTED
    )
    return allow_missing, allow_unexpected


def load_imagenet_weights(model, *, init_cfg: dict, label: str = "imagenet_hrnet_w48") -> dict:
    """加载 ImageNet 权重到 backbone；missing/unexpected 全量报告，不静默部分加载。

    接受范围（白名单）来自配置字段：
      initialization.imagenet_weights.accept_missing_patterns
      initialization.imagenet_weights.accept_unexpected_patterns
    未声明时用保守默认（unexpected 只允许分类头，missing 一律不允许）。
    """
    iw = (init_cfg or {}).get("imagenet_weights") or {}
    path = iw.get("file")
    if is_placeholder(path) or not isinstance(path, str) or not path:
        raise GateError(f"{label}: initialization.imagenet_weights.file 未填写（{path!r}）")
    allow_missing, allow_unexpected = _weight_acceptance(init_cfg)
    report = load_pretrained_subset(
        model.backbone,
        ckpt_path=path,
        label=label,
        allow_missing=allow_missing,
        allow_unexpected=allow_unexpected,
        sha256_expected=iw.get("sha256"),
    )
    report["sha256_checked"] = bool(iw.get("sha256"))
    report["accepted_missing_patterns"] = list(allow_missing)
    report["accepted_unexpected_patterns"] = list(allow_unexpected)
    return report


def load_external_pretrain_weights(model, *, init_cfg: dict, label: str = "external_pretrain") -> dict:
    """加载外部病灶预训练 checkpoint 到完整学生（backbone + decode_head）。

    只允许加载本 trainer 产出的 checkpoint 或结构完全一致的整模权重：
    missing 默认必须为空（结构不一致就报错，不静默回退到部分加载）。
    """
    ext = (init_cfg or {}).get("external_pretrain_checkpoint")
    if ext is None:
        raise GateError(f"{label}: 该臂显式禁止外部预训练权重（external_pretrain_checkpoint=null）")
    if is_placeholder(ext) or not isinstance(ext, dict):
        raise GateError(f"{label}: external_pretrain_checkpoint 未填写为 {{file, sha256}} 映射")
    allow_missing = tuple(ext.get("accept_missing_patterns") or ())
    allow_unexpected = tuple(ext.get("accept_unexpected_patterns") or ())
    return load_pretrained_subset(
        model,
        ckpt_path=ext["file"],
        label=label,
        allow_missing=allow_missing,
        allow_unexpected=allow_unexpected,
        sha256_expected=ext.get("sha256"),
    )


def build_hr_decoder_student(cfg: dict, *, gradient_checkpointing=None):
    """trainer.models.build_model 的 H 臂入口。返回 (model, meta)。"""
    from ..config import require_reference_root  # 延迟导入避免循环

    st = cfg.get("structure") or {}
    if str(st.get("family")) != "hrdecoder":
        raise GateError(f"build_hr_decoder_student 收到 family={st.get('family')!r}")
    if bool(st.get("keep_frozen_b_copy")):
        raise GateError(
            "structure.keep_frozen_b_copy=true 会引入旧 frozen reference 副本；"
            "本实现明确不支持（H 臂只有单个学生）"
        )
    if not bool(st.get("open_all_stages", True)):
        raise GateError("structure.open_all_stages 必须为 true（backbone/transition/stem/decoder 全开放）")

    reference_root = require_reference_root(cfg)
    bn_policy = deep_get(cfg, "ddr_finetune_stage.bn_policy") or {}
    norm_eval = bool(bn_policy.get("backbone_norm_eval", True))
    if bool(bn_policy.get("affine_trainable", True)) is not True:
        raise GateError(
            "ddr_finetune_stage.bn_policy.affine_trainable 必须为 true；"
            "HRNet BN 只冻结统计量，affine 参数必须可训练"
        )

    model = HRDecoderStudent.build(reference_root, norm_eval=norm_eval)
    if gradient_checkpointing is not None:
        model.gradient_checkpointing = bool(gradient_checkpointing)

    weight_report = None
    iw = (cfg.get("initialization") or {}).get("imagenet_weights") or {}
    if iw.get("file") and not is_placeholder(iw.get("file")):
        weight_report = load_imagenet_weights(model, init_cfg=cfg["initialization"])

    meta = {
        "family": "hrdecoder",
        "backbone_prefixes": ("backbone",),
        "param_group_prefixes": ("backbone", "decode_head"),
        "backbone_norm_eval": norm_eval,
        "reference_root": reference_root,
        "weight_report": weight_report,
        "num_classes": NUM_CLASSES,
    }
    return model, meta
