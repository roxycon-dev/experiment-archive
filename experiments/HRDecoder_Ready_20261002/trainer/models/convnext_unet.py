# -*- coding: utf-8 -*-
"""trainer.models.convnext_unet — U 臂 builtin 实现（不依赖第三方分割库）。

构成（configs structure.family=convnext_unet）：
- 编码器：ConvNeXt-Small（facebookresearch 官方结构，逐字对齐其实现命名，
  使官方 1k-224 checkpoint 可直接映射加载；depths=[3,3,27,3]，dims=[96,192,384,768]）。
  LayerNorm 全程原生（norm_policy: layernorm_native，不套 BN 策略）。
- 解码器：四级 skip（stride 4/8/16/32）逐级上采样融合，直到 stride 1，
  1×1 卷积输出 4 通道 logits —— full_input_resolution_decode（不允许
  "低分辨率分类后大倍率插值"）。
- detail_branch：v1 不实现。configs U0/U1 已置 false；配置为 true 时本模块
  构建期即抛 GateError（不允许配置声称 true 而代码无实现）。
- 权重：官方 ConvNeXt checkpoint（weights_only 安全加载），unexpected 仅允许
  分类头 head.* 与最终 norm.*（编码器不使用）；missing 一律拒绝。

形状契约：输入 H/W 必须为 32 的倍数（否则抛错，绝不静默 resize）。

注意（权重键对齐，已修正）：官方 facebookresearch/ConvNeXt 的
`Block.dwconv` 就是一个 `nn.Conv2d`，键名是 `stages.i.j.dwconv.weight`。
旧实现把它包了一层 DWConv 容器，实际键变成 `stages.i.j.dwconv.dwconv.weight`，
与官方 checkpoint 完全对不上（会导致 dwconv 全部 missing）。此处改为直接使用
`nn.Conv2d`，与官方逐字一致；weight_loading 另设 alias 兜底旧键。
"""

from __future__ import annotations

import re

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from ..gate import GateError

CONVNEXT_S_DEPTH = (3, 3, 27, 3)
CONVNEXT_S_DIMS = (96, 192, 384, 768)
NUM_CLASSES = 4  # EX/HE/SE/MA（tools/contract_common.LESION_CHANNEL_ORDER）

# 官方 checkpoint 中编码器不使用的键（分类头/最终_norm），加载白名单
OFFICIAL_ALLOW_UNEXPECTED = (r"^head\.", r"^norm\.")
OFFICIAL_ALLOW_MISSING = ()


# ---------------------------------------------------------------------------
# 基础件（命名对齐 facebookresearch/ConvNeXt，保证官方权重键可直接映射）
# ---------------------------------------------------------------------------

class LayerNorm2d(nn.Module):
    """channels_first LayerNorm（对齐官方 LayerNorm data_format='channels_first'）。

    参数名 weight/bias 与官方一致；eps=1e-6。
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return x * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)


class DropPath(nn.Module):
    """随机深度（per-sample）；eval 恒等。"""

    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = float(p)

    def forward(self, x):
        if self.p <= 0.0 or not self.training:
            return x
        keep = 1.0 - self.p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep)
        return x * mask / keep


class ConvNeXtBlock(nn.Module):
    """官方 Block：dwconv → LN(NHWC) → pwconv1 → GELU → pwconv2 → gamma*scale。"""

    def __init__(self, dim: int, drop_path: float = 0.0, layer_scale_init: float = 1e-6):
        super().__init__()
        # 官方命名：self.dwconv 即 Conv2d ⇒ 键 stages.i.j.dwconv.weight
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init * torch.ones(dim), requires_grad=True) \
            if layer_scale_init > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)          # NCHW -> NHWC
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)          # NHWC -> NCHW
        return shortcut + self.drop_path(x)


class ConvNeXtEncoder(nn.Module):
    """ConvNeXt-S 编码器；输出四级特征（stride 4/8/16/32）。

    命名 downsample_layers.* / stages.* 与官方一致；官方分类 head.* /
    最终 norm.* 不构建（加载时按白名单剔除）。
    """

    def __init__(self, depths=CONVNEXT_S_DEPTH, dims=CONVNEXT_S_DIMS,
                 drop_path_rate: float = 0.1, in_chans: int = 3):
        super().__init__()
        if tuple(depths) != CONVNEXT_S_DEPTH or tuple(dims) != CONVNEXT_S_DIMS:
            raise GateError(
                f"v1 锁定 ConvNeXt-S（depths={CONVNEXT_S_DEPTH}, dims={CONVNEXT_S_DIMS}），"
                f"收到 depths={depths}, dims={dims}"
            )
        self.dims = tuple(dims)
        # stem：4×4 stride4 conv + LN
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm2d(dims[0], eps=1e-6),
        )
        self.downsample_layers.append(stem)
        for i in range(3):  # 3 个中间下采样：LN + 2×2 stride2 conv
            self.downsample_layers.append(nn.Sequential(
                LayerNorm2d(dims[i], eps=1e-6),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            ))
        # 各 stage 残差块
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.stages = nn.ModuleList()
        cur = 0
        for i in range(4):
            stage = nn.Sequential(*[
                ConvNeXtBlock(dim=dims[i], drop_path=float(dpr[cur + j]))
                for j in range(depths[i])
            ])
            self.stages.append(stage)
            cur += depths[i]
        self.gradient_checkpointing = False

    def forward(self, x: torch.Tensor) -> list:
        feats = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            if self.gradient_checkpointing and self.training and x.requires_grad:
                # checkpoint_sequential(..., segments=1) checkpoints nothing.
                for block in self.stages[i]:
                    x = checkpoint(block, x, use_reentrant=False)
            else:
                x = self.stages[i](x)
            feats.append(x)
        return feats  # [stride4, stride8, stride16, stride32]

    def load_official_weights(self, ckpt_path: str, sha256_expected: str | None = None,
                              allow_missing=None, allow_unexpected=None) -> dict:
        """官方 ConvNeXt checkpoint → 编码器（strict 策略 + 显式报告）。

        白名单可由配置覆盖（initialization.imagenet_weights.accept_*_patterns）；
        未提供时用本模块默认（unexpected 只放行分类头/最终 norm，missing 一律拒绝）。
        """
        from .weight_loading import load_pretrained_subset

        return load_pretrained_subset(
            self,
            ckpt_path=ckpt_path,
            label="convnext_encoder_official",
            allow_missing=OFFICIAL_ALLOW_MISSING if allow_missing is None else allow_missing,
            allow_unexpected=OFFICIAL_ALLOW_UNEXPECTED if allow_unexpected is None else allow_unexpected,
            sha256_expected=sha256_expected,
        )


# ---------------------------------------------------------------------------
# U-Net decoder（stride1 全分辨率输出）
# ---------------------------------------------------------------------------

class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class UNetDecoder(nn.Module):
    """四级 skip 逐级上采样到 stride 1，1×1 输出 4 logits。

    通道对齐（此前实现是错的，第一次真实前向就会 RuntimeError）：
    `feats = [stride4, stride8, stride16, stride32]`，通道分别是
    `dims = (96,192,384,768)`。旧代码写成 `e4,e3,e2,e1 = dims` 并按
    "e4=768" 的注释使用，实际得到 e4=96/e1=768，于是
    `c4 = ConvBlock(96, 256)` 却要吃 768 通道的 stride32 特征。
    这里改为按 **stride 索引** 显式取名，杜绝同类错配。
    """

    # 深→浅解码通道（stride32 → stride1）
    DEC_CH = (256, 128, 64, 32, 16)

    def __init__(self, dims=CONVNEXT_S_DIMS, num_classes: int = NUM_CLASSES,
                 dropout: float = 0.0):
        super().__init__()
        if len(dims) != 4:
            raise GateError(f"decoder 需要恰 4 级编码特征，收到 dims={dims}")
        d4, d3, d2, d1, d0 = self.DEC_CH
        e_s4, e_s8, e_s16, e_s32 = dims          # 96, 192, 384, 768
        self.c4 = ConvBlock(e_s32, d4)           # stride32 -> 1/32
        self.c3 = ConvBlock(d4 + e_s16, d3)      # 上采样 + stride16 skip
        self.c2 = ConvBlock(d3 + e_s8, d2)       # 上采样 + stride8 skip
        self.c1 = ConvBlock(d2 + e_s4, d1)       # 上采样 + stride4 skip
        self.c0 = ConvBlock(d1, d0)              # stride 2
        self.c00 = ConvBlock(d0, d0 // 2)        # stride 1
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.cls = nn.Conv2d(d0 // 2, num_classes, 1)

    def forward(self, feats: list) -> torch.Tensor:
        x0, x1, x2, x3 = feats  # stride 4, 8, 16, 32
        d4 = self.c4(x3)
        d3 = self.c3(torch.cat([_up(d4, x2.shape[2:]), x2], dim=1))
        d2 = self.c2(torch.cat([_up(d3, x1.shape[2:]), x1], dim=1))
        d1 = self.c1(torch.cat([_up(d2, x0.shape[2:]), x0], dim=1))
        d0 = self.c0(_up(d1, (x0.shape[2] * 2, x0.shape[3] * 2)))   # stride 2
        d00 = self.c00(_up(d0, (x0.shape[2] * 4, x0.shape[3] * 4)))  # stride 1
        return self.cls(self.dropout(d00))


def _up(x: torch.Tensor, size) -> torch.Tensor:
    return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


# ---------------------------------------------------------------------------
# 整臂模型
# ---------------------------------------------------------------------------

class ConvNeXtUNet(nn.Module):
    """ConvNeXt-S + U-Net decoder；forward(image) → [B,4,H,W]（与输入同尺寸）。"""

    def __init__(self, *, drop_path_rate: float = 0.1, decoder_dropout: float = 0.0,
                 detail_branch: bool = False):
        super().__init__()
        if detail_branch:
            raise GateError(
                "U 臂 detail_branch=true 在 v1 无实现：configs U0/U1 必须置 false"
                "（或先完成实现+shape test 再允许 true）"
            )
        self.encoder = ConvNeXtEncoder(drop_path_rate=drop_path_rate)
        self.decoder = UNetDecoder(dropout=decoder_dropout)

    @property
    def gradient_checkpointing(self) -> bool:
        return self.encoder.gradient_checkpointing

    @gradient_checkpointing.setter
    def gradient_checkpointing(self, flag: bool) -> None:
        self.encoder.gradient_checkpointing = bool(flag)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        h, w = image.shape[-2:]
        if h % 32 != 0 or w % 32 != 0:
            raise GateError(
                f"ConvNeXtUNet 输入需为 32 的倍数，当前 ({h},{w})；"
                "禁止静默 resize/pad——请由数据层显式 pad"
            )
        feats = self.encoder(image)
        logits = self.decoder(feats)
        if logits.shape[-2:] != (h, w):
            raise RuntimeError(
                f"decoder 输出 {tuple(logits.shape[-2:])} != 输入 ({h},{w})"
            )
        return logits

    def eval_forward(self, image: torch.Tensor) -> torch.Tensor:
        """整图推理：同 forward（单次前向，无滑窗；OOM 属校准问题，绝不 resize）。"""
        return self.forward(image)


def build_convnext_unet(cfg: dict, *, gradient_checkpointing: bool | None = None):
    """从 configs structure 节构建 U 臂；版本/权重来源完全由配置声明。"""
    st = cfg.get("structure") or {}
    library = str(st.get("library", "")).lower()
    if library not in ("builtin", "builtin_convnext_unet"):
        raise GateError(
            f"U 臂 library={st.get('library')!r}：v1 只实现 builtin（纯 torch，"
            "无 smp 依赖）；如需 smp 须先在服务器实测其编码器接口与版本锁定"
        )
    if str(st.get("decoder", "unet")).lower() != "unet":
        raise GateError(f"未知 decoder: {st.get('decoder')!r}")
    if int(st.get("output_heads", 4)) != 4:
        raise GateError("output_heads 必须为 4（EX/HE/SE/MA 独立 logits）")
    detail = bool(st.get("detail_branch", False))
    if str(st.get("norm_policy", "layernorm_native")) != "layernorm_native":
        raise GateError("U 臂 norm_policy 必须为 layernorm_native")

    model = ConvNeXtUNet(
        drop_path_rate=float(st.get("drop_path", 0.1)),
        decoder_dropout=float(st.get("decoder_dropout", 0.0)),
        detail_branch=detail,
    )
    if gradient_checkpointing is not None:
        model.gradient_checkpointing = bool(gradient_checkpointing)
    # 权重加载（若配置已给出具体文件）
    wf, ws = st.get("encoder_weights_file"), st.get("encoder_weights_sha256")
    weight_report = None
    if wf and not _is_placeholder(wf):
        iw = (cfg.get("initialization") or {}).get("imagenet_weights") or {}
        # 接受范围由配置决定；未声明时用模块默认（保守）
        am = iw.get("accept_missing_patterns")
        au = iw.get("accept_unexpected_patterns")
        weight_report = model.encoder.load_official_weights(
            wf, sha256_expected=ws,
            allow_missing=None if am is None else tuple(am),
            allow_unexpected=None if au is None else tuple(au),
        )
    meta = {
        "family": "convnext_unet",
        "param_group_prefixes": ("encoder.",),
        "backbone_prefixes": ("encoder.",),
        "backbone_norm_eval": False,   # ConvNeXt 无 BN；LayerNorm 原生
        "weight_report": weight_report,
        "norm_policy": "layernorm_native",
        "num_classes": NUM_CLASSES,
    }
    return model, meta


def _is_placeholder(v) -> bool:
    return isinstance(v, str) and ("REQUIRED" in v or "TBD" in v)


def state_dict_is_official_convnext(sd_keys) -> bool:
    """快速判断 state_dict 键是否官方格式（供工具诊断用）。"""
    joined = "\n".join(sd_keys)
    return bool(re.search(r"^downsample_layers\.0\.0\.weight$", joined, re.M))
