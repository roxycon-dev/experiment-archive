#!/usr/bin/env python
"""Pure-CPU structural checks for the v3 private-path prototype.

This is not a model reproduction: the shared feature provider is a random,
small shape stub. It loads no checkpoint, never evaluates a real image, and
does not construct an optimizer or update a parameter.
"""
from __future__ import annotations

import json
import math
import copy
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parent
REPORT_PATH = ROOT / "evidence" / "v3_module_checks.json"
SEED = 20261008
SMALL_HW = 128
CANVAS_HW = 2048


class ChannelLayerNorm2d(nn.Module):
    """LayerNorm over channels independently at each N,H,W location."""

    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=1e-5, elementwise_affine=True)

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm(x.permute(0, 2, 3, 1))
        return x.permute(0, 3, 1, 2).contiguous()


class Pointwise(nn.Module):
    """P(a,b): bias-free 1x1 convolution, channel LayerNorm, GELU."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.norm = ChannelLayerNorm2d(out_channels)
        self.act = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.norm(self.conv(x)))


class Down(nn.Module):
    """3x3 stride-2 bias-free convolution, channel LN, GELU."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, stride=2,
                              padding=1, bias=False)
        self.norm = ChannelLayerNorm2d(out_channels)
        self.act = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.norm(self.conv(x)))


class ResidualBlock(nn.Module):
    """B(c,k,e): DWConv, channel LN, expand/GELU/project, residual add."""

    def __init__(self, channels: int, kernel: int, expansion: int):
        super().__init__()
        self.dw = nn.Conv2d(channels, channels, kernel, padding=kernel // 2,
                            groups=channels, bias=False)
        self.norm = ChannelLayerNorm2d(channels)
        self.expand = nn.Conv2d(channels, expansion * channels, 1, bias=False)
        self.act = nn.GELU()
        self.project = nn.Conv2d(expansion * channels, channels, 1, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        y = self.norm(self.dw(x))
        y = self.act(self.expand(y))
        return x + self.project(y)


def resize_add(x: Tensor, skip: Tensor) -> Tensor:
    x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear",
                       align_corners=False)
    return x + skip


class LocalPath(nn.Module):
    """Private L path at the feature grids specified in v3 section 4.1."""

    def __init__(self):
        super().__init__()
        self.p_r = Pointwise(48, 32)
        self.l4_blocks = nn.Sequential(ResidualBlock(32, 3, 2),
                                       ResidualBlock(32, 3, 2))

        self.down8 = Down(32, 64)
        self.p_f1 = Pointwise(48, 64)
        self.l8_block = ResidualBlock(64, 7, 4)

        self.down16 = Down(64, 128)
        self.p_f2 = Pointwise(96, 128)
        self.l16_block = ResidualBlock(128, 7, 4)

        self.down32 = Down(128, 256)
        self.p_f3 = Pointwise(192, 256)
        self.l32_blocks = nn.Sequential(*[ResidualBlock(256, 7, 4)
                                          for _ in range(4)])

        self.up16_proj = Pointwise(256, 128)
        self.up16_block = ResidualBlock(128, 3, 2)
        self.up8_proj = Pointwise(128, 64)
        self.up8_block = ResidualBlock(64, 3, 2)
        self.up4_proj = Pointwise(64, 32)
        self.up4_block = ResidualBlock(32, 3, 2)

    def forward(self, r: Tensor, f1: Tensor, f2: Tensor,
                f3: Tensor) -> Tensor:
        l4 = self.l4_blocks(self.p_r(r))
        l8 = self.l8_block(self.down8(l4) + self.p_f1(f1))
        l16 = self.l16_block(self.down16(l8) + self.p_f2(f2))
        l32 = self.l32_blocks(self.down32(l16) + self.p_f3(f3))

        u16 = self.up16_block(resize_add(self.up16_proj(l32), l16))
        u8 = self.up8_block(resize_add(self.up8_proj(u16), l8))
        return self.up4_block(resize_add(self.up4_proj(u8), l4))


def sinusoidal_2d(height: int, width: int, *, device: torch.device,
                  dtype: torch.dtype) -> Tensor:
    """Fixed x/y sin-cos encoding, 128 dimensions per axis."""
    i = torch.arange(64, device=device, dtype=torch.float32)
    frequencies = torch.pow(10000.0, -2.0 * i / 128.0)
    y = torch.arange(height, device=device, dtype=torch.float32)
    x = torch.arange(width, device=device, dtype=torch.float32)
    yv, xv = torch.meshgrid(y, x, indexing="ij")
    ax = xv.unsqueeze(-1) * frequencies
    ay = yv.unsqueeze(-1) * frequencies
    # 128 channels for x and 128 for y: interleaved sine/cosine pairs.
    ex = torch.stack((torch.sin(ax), torch.cos(ax)), dim=-1).flatten(-2)
    ey = torch.stack((torch.sin(ay), torch.cos(ay)), dim=-1).flatten(-2)
    return torch.cat((ex, ey), dim=-1).permute(2, 0, 1).unsqueeze(0).to(dtype)


class PreLNTransformerBlock(nn.Module):
    """Ordinary global self-attention block, with all linear biases disabled."""

    def __init__(self, dim: int = 256, heads: int = 8, mlp_ratio: int = 4):
        super().__init__()
        if dim % heads:
            raise ValueError("dim must be divisible by heads")
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.norm1 = nn.LayerNorm(dim, eps=1e-5, elementwise_affine=True)
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.attn_proj = nn.Linear(dim, dim, bias=False)
        self.norm2 = nn.LayerNorm(dim, eps=1e-5, elementwise_affine=True)
        self.mlp1 = nn.Linear(dim, mlp_ratio * dim, bias=False)
        self.act = nn.GELU()
        self.mlp2 = nn.Linear(mlp_ratio * dim, dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        batch, tokens, dim = x.shape
        qkv = self.qkv(self.norm1(x))
        qkv = qkv.view(batch, tokens, 3, self.heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        weights = torch.softmax(
            torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim),
            dim=-1)
        attended = torch.matmul(weights, v)
        attended = attended.transpose(1, 2).reshape(batch, tokens, dim)
        x = x + self.attn_proj(attended)
        return x + self.mlp2(self.act(self.mlp1(self.norm2(x))))


class GlobalPath(nn.Module):
    """Private G path with a 32x32, four-block transformer bottleneck."""

    def __init__(self):
        super().__init__()
        self.p_f34 = Pointwise(576, 256)
        self.p_r32 = Pointwise(48, 256)
        self.blocks = nn.ModuleList([PreLNTransformerBlock() for _ in range(4)])

        self.p_32_16 = Pointwise(256, 128)
        self.p_f2 = Pointwise(96, 128)
        self.block16 = ResidualBlock(128, 3, 2)

        self.p_16_8 = Pointwise(128, 64)
        self.p_f1 = Pointwise(48, 64)
        self.block8 = ResidualBlock(64, 3, 2)

        self.p_8_4 = Pointwise(64, 32)
        self.p_r4 = Pointwise(48, 32)
        self.block4 = ResidualBlock(32, 3, 2)

    def forward(self, r: Tensor, f1: Tensor, f2: Tensor,
                f3: Tensor, f4: Tensor) -> Tensor:
        grid = f4.shape[-2:]
        f3_low = F.adaptive_avg_pool2d(f3, output_size=grid)
        r_low = F.adaptive_avg_pool2d(r, output_size=grid)
        x = self.p_f34(torch.cat((f3_low, f4), dim=1)) + self.p_r32(r_low)
        x = x + sinusoidal_2d(*grid, device=x.device, dtype=x.dtype)
        b, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        for block in self.blocks:
            tokens = block(tokens)
        x = tokens.transpose(1, 2).reshape(b, c, h, w)

        x = self.p_32_16(x)
        x = F.interpolate(x, size=f2.shape[-2:], mode="bilinear",
                          align_corners=False) + self.p_f2(f2)
        x = self.block16(x)

        x = self.p_16_8(x)
        x = F.interpolate(x, size=f1.shape[-2:], mode="bilinear",
                          align_corners=False) + self.p_f1(f1)
        x = self.block8(x)

        x = self.p_8_4(x)
        x = F.interpolate(x, size=r.shape[-2:], mode="bilinear",
                          align_corners=False) + self.p_r4(r)
        return self.block4(x)


class ContrastReadout(nn.Module):
    """Nonzero-init 32->64 readout; shuffle to signed probability-difference maps."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(32, 64, 1, bias=True)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        coarse = self.conv(x)
        return coarse, F.pixel_shuffle(coarse, 4)


class FusionClassifier(nn.Module):
    """Single classifier; only its final 64->64 convolution is zero-init."""

    def __init__(self):
        super().__init__()
        self.project = Pointwise(240, 64)
        self.block = ResidualBlock(64, 3, 2)
        self.final_conv = nn.Conv2d(64, 64, 1, bias=True)
        nn.init.zeros_(self.final_conv.weight)
        nn.init.zeros_(self.final_conv.bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.block(self.project(x))
        return F.pixel_shuffle(self.final_conv(x), 4)


class V3PrivateModules(nn.Module):
    """New modules only; HRNet/HRDecoder remain an external shared provider."""

    def __init__(self):
        super().__init__()
        self.local = LocalPath()
        self.global_path = GlobalPath()
        self.contrast_u = ContrastReadout()
        self.contrast_s = ContrastReadout()
        self.fusion = FusionClassifier()
        self.pixel_unshuffle = nn.PixelUnshuffle(4)

    def forward(self, rgb: Tensor, features: Sequence[Tensor],
                z0: Tensor) -> Dict[str, Tensor]:
        if len(features) != 4:
            raise ValueError("features must be (F1,F2,F3,F4)")
        h, w = rgb.shape[-2:]
        if h % 64 or w % 64:
            raise ValueError("canvas dimensions must be divisible by 64")
        f1, f2, f3, f4 = features
        expected = ((48, h // 8, w // 8), (96, h // 16, w // 16),
                    (192, h // 32, w // 32), (384, h // 64, w // 64))
        for index, (feature, shape) in enumerate(zip(features, expected), 1):
            if tuple(feature.shape[1:]) != shape:
                raise ValueError(f"F{index} expected {shape}, got {tuple(feature.shape[1:])}")
        if tuple(z0.shape) != (rgb.shape[0], 4, h, w):
            raise ValueError("z0 must be full-grid four-class logits")

        r = self.pixel_unshuffle(rgb)
        h_l = self.local(r, f1, f2, f3)
        h_g = self.global_path(r, f1, f2, f3, f4)
        b_u, d_hat_u = self.contrast_u(h_l)
        b_s, d_hat_s = self.contrast_s(h_g)
        classifier_features = torch.cat((h_l, h_g, b_u, b_s, r), dim=1)
        residual = self.fusion(classifier_features)
        return {
            "r": r,
            "h_l": h_l,
            "h_g": h_g,
            "b_u": b_u,
            "b_s": b_s,
            "d_hat_u": d_hat_u,
            "d_hat_s": d_hat_s,
            "residual": residual,
            "logits": z0 + residual,
        }


class RandomSharedFeatureStub(nn.Module):
    """Shape-compatible random provider; not an HRNet implementation."""

    def __init__(self):
        super().__init__()
        self.calls = 0
        self.f1_layer = nn.Conv2d(3, 48, 3, stride=8, padding=1, bias=False)
        self.f2_layer = nn.Conv2d(48, 96, 3, stride=2, padding=1, bias=False)
        self.f3_layer = nn.Conv2d(96, 192, 3, stride=2, padding=1, bias=False)
        self.f4_layer = nn.Conv2d(192, 384, 3, stride=2, padding=1, bias=False)
        self.base_head = nn.Conv2d(48, 4, 1, bias=True)

    def forward(self, rgb: Tensor) -> Tuple[Tuple[Tensor, ...], Tensor]:
        self.calls += 1
        f1 = F.gelu(self.f1_layer(rgb))
        f2 = F.gelu(self.f2_layer(f1))
        f3 = F.gelu(self.f3_layer(f2))
        f4 = F.gelu(self.f4_layer(f3))
        z0 = F.interpolate(self.base_head(f1), size=rgb.shape[-2:],
                           mode="bilinear", align_corners=False)
        return (f1, f2, f3, f4), z0


def count_parameters(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def p_params(a: int, b: int) -> int:
    return a * b + 2 * b


def down_params(a: int, b: int) -> int:
    return 9 * a * b + 2 * b


def block_params(c: int, k: int, e: int) -> int:
    return c * k * k + 2 * c + 2 * e * c * c


def expected_parameter_counts() -> Dict[str, int]:
    local = (
        p_params(48, 32) + 2 * block_params(32, 3, 2)
        + down_params(32, 64) + p_params(48, 64) + block_params(64, 7, 4)
        + down_params(64, 128) + p_params(96, 128) + block_params(128, 7, 4)
        + down_params(128, 256) + p_params(192, 256) + 4 * block_params(256, 7, 4)
        + p_params(256, 128) + block_params(128, 3, 2)
        + p_params(128, 64) + block_params(64, 3, 2)
        + p_params(64, 32) + block_params(32, 3, 2)
    )
    transformer = 256 * 768 + 256 * 256 + 256 * 1024 + 1024 * 256 + 4 * 256
    global_path = (
        p_params(576, 256) + p_params(48, 256) + 4 * transformer
        + p_params(256, 128) + p_params(96, 128) + block_params(128, 3, 2)
        + p_params(128, 64) + p_params(48, 64) + block_params(64, 3, 2)
        + p_params(64, 32) + p_params(48, 32) + block_params(32, 3, 2)
    )
    contrast = 2 * (32 * 64 + 64)
    fusion = p_params(240, 64) + block_params(64, 3, 2) + 64 * 64 + 64
    return {"L": local, "G": global_path, "fusion": fusion,
            "contrast": contrast}


class MacLedger:
    """Analytic forward MAC ledger for a specified full-canvas shape."""

    def __init__(self):
        self.groups: Dict[str, int] = {"L": 0, "G": 0, "contrast": 0,
                                      "fusion": 0}
        self.entries: List[dict] = []
        self.attention_qk = 0
        self.attention_av = 0
        self.attention_entries: List[dict] = []

    def conv(self, group: str, name: str, cin: int, cout: int,
             kernel: int, h: int, w: int, groups: int = 1) -> None:
        mac = h * w * cout * (cin // groups) * kernel * kernel
        self.groups[group] += mac
        self.entries.append({"group": group, "op": name, "mac": mac,
                             "grid": [h, w], "in": cin, "out": cout,
                             "kernel": kernel, "groups": groups})

    def linear(self, group: str, name: str, cin: int, cout: int,
               positions: int) -> None:
        mac = positions * cin * cout
        self.groups[group] += mac
        self.entries.append({"group": group, "op": name, "mac": mac,
                             "positions": positions, "in": cin, "out": cout})

    def block(self, group: str, name: str, c: int, k: int, e: int,
              h: int, w: int) -> None:
        self.conv(group, name + ".dw", c, c, k, h, w, groups=c)
        self.conv(group, name + ".expand", c, e * c, 1, h, w)
        self.conv(group, name + ".project", e * c, c, 1, h, w)

    def point(self, group: str, name: str, a: int, b: int,
              h: int, w: int) -> None:
        self.conv(group, name, a, b, 1, h, w)

    def attention(self, block_name: str, tokens: int = 32 * 32,
                  heads: int = 8, head_dim: int = 32) -> None:
        one = heads * tokens * tokens * head_dim
        self.attention_qk += one
        self.attention_av += one
        self.attention_entries.append({"block": block_name, "tokens": tokens,
                                       "heads": heads, "head_dim": head_dim,
                                       "qk_mac": one, "av_mac": one})


def build_mac_ledger() -> MacLedger:
    ledger = MacLedger()
    # L, for input canvas 2048: R512, F1=256, F2=128, F3=64.
    ledger.point("L", "L.P_R", 48, 32, 512, 512)
    for i in range(2):
        ledger.block("L", f"L.L4.B{i+1}", 32, 3, 2, 512, 512)
    ledger.conv("L", "L.Down8", 32, 64, 3, 256, 256)
    ledger.point("L", "L.P_F1", 48, 64, 256, 256)
    ledger.block("L", "L.L8.B1", 64, 7, 4, 256, 256)
    ledger.conv("L", "L.Down16", 64, 128, 3, 128, 128)
    ledger.point("L", "L.P_F2", 96, 128, 128, 128)
    ledger.block("L", "L.L16.B1", 128, 7, 4, 128, 128)
    ledger.conv("L", "L.Down32", 128, 256, 3, 64, 64)
    ledger.point("L", "L.P_F3", 192, 256, 64, 64)
    for i in range(4):
        ledger.block("L", f"L.L32.B{i+1}", 256, 7, 4, 64, 64)
    ledger.point("L", "L.Up16.P", 256, 128, 64, 64)
    ledger.block("L", "L.Up16.B", 128, 3, 2, 128, 128)
    ledger.point("L", "L.Up8.P", 128, 64, 128, 128)
    ledger.block("L", "L.Up8.B", 64, 3, 2, 256, 256)
    ledger.point("L", "L.Up4.P", 64, 32, 256, 256)
    ledger.block("L", "L.Up4.B", 32, 3, 2, 512, 512)

    # G projections and transformer. Attention matrix terms are counted below
    # from exact 32x32 tokens, never scaled from the 128 CPU test canvas.
    ledger.point("G", "G.P_F34", 576, 256, 32, 32)
    ledger.point("G", "G.P_R32", 48, 256, 32, 32)
    tokens = 32 * 32
    for i in range(4):
        name = f"G.Transformer{i+1}"
        ledger.linear("G", name + ".QKV", 256, 3 * 256, tokens)
        ledger.linear("G", name + ".attention_output", 256, 256, tokens)
        ledger.linear("G", name + ".MLP1", 256, 1024, tokens)
        ledger.linear("G", name + ".MLP2", 1024, 256, tokens)
        ledger.attention(name, tokens=tokens, heads=8, head_dim=32)
    ledger.point("G", "G.P_32_16", 256, 128, 32, 32)
    ledger.point("G", "G.P_F2", 96, 128, 128, 128)
    ledger.block("G", "G.B16", 128, 3, 2, 128, 128)
    ledger.point("G", "G.P_16_8", 128, 64, 128, 128)
    ledger.point("G", "G.P_F1", 48, 64, 256, 256)
    ledger.block("G", "G.B8", 64, 3, 2, 256, 256)
    ledger.point("G", "G.P_8_4", 64, 32, 256, 256)
    ledger.point("G", "G.P_R4", 48, 32, 512, 512)
    ledger.block("G", "G.B4", 32, 3, 2, 512, 512)

    # The two signed readouts run on 512^2 before PixelShuffle4.
    for name in ("contrast.dU", "contrast.dS"):
        ledger.conv("contrast", name, 32, 64, 1, 512, 512)

    ledger.point("fusion", "fusion.P_240_64", 240, 64, 512, 512)
    ledger.block("fusion", "fusion.B64", 64, 3, 2, 512, 512)
    ledger.conv("fusion", "fusion.final_64_64", 64, 64, 1, 512, 512)
    return ledger


def grad_norm(grad: Tensor | None) -> float:
    if grad is None:
        return 0.0
    return float(torch.linalg.vector_norm(grad.detach()).item())


def max_abs_difference(a: Tensor | None, b: Tensor | None) -> float:
    if a is None and b is None:
        return 0.0
    if a is None or b is None:
        return float("inf")
    return float((a.detach() - b.detach()).abs().max().item())


def aggregate_grad_norm(grads: Iterable[Tensor | None]) -> float:
    squared = 0.0
    for grad in grads:
        if grad is not None:
            squared += float(grad.detach().float().pow(2).sum().item())
    return math.sqrt(squared)


def run_checks() -> dict:
    torch.set_num_threads(1)
    torch.manual_seed(SEED)

    addons = V3PrivateModules().cpu()
    provider = RandomSharedFeatureStub().cpu()
    rgb = torch.randn(1, 3, SMALL_HW, SMALL_HW)
    # Count actual Conv/Linear calls on the small CPU shape. The ratio to
    # 2048 is exactly area-scaled for these operators; attention QK/AV below
    # is separately evaluated at its exact 1024-token grid.
    small_hook_macs: Dict[str, int] = {"L": 0, "G": 0, "fusion": 0,
                                       "contrast": 0}
    handles = []

    def group_for_name(name: str) -> str:
        if name.startswith("local."):
            return "L"
        if name.startswith("global_path."):
            return "G"
        if name.startswith("contrast_"):
            return "contrast"
        if name.startswith("fusion."):
            return "fusion"
        raise AssertionError(f"unrecognized v3 parameterized operator: {name}")

    for name, module in addons.named_modules():
        if isinstance(module, nn.Conv2d):
            group = group_for_name(name)

            def conv_hook(_module: nn.Module, _inputs: Tuple[Tensor, ...],
                          out: Tensor, group: str = group) -> None:
                conv = _module
                assert isinstance(conv, nn.Conv2d)
                kh, kw = conv.kernel_size
                cin = conv.in_channels // conv.groups
                mac = out.numel() * cin * kh * kw
                small_hook_macs[group] += int(mac)

            handles.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, nn.Linear):
            group = group_for_name(name)

            def linear_hook(_module: nn.Module, _inputs: Tuple[Tensor, ...],
                            out: Tensor, group: str = group) -> None:
                linear = _module
                assert isinstance(linear, nn.Linear)
                mac = (out.numel() // linear.out_features) * linear.in_features * linear.out_features
                small_hook_macs[group] += int(mac)

            handles.append(module.register_forward_hook(linear_hook))

    try:
        # Exactly one call across the shared feature interface.
        features, z0 = provider(rgb)
        output = addons(rgb, features, z0)
    finally:
        for handle in handles:
            handle.remove()

    actual_params = {
        "L": count_parameters(addons.local),
        "G": count_parameters(addons.global_path),
        "fusion": count_parameters(addons.fusion),
        "contrast": count_parameters(addons.contrast_u)
                    + count_parameters(addons.contrast_s),
    }
    expected_params = expected_parameter_counts()
    if actual_params != expected_params:
        raise AssertionError(f"parameter count mismatch: {actual_params} != {expected_params}")

    expected_shapes = {
        "R": [1, 48, 32, 32],
        "F1": [1, 48, 16, 16],
        "F2": [1, 96, 8, 8],
        "F3": [1, 192, 4, 4],
        "F4": [1, 384, 2, 2],
        "H_L": [1, 32, 32, 32],
        "H_G": [1, 32, 32, 32],
        "B_U": [1, 64, 32, 32],
        "B_S": [1, 64, 32, 32],
        "d_hat_U": [1, 4, SMALL_HW, SMALL_HW],
        "d_hat_S": [1, 4, SMALL_HW, SMALL_HW],
        "residual": [1, 4, SMALL_HW, SMALL_HW],
        "logits": [1, 4, SMALL_HW, SMALL_HW],
    }
    actual_shapes = {
        "R": list(output["r"].shape),
        "F1": list(features[0].shape),
        "F2": list(features[1].shape),
        "F3": list(features[2].shape),
        "F4": list(features[3].shape),
        "H_L": list(output["h_l"].shape),
        "H_G": list(output["h_g"].shape),
        "B_U": list(output["b_u"].shape),
        "B_S": list(output["b_s"].shape),
        "d_hat_U": list(output["d_hat_u"].shape),
        "d_hat_S": list(output["d_hat_s"].shape),
        "residual": list(output["residual"].shape),
        "logits": list(output["logits"].shape),
    }
    if actual_shapes != expected_shapes:
        raise AssertionError(f"small-canvas shape mismatch: {actual_shapes} != {expected_shapes}")
    if not torch.equal(output["residual"], torch.zeros_like(output["residual"])):
        raise AssertionError("zero-initialized final classifier did not emit zero residual")
    if not torch.equal(output["logits"], z0):
        raise AssertionError("zero residual is not identity relative to z0")

    # A contrast-only probe must pass a nonzero first-step gradient into H_L.
    target_du = torch.randn_like(output["d_hat_u"])
    contrast_loss = F.smooth_l1_loss(output["d_hat_u"], target_du, beta=0.1)
    grad_hl, grad_contrast_weight = torch.autograd.grad(
        contrast_loss, (output["h_l"], addons.contrast_u.conv.weight),
        retain_graph=True)
    if grad_norm(grad_hl) <= 1e-8 or grad_norm(grad_contrast_weight) <= 1e-8:
        raise AssertionError("contrast first-step hidden/head gradient is zero")

    # The zero final classifier gets a first-step weight gradient, while its
    # upstream projection is initially shielded by that zero output layer.
    gt = torch.randint(0, 2, output["logits"].shape).float()
    task_loss = F.binary_cross_entropy_with_logits(output["logits"], gt)
    grad_final_weight, grad_fusion_project = torch.autograd.grad(
        task_loss,
        (addons.fusion.final_conv.weight, addons.fusion.project.conv.weight),
        retain_graph=True)
    if grad_norm(grad_final_weight) <= 1e-8:
        raise AssertionError("classifier final layer receives no first-step gradient")
    if grad_norm(grad_fusion_project) != 0.0:
        raise AssertionError("zero classifier output layer should block first-step gradient upstream")

    # PixelUnshuffle4/PixelShuffle4 are inverse permutations.
    image = torch.randn(1, 3, 12, 20)
    roundtrip = F.pixel_shuffle(F.pixel_unshuffle(image, 4), 4)
    if not torch.equal(roundtrip, image):
        raise AssertionError("PixelUnshuffle4/PixelShuffle4 roundtrip failed")

    # The 64-channel final classifier bias can create a 4x4 alternating
    # subpixel pattern for one class from a single 1x1 coarse location.
    checker = torch.tensor([[0., 1., 0., 1.], [1., 0., 1., 0.],
                            [0., 1., 0., 1.], [1., 0., 1., 0.]])
    old_bias = addons.fusion.final_conv.bias.detach().clone()
    pattern_bias = torch.zeros_like(old_bias)
    pattern_bias[:16] = checker.reshape(-1)
    with torch.no_grad():
        addons.fusion.final_conv.bias.copy_(pattern_bias)
    coarse_pattern = addons.fusion.final_conv(torch.zeros(1, 64, 1, 1))
    pixel_pattern = F.pixel_shuffle(coarse_pattern, 4)[0, 0]
    checker_represented = torch.equal(pixel_pattern, checker)
    with torch.no_grad():
        addons.fusion.final_conv.bias.copy_(old_bias)
    if not checker_represented:
        raise AssertionError("PixelShuffle classifier failed subpixel checker test")

    # Section 6 route diagnostic. Use a separate copy with a seeded nonzero
    # final classifier, so task gradients can traverse the private paths. The
    # base-logit path is detached to isolate gradients through shared features.
    diag_addons = copy.deepcopy(addons).cpu()
    torch.manual_seed(SEED + 1)
    with torch.no_grad():
        diag_addons.fusion.final_conv.weight.copy_(
            0.02 * torch.randn_like(diag_addons.fusion.final_conv.weight))
        diag_addons.fusion.final_conv.bias.copy_(
            0.02 * torch.randn_like(diag_addons.fusion.final_conv.bias))
    diag_final_weight_norm = float(
        torch.linalg.vector_norm(diag_addons.fusion.final_conv.weight).item())
    diag_z0 = z0.detach()
    diag_output = diag_addons(rgb, features, diag_z0)
    diag_gt = torch.randint(0, 2, diag_output["logits"].shape).float()
    diag_task_loss = F.binary_cross_entropy_with_logits(diag_output["logits"], diag_gt)
    diag_target_du = torch.randn_like(diag_output["d_hat_u"])
    diag_target_ds = torch.randn_like(diag_output["d_hat_s"])
    diag_aux_loss = 0.5 * (
        F.smooth_l1_loss(diag_output["d_hat_u"], diag_target_du, beta=0.1)
        + F.smooth_l1_loss(diag_output["d_hat_s"], diag_target_ds, beta=0.1))
    lambda_d = 0.1  # synthetic scale for route mechanics only

    main_params = tuple(provider.parameters())
    feature_main_params = tuple(
        parameter for name, parameter in provider.named_parameters()
        if not name.startswith("base_head."))
    private_params = (tuple(diag_addons.local.parameters())
                      + tuple(diag_addons.global_path.parameters())
                      + tuple(diag_addons.contrast_u.parameters())
                      + tuple(diag_addons.contrast_s.parameters()))
    classifier_params = tuple(diag_addons.fusion.parameters())
    all_params = main_params + private_params + classifier_params
    diagnostic_assignment_names = [
        name for (name, original), (_, diagnostic) in zip(
            addons.named_parameters(), diag_addons.named_parameters())
        if not torch.equal(original.detach(), diagnostic.detach())]
    if set(diagnostic_assignment_names) != {
            "fusion.final_conv.weight", "fusion.final_conv.bias"}:
        raise AssertionError(f"unexpected diagnostic parameter assignments: {diagnostic_assignment_names}")

    # Prove task loss uses private L/G to reach the shared feature tensors and
    # their generating parameters even when the z0 shortcut is detached.
    task_feature_grads = torch.autograd.grad(
        diag_task_loss, features, retain_graph=True, allow_unused=True)
    task_shared_parameter_grads = torch.autograd.grad(
        diag_task_loss, feature_main_params, retain_graph=True, allow_unused=True)
    task_feature_activation_norm = aggregate_grad_norm(task_feature_grads)
    task_shared_parameter_norm = aggregate_grad_norm(task_shared_parameter_grads)
    if task_feature_activation_norm <= 1e-8 or task_shared_parameter_norm <= 1e-8:
        raise AssertionError("task loss did not reach shared features/parameters through private paths")

    # Without routing, the normal auxiliary graph also reaches shared theta.
    # autograd.grad inspects this path without populating .grad or updating it.
    raw_aux_main_grads = torch.autograd.grad(
        lambda_d * diag_aux_loss, feature_main_params,
        retain_graph=True, allow_unused=True)
    raw_aux_main_norm = aggregate_grad_norm(raw_aux_main_grads)
    if raw_aux_main_norm <= 1e-8:
        raise AssertionError("unrouted auxiliary graph has no shared-parameter gradient to exclude")

    # Routed references: task gradients for all groups, auxiliary gradients
    # only for private psi_L/psi_G and signed readouts. No optimizer is used.
    param_snapshot = [p.detach().clone() for p in all_params]
    task_grads = torch.autograd.grad(diag_task_loss, all_params, retain_graph=True,
                                     allow_unused=True)
    aux_private_grads = torch.autograd.grad(lambda_d * diag_aux_loss, private_params,
                                            retain_graph=True, allow_unused=True)
    if any(parameter.grad is not None for parameter in all_params):
        raise AssertionError("unexpected preexisting .grad before routed backward")
    diag_task_loss.backward()
    for parameter, aux_grad in zip(private_params, aux_private_grads):
        if aux_grad is not None:
            if parameter.grad is None:
                parameter.grad = aux_grad.detach().clone()
            else:
                parameter.grad.add_(aux_grad.detach())

    n_main = len(main_params)
    n_private = len(private_params)
    task_main = task_grads[:n_main]
    task_private = task_grads[n_main:n_main + n_private]
    task_classifier = task_grads[n_main + n_private:]
    main_route_diff = max(max_abs_difference(p.grad, g)
                          for p, g in zip(main_params, task_main))
    classifier_route_diff = max(max_abs_difference(p.grad, g)
                                for p, g in zip(classifier_params, task_classifier))
    private_route_diff = max(
        max_abs_difference(p.grad,
                           None if task_grad is None and aux_grad is None else
                           (torch.zeros_like(p) if task_grad is None else task_grad)
                           + (torch.zeros_like(p) if aux_grad is None else aux_grad))
        for p, task_grad, aux_grad in zip(private_params, task_private,
                                          aux_private_grads)
    )
    task_private_norm = aggregate_grad_norm(task_private)
    aux_private_norm = aggregate_grad_norm(aux_private_grads)
    private_combined_norm = aggregate_grad_norm(p.grad for p in private_params)
    if task_private_norm <= 1e-8 or aux_private_norm <= 1e-8:
        raise AssertionError("task or auxiliary component of private gradient is zero")
    if main_route_diff > 1e-6 or classifier_route_diff > 1e-6 or private_route_diff > 1e-6:
        raise AssertionError("section 6 routed gradient differs from reference")
    params_unchanged = all(torch.equal(p.detach(), before)
                           for p, before in zip(all_params, param_snapshot))
    original_zero_init_preserved = (
        torch.count_nonzero(addons.fusion.final_conv.weight).item() == 0
        and torch.count_nonzero(addons.fusion.final_conv.bias).item() == 0)
    if not params_unchanged or not original_zero_init_preserved:
        raise AssertionError("diagnostic assignment changed parameters or original zero-init")

    ledger = build_mac_ledger()
    area_scale_to_2048 = (CANVAS_HW // SMALL_HW) ** 2
    scaled_hook_macs = {group: mac * area_scale_to_2048
                        for group, mac in small_hook_macs.items()}
    if scaled_hook_macs != ledger.groups:
        raise AssertionError(
            f"small Conv/Linear hooks scaled by {area_scale_to_2048} disagree "
            f"with the 2048 shape ledger: {scaled_hook_macs} != {ledger.groups}")
    analytic_mac = {
        "conv_linear_by_group": ledger.groups,
        "conv_linear_total": sum(ledger.groups.values()),
        "small_128_conv_linear_by_group_from_hooks": small_hook_macs,
        "small_hook_area_scale_to_2048": area_scale_to_2048,
        "scaled_small_hooks_match_analytic": True,
        "attention_qk_mac": ledger.attention_qk,
        "attention_av_mac": ledger.attention_av,
        "attention_qk_plus_av_mac": ledger.attention_qk + ledger.attention_av,
        "all_new_forward_mac": (sum(ledger.groups.values())
                                 + ledger.attention_qk + ledger.attention_av),
        "attention_details": ledger.attention_entries,
        "conv_linear_entry_count": len(ledger.entries),
        "notes": [
            "Counts are analytic for 2048 canvas feature grids; no 2048 forward was run.",
            "QK and AV use 1024 tokens per block and are counted as N^2*embed_dim each.",
            "Bias adds, LN, GELU, softmax, positional encoding, interpolation, Pixel(Un)Shuffle and elementwise adds are excluded.",
            "Original HRNet/HRDecoder MAC and backward MAC are excluded.",
        ],
    }
    if ledger.attention_qk != 4 * (32 * 32) ** 2 * 256:
        raise AssertionError("QK attention MAC formula mismatch")
    if ledger.attention_av != ledger.attention_qk:
        raise AssertionError("AV attention MAC formula mismatch")

    return {
        "schema": "hrdecoder-v3-module-checks/v1",
        "status": "PASS",
        "scope": {
            "cpu_only": True,
            "real_hrnet_weights_loaded": False,
            "real_model_inference": False,
            "optimizer_created_or_step": False,
            "canvas_2048_forward_run": False,
            "prototype_note": "The feature provider is a random shape stub, not HRNet; all reported structural checks use random tensors.",
            "design_source": "05_v3_机制重构与反驳审查.md sections 4 and 6",
        },
        "small_canvas": {"input_hw": [SMALL_HW, SMALL_HW],
                         "shapes": actual_shapes,
                         "shared_feature_provider_calls": provider.calls,
                         "shared_once_interface_pass": provider.calls == 1,
                         "private_module_has_no_internal_backbone": not hasattr(addons, "backbone")},
        "parameters": {"actual": actual_params, "analytic_expected": expected_params,
                       "sum_new_parameters": sum(actual_params.values()),
                       "contrast_each_head": count_parameters(addons.contrast_u),
                       "groups": {
                           "L": "private local encoder/decoder; excludes signed readout",
                           "G": "private global encoder/decoder and four transformer blocks; excludes signed readout",
                           "contrast": "both independent 32->64 signed readout convolutions",
                           "fusion": "P(240,64), B(64,3,2), final zero-init Conv(64,64)",
                       }},
        "initialization_and_gradients": {
            "zero_final_classifier_emits_zero_residual": True,
            "zero_init_output_is_identity_z0_plus_zero": True,
            "contrast_hidden_first_probe_grad_norm": grad_norm(grad_hl),
            "contrast_readout_weight_first_probe_grad_norm": grad_norm(grad_contrast_weight),
            "classifier_final_weight_first_task_grad_norm": grad_norm(grad_final_weight),
            "classifier_project_first_task_grad_norm": grad_norm(grad_fusion_project),
            "classifier_upstream_first_step_blocked_by_zero_output": True,
        },
        "pixel_rearrangement": {
            "pixel_unshuffle4_shuffle4_exact_inverse": True,
            "final_pixel_shuffle4_checkerboard_representable": checker_represented,
            "checker_pattern": checker.int().tolist(),
        },
        "gradient_route": {
            "routed_aux_gradient_extracted_only_for_private_L_G_contrast": True,
            "unrouted_aux_theta_gradient_inspected_without_accumulating": True,
            "diagnostic_copy_final_conv_seeded_nonzero_weight_norm": diag_final_weight_norm,
            "z0_detached_for_private_path_probe": not diag_z0.requires_grad,
            "task_gradient_to_shared_feature_activations_norm": task_feature_activation_norm,
            "task_gradient_to_shared_theta_via_private_paths_norm": task_shared_parameter_norm,
            "task_reaches_shared_theta_through_private": task_shared_parameter_norm > 1e-8,
            "unrouted_aux_gradient_to_shared_theta_norm": raw_aux_main_norm,
            "unrouted_aux_would_change_shared_theta": raw_aux_main_norm > 1e-8,
            "private_task_gradient_norm": task_private_norm,
            "private_scaled_aux_gradient_norm": aux_private_norm,
            "private_combined_gradient_norm": private_combined_norm,
            "both_private_gradient_components_nonzero": task_private_norm > 1e-8 and aux_private_norm > 1e-8,
            "main_grad_matches_task_only_max_abs": main_route_diff,
            "classifier_grad_matches_task_only_max_abs": classifier_route_diff,
            "private_grad_matches_task_plus_scaled_aux_max_abs": private_route_diff,
            "all_parameters_unchanged_no_optimizer_step": params_unchanged,
            "original_zero_init_preserved": original_zero_init_preserved,
            "diagnostic_only_assigned_parameter_names": diagnostic_assignment_names,
            "lambda_D_synthetic_probe": lambda_d,
        },
        "mac_2048_analytic": analytic_mac,
    }


def main() -> None:
    report = run_checks()
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "parameters": report["parameters"]["actual"],
        "parameter_total": report["parameters"]["sum_new_parameters"],
        "mac_conv_linear": report["mac_2048_analytic"]["conv_linear_total"],
        "mac_attention_qk_av": report["mac_2048_analytic"]["attention_qk_plus_av_mac"],
        "gradient_route": report["gradient_route"],
        "report": str(REPORT_PATH),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
