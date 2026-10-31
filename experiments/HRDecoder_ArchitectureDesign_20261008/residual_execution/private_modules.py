"""Private modules copied unchanged from the reviewed v3 prototype."""
import math
from typing import Dict, Sequence, Tuple
import torch
from torch import Tensor, nn
import torch.nn.functional as F

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


