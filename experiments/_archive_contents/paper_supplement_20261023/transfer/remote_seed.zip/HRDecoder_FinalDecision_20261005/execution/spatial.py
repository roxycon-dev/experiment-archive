"""Frozen 2026-10-05 spatial interface; no pooled-attention substitute.

MSDA coordinates and initialization follow the Apache-2.0 ViT-Adapter/
Deformable-DETR implementation retained under vendor/vit_adapter_reference.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Function
from torch.autograd.function import once_differentiable


def reference_attention(value, shapes, locations, weights):
    batch, _, heads, channels = value.shape
    query = locations.shape[1]
    levels, points = locations.shape[3:5]
    pieces = value.split([h * w for h, w in shapes], dim=1)
    sampled = []
    for level, ((h, w), piece) in enumerate(zip(shapes, pieces)):
        image = piece.flatten(2).transpose(1, 2).reshape(batch * heads, channels, h, w)
        grid = (2 * locations[:, :, :, level] - 1).transpose(1, 2).flatten(0, 1)
        sampled.append(F.grid_sample(image, grid, mode='bilinear', padding_mode='zeros',
                                     align_corners=False))
    attention = weights.transpose(1, 2).reshape(batch * heads, 1, query, levels * points)
    result = (torch.stack(sampled, dim=-2).flatten(-2) * attention).sum(-1)
    return result.view(batch, heads * channels, query).transpose(1, 2).contiguous()


class CUDAAttention(Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type='cuda', cast_inputs=torch.float32)
    def forward(ctx, value, shapes, starts, locations, weights):
        import MultiScaleDeformableAttention as kernel
        args = (value.contiguous(), shapes, starts, locations.contiguous(), weights.contiguous())
        result = kernel.ms_deform_attn_forward(*args, 64)
        ctx.save_for_backward(*args)
        return result

    @staticmethod
    @once_differentiable
    @torch.amp.custom_bwd(device_type='cuda')
    def backward(ctx, grad):
        import MultiScaleDeformableAttention as kernel
        dv, dl, dw = kernel.ms_deform_attn_backward(*ctx.saved_tensors, grad.contiguous(), 64)
        return dv, None, None, dl, dw


def grid_centers(shapes, device):
    result = []
    for h, w in shapes:
        y, x = torch.meshgrid(torch.arange(h, dtype=torch.float32, device=device) + .5,
                              torch.arange(w, dtype=torch.float32, device=device) + .5,
                              indexing='ij')
        result.append(torch.stack((x / w, y / h), dim=-1).reshape(-1, 2))
    return torch.cat(result, dim=0)


class DeformAttention(nn.Module):
    def __init__(self, levels, zero_output=False):
        super().__init__()
        self.dimension, self.heads, self.points, self.levels = 768, 12, 4, levels
        self.offset = nn.Linear(768, 12 * levels * 4 * 2)
        self.weight = nn.Linear(768, 12 * levels * 4)
        self.value = nn.Linear(768, 768)
        self.output = nn.Linear(768, 768)
        nn.init.zeros_(self.offset.weight)
        angle = torch.arange(12, dtype=torch.float32) * (2 * math.pi / 12)
        direction = torch.stack((angle.cos(), angle.sin()), dim=-1)
        direction = direction / direction.abs().amax(dim=-1, keepdim=True)
        bias = direction[:, None, None, :].repeat(1, levels, 4, 1)
        bias *= torch.arange(1, 5, dtype=torch.float32)[None, None, :, None]
        with torch.no_grad():
            self.offset.bias.copy_(bias.flatten())
        nn.init.zeros_(self.weight.weight)
        nn.init.zeros_(self.weight.bias)
        nn.init.xavier_uniform_(self.value.weight)
        nn.init.zeros_(self.value.bias)
        if zero_output:
            nn.init.zeros_(self.output.weight)
        else:
            nn.init.xavier_uniform_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        self._geometry = {}

    def geometry(self, query_shapes, value_shapes, device):
        key = (tuple(query_shapes), tuple(value_shapes), str(device))
        if key not in self._geometry:
            shapes = torch.tensor(value_shapes, dtype=torch.long, device=device)
            sizes = shapes.prod(dim=1)
            starts = torch.cat((sizes.new_zeros(1), sizes.cumsum(0)[:-1]))
            refs = grid_centers(query_shapes, device)
            normalizer = torch.tensor([(w, h) for h, w in value_shapes],
                                      dtype=torch.float32, device=device)
            self._geometry[key] = shapes, starts, refs, normalizer
        return self._geometry[key]

    def forward(self, query, value_tokens, query_shapes, value_shapes, backend):
        batch, length, _ = query.shape
        assert len(value_shapes) == self.levels
        assert value_tokens.shape[1] == sum(h * w for h, w in value_shapes)
        shapes, starts, refs, normalizer = self.geometry(query_shapes, value_shapes, query.device)
        value = self.value(value_tokens).view(batch, -1, 12, 64)
        offsets = self.offset(query).view(batch, length, 12, self.levels, 4, 2).float()
        weights = self.weight(query).view(batch, length, 12, self.levels * 4).float()
        weights = weights.softmax(dim=-1).view(batch, length, 12, self.levels, 4)
        locations = refs[None, :, None, None, None, :] + offsets / normalizer[None, None, None, :, None, :]
        # The author CUDA kernel supports FP32/FP64, not BF16. Both reference and
        # optimized sampling use FP32; surrounding projections remain under AMP.
        with torch.autocast(device_type=query.device.type, enabled=False):
            if backend == 'cuda':
                assert query.is_cuda, 'CUDA kernel needs a CUDA tensor'
                sampled = CUDAAttention.apply(value.float(), shapes, starts, locations, weights)
            elif backend == 'reference':
                sampled = reference_attention(value.float(), value_shapes, locations, weights)
            else:
                raise ValueError('Unaccepted attention backend')
        return self.output(sampled)


class Interaction(nn.Module):
    def __init__(self):
        super().__init__()
        self.inject_patch_norm = nn.LayerNorm(768)
        self.inject_spatial_norm = nn.LayerNorm(768)
        self.extract_patch_norm = nn.LayerNorm(768)
        self.extract_spatial_norm = nn.LayerNorm(768)
        self.injector = DeformAttention(3, zero_output=True)
        self.extractor = DeformAttention(1)


class Spatial(nn.Module):
    def __init__(self):
        super().__init__()
        channels = [3, 32, 64, 128, 256, 256]
        self.pyramid = nn.ModuleList(nn.Sequential(
            nn.Conv2d(channels[i], channels[i + 1], 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, channels[i + 1]), nn.GELU()) for i in range(5))
        self.input_projection = nn.ModuleList(nn.Conv2d(c, 768, 1) for c in (128, 256, 256))
        self.level_embedding = nn.Parameter(torch.empty(3, 768))
        nn.init.normal_(self.level_embedding, std=.02)
        self.interactions = nn.ModuleList(Interaction() for _ in range(4))

    def prepare(self, rgb):
        maps = []
        x = rgb
        for i, block in enumerate(self.pyramid):
            x = block(x)
            if i > 0:
                maps.append(x)
        shapes = [tuple(m.shape[-2:]) for m in maps[1:]]
        tokens = torch.cat([projection(m).flatten(2).transpose(1, 2) + self.level_embedding[i]
                            for i, (m, projection) in enumerate(zip(maps[1:], self.input_projection))], dim=1)
        return maps[0], tokens, shapes

    @staticmethod
    def unpack(fine, tokens, shapes):
        batch = tokens.shape[0]
        pieces = tokens.split([h * w for h, w in shapes], dim=1)
        return [fine] + [x.transpose(1, 2).reshape(batch, 768, h, w)
                         for x, (h, w) in zip(pieces, shapes)]
