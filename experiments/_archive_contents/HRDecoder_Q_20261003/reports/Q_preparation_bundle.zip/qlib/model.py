import sys
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

from .common import sha


def conv(in_channels, out_channels, stride=1):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, stride, 1),
                         nn.GroupNorm(8, out_channels), nn.GELU())


class ResidualRefine(nn.Module):
    def __init__(self, channels=128):
        super().__init__()
        self.body = nn.Sequential(conv(channels, channels),
                                  nn.Conv2d(channels, channels, 3, padding=1), nn.GroupNorm(8, channels))
        self.activation = nn.GELU()

    def forward(self, x):
        return self.activation(x + self.body(x))


class Q(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        if len(backbone.blocks) != 12 or backbone.embed_dim != 768 or backbone.patch_size != 16:
            raise ValueError('Wrong official ViT-B/16 architecture')
        self.projections = nn.ModuleList(nn.Conv2d(768, 128, 1) for _ in range(4))
        self.downsample = nn.Conv2d(128, 128, 3, stride=2, padding=1)
        self.scale_refine = nn.ModuleList(conv(128, 128) for _ in range(4))
        self.fusion_refine = nn.ModuleList(ResidualRefine() for _ in range(4))
        self.rgb = nn.Sequential(conv(3, 32, stride=2), conv(32, 64, stride=2), conv(64, 64))
        self.merge = conv(192, 128)
        self.head = nn.Conv2d(128, 4, 1)
        self.set_update(0)

    def set_update(self, completed):
        # 'completed=2000' configures optimization step 2001.
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        if completed >= 2000:
            for block in self.backbone.blocks[8:]:
                block.requires_grad_(True)
            self.backbone.norm.requires_grad_(True)
        self.completed = completed

    def forward(self, x):
        h, w = x.shape[-2:]
        if h != w or h % 32:
            raise ValueError('Window must be square and divisible by 32')
        layers = self.backbone.get_intermediate_layers(x, n=[2, 5, 8, 11], reshape=True, norm=True)
        expected = (len(x), 768, h//16, w//16)
        if len(layers) != 4 or any(tuple(f.shape) != expected for f in layers):
            raise ValueError('Official patch maps include unexpected tokens or geometry')
        features = []
        for i, (projection, refine, feature) in enumerate(zip(self.projections, self.scale_refine, layers)):
            f = projection(feature)
            if i < 2:
                f = F.interpolate(f, scale_factor=4 if i == 0 else 2, mode='bilinear', align_corners=False)
            elif i == 3:
                f = self.downsample(f)
            features.append(refine(f))
        expected_scales = [(h//s, w//s) for s in (4, 8, 16, 32)]
        if [tuple(f.shape[-2:]) for f in features] != expected_scales:
            raise ValueError('Explicit stride4/8/16/32 reconstruction failed')
        f = self.fusion_refine[3](features[3])
        for i in (2, 1, 0):
            f = F.interpolate(f, size=features[i].shape[-2:], mode='bilinear', align_corners=False)
            f = self.fusion_refine[i](f + features[i])
        f = self.merge(torch.cat((f, self.rgb(x)), dim=1))
        return F.interpolate(self.head(f), size=(h, w), mode='bilinear', align_corners=False)


def build_model(repo, checkpoint=None, expected_sha=None, allow_random_for_wiring=False):
    repo = Path(repo).resolve()
    sys.path.insert(0, str(repo))
    from dinov3.hub.backbones import dinov3_vitb16
    if checkpoint is None and not allow_random_for_wiring:
        raise ValueError('Official pretrained asset is not READY; no random formal training')
    backbone = dinov3_vitb16(pretrained=False)
    evidence = {'model': 'dinov3_vitb16', 'pretraining': 'LVD1689M', 'missing_keys': [], 'unexpected_keys': [],
                'mode': 'synthetic_wiring_only' if checkpoint is None else 'strict_pretrained'}
    if checkpoint is not None:
        if not expected_sha or sha(checkpoint) != expected_sha:
            raise ValueError('Pretraining SHA missing or changed')
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        result = backbone.load_state_dict(state, strict=True)
        evidence.update(sha256=expected_sha, missing_keys=list(result.missing_keys), unexpected_keys=list(result.unexpected_keys))
    model = Q(backbone)
    evidence['backbone_parameters'] = sum(p.numel() for p in backbone.parameters())
    evidence['total_parameters'] = sum(p.numel() for p in model.parameters())
    return model, evidence
