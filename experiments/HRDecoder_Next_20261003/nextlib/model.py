"""Frozen exact D student semantics + complete trained U0 + real RGB skips."""
import os
import sys
import torch
from torch import nn
from torch.nn import functional as F

from trainer.models.convnext_unet import ConvNeXtUNet, LayerNorm2d, _up
from .common import ROOT, checked_asset, strict_load, write_json, sha
from .geometry import tiles, global_view, align_global, patch_tensor, center_logits


class FrozenD(nn.Module):
    def __init__(self, backbone, head, report=None):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.report = report
        if head.channels != 64:
            raise ValueError('D pre-classifier must be the trained 64-channel head')
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    def train(self, mode=True):
        super().train(False)
        return self

    @torch.no_grad()
    def forward(self, image):
        branches = self.backbone(image)
        fused = torch.cat([b if b.shape[-2:] == branches[0].shape[-2:]
                           else F.interpolate(b, branches[0].shape[-2:],
                                              mode='bilinear', align_corners=False)
                           for b in branches], dim=1)
        x = self.head._transform_inputs([fused])
        x = self.head.convs(x)
        if self.head.context is not None:
            x = self.head.context(x)
        if self.head.concat_input:
            x = self.head.conv_cat(torch.cat((fused, x), dim=1))
        if x.shape[1] != 64:
            raise RuntimeError('Wrong pre-classifier feature tensor')
        return x


def load_frozen_d(settings, out=None):
    """Use the champion's own constructor and strict initializer, then retain student."""
    from .common import read_json
    s = read_json(settings['champion_settings'])
    for rel, expected in s['expected_sha256'].items():
        path = rel if os.path.isabs(rel) else str(ROOT/'vendor/champion_repo'/rel)
        if sha(path) != expected:
            raise ValueError(f'Champion source/weight identity mismatch: {path}')
    sys.path.insert(0, str(ROOT/'legacy'))
    import runtime
    cwd = os.getcwd()
    try:
        runtime.inject_repo(s['repo'])
        cfg = runtime.load_config(s['repo'], s['config'])
        cfg.model['source_checkpoint'] = str(checked_asset(settings, 'B'))
        model = runtime.build_model(cfg)
        dpath = checked_asset(settings, 'D')
        bpath = checked_asset(settings, 'B')
        runtime.strict_initialize(model, str(dpath), str(bpath))
        state = runtime._load_state(str(dpath))
        current = model.state_dict()
        for k, v in state.items():
            if not torch.equal(current[k].cpu(), v.cpu()):
                raise ValueError(f'Exact D load failed: {k}')
        report = {'D_sha256': sha(dpath), 'B_sha256': sha(bpath),
                  'loaded_keys': sorted(state), 'post_load_equal': True,
                  'retained': ['backbone', 'decode_head'],
                  'feature': 'FCN pre-classifier 64 channels', 'missing': [], 'unexpected': []}
        result = FrozenD(model.backbone, model.decode_head, report)
        del model
        if out:
            write_json(out, report)
        return result
    finally:
        os.chdir(cwd)


def zero_last(seq):
    nn.init.zeros_(seq[-1].weight)
    nn.init.zeros_(seq[-1].bias)


class SemanticDetailUNet(nn.Module):
    def __init__(self, local, global_branch, detail=False, lowpass_detail=False):
        super().__init__()
        self.local = local
        self.global_branch = global_branch
        self.has_detail = bool(detail)
        self.lowpass_detail = bool(lowpass_detail)
        self.global_adapter = nn.Sequential(nn.Conv2d(64, 32, 1), LayerNorm2d(32),
                                            nn.GELU(), nn.Conv2d(32, 32, 1))
        zero_last(self.global_adapter)
        if detail:
            self.detail_stem1 = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1),
                                               LayerNorm2d(16), nn.GELU())
            self.detail_stem2 = nn.Sequential(nn.Conv2d(16, 32, 3, stride=2, padding=1),
                                               LayerNorm2d(32), nn.GELU())
            self.detail_project2 = nn.Sequential(nn.Conv2d(32, 16, 1))
            self.detail_project1 = nn.Sequential(nn.Conv2d(16, 8, 1))
            zero_last(self.detail_project2)
            zero_last(self.detail_project1)
        for name, param in self.local.encoder.named_parameters():
            # Freeze stem, stages 0/1 and their downsampling; train stages 2/3.
            param.requires_grad_(name.startswith(('stages.2.', 'stages.3.',
                                                   'downsample_layers.2.', 'downsample_layers.3.')))

    def train(self, mode=True):
        super().train(mode)
        self.global_branch.eval()
        # Frozen early stochastic-depth blocks must also stay in eval mode.
        for i in (0, 1):
            self.local.encoder.stages[i].eval()
            self.local.encoder.downsample_layers[i].eval()
        return self

    def forward(self, local_rgb, aligned_global):
        if local_rgb.shape[-2:] != (aligned_global.shape[-2]*4, aligned_global.shape[-1]*4):
            raise ValueError('Coordinate-aligned global feature must be stride4')
        x0, x1, x2, x3 = self.local.encoder(local_rgb)
        d = self.local.decoder
        d4 = d.c4(x3)
        d3 = d.c3(torch.cat((_up(d4, x2.shape[-2:]), x2), 1))
        d2 = d.c2(torch.cat((_up(d3, x1.shape[-2:]), x1), 1))
        d1 = d.c1(torch.cat((_up(d2, x0.shape[-2:]), x0), 1))
        d1 = d1 + self.global_adapter(aligned_global)
        d0 = d.c0(_up(d1, (x0.shape[-2]*2, x0.shape[-1]*2)))
        if self.has_detail:
            detail_rgb = local_rgb
            if self.lowpass_detail:
                detail_rgb = F.interpolate(local_rgb, scale_factor=.25, mode='area')
                detail_rgb = F.interpolate(detail_rgb, local_rgb.shape[-2:], mode='bilinear', align_corners=False)
            rgb1 = self.detail_stem1(detail_rgb)
            rgb2 = self.detail_stem2(rgb1)
            d0 = d0 + self.detail_project2(rgb2)
        d00 = d.c00(_up(d0, local_rgb.shape[-2:]))
        if self.has_detail:
            d00 = d00 + self.detail_project1(rgb1)
        return d.cls(d.dropout(d00))

    def parameter_groups(self, pretrained_lr=1e-5, new_lr=1e-4):
        old, new = [], []
        for name, p in self.named_parameters():
            if p.requires_grad:
                (old if name.startswith('local.') else new).append(p)
        return [{'params': old, 'lr': pretrained_lr, 'base_lr': pretrained_lr, 'name': 'pretrained'},
                {'params': new, 'lr': new_lr, 'base_lr': new_lr, 'name': 'new'}]

    @torch.no_grad()
    def predict(self, rgb, center=768, halo=128, long_side=1024):
        self.eval()
        device = next(self.local.parameters()).device
        view, ledger = global_view(rgb, long_side, device)
        feature = self.global_branch(view)
        h, w = rgb.shape[:2]
        output = torch.empty((1, 4, h, w), dtype=torch.float32, device='cpu')
        for tile in tiles(h, w, center, halo):
            logits = self(patch_tensor(rgb, tile, device), align_global(feature, ledger, tile))
            cropped = center_logits(logits, tile)[..., :tile.height, :tile.width]
            output[..., tile.y:tile.y+tile.height, tile.x:tile.x+tile.width] = cropped.float().cpu()
        return output


def build_fusion(settings, start_name, arm, seed=42):
    torch.manual_seed(seed)
    global_branch = load_frozen_d(settings)
    local = ConvNeXtUNet(drop_path_rate=.1)
    report = strict_load(local, checked_asset(settings, start_name))
    # Same common-module RNG state in both arms; optional stem is constructed last.
    torch.manual_seed(seed + 1)
    if arm not in ('M0','M1','M1_lowpass'):
        raise ValueError('Unknown declared structure arm')
    model = SemanticDetailUNet(local, global_branch, detail=arm != 'M0', lowpass_detail=arm == 'M1_lowpass')
    report['D'] = global_branch.report
    report['trainable'] = [n for n, p in model.named_parameters() if p.requires_grad]
    report['frozen'] = [n for n, p in model.named_parameters() if not p.requires_grad]
    return model, report
