"""C/S/D/SD share a strictly loaded F12 parent and the original DDR decoder."""
import copy
import hashlib
import torch
from torch import nn
from torch.nn import functional as F
from spatial import Spatial


DECODER_MODULES = ('projections', 'downsample', 'scale_refine', 'fusion_refine', 'rgb', 'merge', 'head')


class DomainDecoder(nn.Module):
    def __init__(self, parent, has_spatial):
        super().__init__()
        for name in DECODER_MODULES:
            setattr(self, name, getattr(parent, name))
        if has_spatial:
            self.spatial_residual = nn.ModuleList(nn.Conv2d(c, 128, 1) for c in (64, 768, 768, 768))
            for projection in self.spatial_residual:
                nn.init.zeros_(projection.weight)
                nn.init.zeros_(projection.bias)

    def forward(self, rgb, layers, spatial_maps=None):
        h, w = rgb.shape[-2:]
        features = []
        for i, (projection, refine, layer) in enumerate(zip(self.projections, self.scale_refine, layers)):
            feature = projection(layer)
            if i < 2:
                feature = F.interpolate(feature, scale_factor=4 if i == 0 else 2,
                                        mode='bilinear', align_corners=False)
            elif i == 3:
                feature = self.downsample(feature)
            if spatial_maps is not None:
                residual = self.spatial_residual[i](spatial_maps[i])
                assert residual.shape == feature.shape
                # After projection and stride alignment, before the unchanged scale_refine.
                feature = feature + residual
            features.append(refine(feature))
        assert [tuple(x.shape[-2:]) for x in features] == [(h // s, w // s) for s in (4, 8, 16, 32)]
        feature = self.fusion_refine[3](features[3])
        for i in (2, 1, 0):
            feature = F.interpolate(feature, size=features[i].shape[-2:], mode='bilinear', align_corners=False)
            feature = self.fusion_refine[i](feature + features[i])
        feature = self.merge(torch.cat((feature, self.rgb(rgb)), dim=1))
        return F.interpolate(self.head(feature), size=(h, w), mode='bilinear', align_corners=False)


class FinalModel(nn.Module):
    def __init__(self, parent, arm, seed, backend='cuda', checkpoint_blocks=False):
        super().__init__()
        assert arm in ('C', 'S', 'D', 'SD')
        self.arm, self.backend, self.checkpoint_blocks = arm, backend, checkpoint_blocks
        self.backbone = parent.backbone
        assert self.backbone.n_storage_tokens == 4 and len(self.backbone.blocks) == 12
        self.has_spatial, self.has_source = arm in ('S', 'SD'), arm in ('D', 'SD')
        # Independent module initialization never advances paired backbone/data RNG.
        init_seed = int.from_bytes(hashlib.sha256(f'campaign20261005|{seed}|module_init'.encode()).digest()[:8], 'little') % 2**63
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(init_seed)
            if self.has_spatial:
                self.spatial = Spatial()
            self.ddr = DomainDecoder(parent, self.has_spatial)
            if self.has_source:
                self.tjdr = copy.deepcopy(self.ddr)
        self.backbone.requires_grad_(True)
        for name, parameter in self.backbone.named_parameters():
            if name == 'mask_token' or name.startswith(('cls_norm.', 'local_cls_norm.')):
                parameter.requires_grad_(False)

    def set_update(self, completed):
        self.completed = completed

    def forward(self, rgb, source='DDR'):
        assert source in ('DDR', 'TJDR')
        if source == 'TJDR' and not self.has_source:
            raise ValueError('This arm has no source-domain decoder')
        h, w = rgb.shape[-2:]
        assert h == w and h % 32 == 0
        spatial_maps = None
        if not self.has_spatial:
            layers = self.backbone.get_intermediate_layers(rgb, n=[2, 5, 8, 11], reshape=True, norm=True)
        else:
            sequence, (ph, pw) = self.backbone.prepare_tokens_with_masks(rgb)
            prefix = self.backbone.n_storage_tokens + 1
            assert sequence.shape[1] == prefix + ph * pw
            fine, spatial_tokens, spatial_shapes = self.spatial.prepare(rgb)
            layers = []
            for stage, interaction in enumerate(self.spatial.interactions):
                patch = sequence[:, prefix:]
                patch = patch + interaction.injector(interaction.inject_patch_norm(patch),
                    interaction.inject_spatial_norm(spatial_tokens), [(ph, pw)], spatial_shapes, self.backend)
                sequence = torch.cat((sequence[:, :prefix], patch), dim=1)
                for block in self.backbone.blocks[stage * 3:stage * 3 + 3]:
                    rope = self.backbone.rope_embed(H=ph, W=pw) if self.backbone.rope_embed is not None else None
                    if self.checkpoint_blocks and self.training:
                        from torch.utils.checkpoint import checkpoint
                        sequence = checkpoint(block, sequence, rope, use_reentrant=False, preserve_rng_state=True)
                    else:
                        sequence = block(sequence, rope)
                patch = sequence[:, prefix:]
                if self.backbone.untie_cls_and_patch_norms:
                    normalized = self.backbone.norm(patch)
                else:
                    normalized = self.backbone.norm(sequence)[:, prefix:]
                layers.append(normalized.reshape(len(rgb), ph, pw, 768).permute(0, 3, 1, 2).contiguous())
                spatial_tokens = spatial_tokens + interaction.extractor(
                    interaction.extract_spatial_norm(spatial_tokens), interaction.extract_patch_norm(patch),
                    spatial_shapes, [(ph, pw)], self.backend)
            spatial_maps = self.spatial.unpack(fine, spatial_tokens, spatial_shapes)
        decoder = self.ddr if source == 'DDR' else self.tjdr
        return decoder(rgb, layers, spatial_maps)


def inherited_parameter_map(parent):
    """Explicit one-to-one identity mapping; no loose checkpoint key filtering."""
    return {name: (name if name.startswith('backbone.') else 'ddr.' + name)
            for name in parent.state_dict()}
