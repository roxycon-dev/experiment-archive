"""Official HACDR backbone + project four-logit U-Net; NOT a paper reproduction.

The published UnetHead hardcodes unrelated channel sizes and num_classes=5.
We preserve the published backbone code and use the common project decoder and
loss to obtain comparable independent four-channel predictions. Isolated import
avoids importing dozens of unrelated backbones and prevents mmseg collisions.
"""
import importlib.util
import sys
import types
from pathlib import Path

import torch.nn as nn
from . import DependencyBlocked
from .convnext_unet import UNetDecoder
from ..gate import GateError

def load_official_backbone(root):
    try:
        from mmcv.utils import Registry
        from mmcv.ops import ModulatedDeformConv2dPack
    except Exception as e:
        raise DependencyBlocked(f'HACDR requires working mmcv-full CUDA DCNv2: {e}') from e
    path = Path(root)/'mmseg/models/backbones/hacdr.py'
    if not path.is_file():
        raise DependencyBlocked(f'Official HACDR source missing: {path}')
    prefix = 'hr_experiment_hacdr'
    for name in [prefix, prefix+'.backbones']:
        package = types.ModuleType(name)
        package.__path__ = []
        sys.modules[name] = package
    builder = types.ModuleType(prefix+'.builder')
    builder.BACKBONES = Registry('experiment_official_hacdr')
    sys.modules[builder.__name__] = builder
    spec = importlib.util.spec_from_file_location(prefix+'.backbones.hacdr', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.HACDR

class HACDRFourLogits(nn.Module):
    def __init__(self, cls):
        super().__init__()
        self.backbone = cls(embed_dims=[32,64,160,256], DFFN_ratios=[8,8,4,4],
                            depths=[3,3,5,2], drop_rate=.1, drop_path_rate=.1,
                            option_k=7, norm_cfg=dict(type='BN', requires_grad=True))
        self.backbone.init_weights()
        self.decoder = UNetDecoder(dims=(32,64,160,256), dropout=.2)

    def forward(self, image):
        if any(s%32 for s in image.shape[-2:]):
            raise GateError('HACDR requires explicit 32-divisible inputs')
        return self.decoder(self.backbone(image))

    eval_forward = forward

def build_hacdr(cfg, *, gradient_checkpointing=None):
    if gradient_checkpointing:
        raise GateError('HACDR activation checkpointing is not yet verified')
    model = HACDRFourLogits(load_official_backbone(cfg['structure']['reference_root']))
    return model, dict(family='hacdr', backbone_prefixes=('backbone',),
                      backbone_norm_eval=False, weight_report=None, num_classes=4)
