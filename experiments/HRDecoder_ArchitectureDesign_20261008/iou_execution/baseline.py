"""Real archived HRDecoder, one shared HRNet forward, and reviewed v3 modules."""
import copy
import hashlib
import random
import sys
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

W48_SHA = "d2186c55782308392ae14cf179ac4840143b09abdbba81aa3f9274c795ebc960"

def install_legacy_bf16_interpolation():
    """Torch 2.0 CUDA bilinear lacks BF16: compute FP32, return original dtype."""
    if getattr(F.interpolate, "_hrdecoder_bf16_compatible", False):
        return
    original = F.interpolate
    def interpolate(input, *args, **kwargs):
        mode = kwargs.get("mode", args[2] if len(args) > 2 else "nearest")
        if input.is_cuda and input.dtype == torch.bfloat16 and mode == "bilinear":
            return original(input.float(), *args, **kwargs).to(input.dtype)
        return original(input, *args, **kwargs)
    interpolate._hrdecoder_bf16_compatible = True
    F.interpolate = interpolate

def file_sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(8*1024**2), b""): digest.update(block)
    return digest.hexdigest()

def tensor_state_sha(module):
    digest = hashlib.sha256()
    for name, value in module.state_dict().items():
        value = value.detach().cpu().contiguous()
        digest.update(name.encode()); digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode()); digest.update(value.numpy().tobytes())
    return digest.hexdigest()

def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def build_baseline(repo, pretrained, *, seed=42, canvas=2048, verified_weight_sha=None):
    repo = str(Path(repo).resolve())
    if repo not in sys.path: sys.path.insert(0, repo)
    from mmcv import Config
    from mmseg.models import build_segmentor
    assert verified_weight_sha == W48_SHA, "Require previously audited ImageNet W48 identity"
    cfg = Config.fromfile(str(Path(repo)/"configs/lesion/exp/baseline_split613.py"))
    settings = copy.deepcopy(cfg.model)
    settings.pretrained = str(Path(pretrained).resolve())
    if canvas != 2048:
        assert canvas % 64 == 0, "Probe canvas must be divisible by 64"
        settings.hr_settings.hr_scale = (canvas//2, canvas//2)
    seed_everything(seed)
    baseline = build_segmentor(settings)
    baseline.init_weights()
    # Check every segmentation-backbone tensor, including BN buffers, after
    # the original constructor and BaseModule initialization have both run.
    loaded = torch.load(pretrained, map_location="cpu")
    state = loaded.get("state_dict", loaded)
    for name, tensor in baseline.backbone.state_dict().items():
        assert name in state and torch.equal(tensor.cpu(), state[name]), "ImageNet tensor mismatch: "+name
    del state, loaded
    assert baseline.backbone.norm_eval is False
    assert baseline.crop_num == 4 and baseline.fuse_mode == "simple"
    assert baseline.detail_branch is None and baseline.detail_head is None
    return baseline, cfg

