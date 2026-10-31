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
from private_modules import V3PrivateModules
from objectives import masked_dice

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

def build_baseline(repo, pretrained, *, seed=42, canvas=2048):
    repo = str(Path(repo).resolve())
    if repo not in sys.path: sys.path.insert(0, repo)
    from mmcv import Config
    from mmseg.models import build_segmentor
    assert file_sha(pretrained) == W48_SHA, "Wrong ImageNet W48 file"
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

class V3Student(nn.Module):
    def __init__(self, baseline, *, private_seed=20261008, canvas=2048, base_only=False):
        super().__init__()
        self.base = baseline
        self.canvas = canvas
        # No new initialization advances the already-initialized baseline RNG.
        self.private = None
        if not base_only:
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(private_seed)
                self.private = V3PrivateModules()
            assert sum(p.numel() for p in self.private.parameters()) == 6419648

    def private_parameters(self):
        if self.private is None: return ()
        parts = (self.private.local, self.private.global_path,
            self.private.contrast_u, self.private.contrast_s)
        return tuple(p for part in parts for p in part.parameters())

    def forward(self, image, label=None, valid=None):
        assert image.shape[-2:] == (self.canvas, self.canvas)
        base = self.base
        size = image.shape[-2:]
        low = F.interpolate(image, size=base.hr_scale, mode="bilinear", align_corners=False)
        features = base.extract_feat(low)  # The only HRNet call.
        fused = base._transform_inputs(features)
        lr_logits = base.decode_head([fused])
        hr_loss = lr_logits.float().sum()*0
        if self.training and label is not None:
            from mmseg.models.segmentors.HRDecoder import get_crop_bbox, crop
            scale = base.get_random_hr_scale()
            boxes = [get_crop_bbox(size, scale, base.divisible) for _ in range(base.crop_num)]
            crops = torch.cat([base._resizecrop_feats(size, fused, box) for box in boxes])
            crop_logits = base.decode_head([crops])
            crop_labels = torch.cat([crop(label, box) for box in boxes])
            crop_valid = torch.cat([crop(valid, box) for box in boxes])
            hr_loss = masked_dice(crop_logits, crop_labels, crop_valid)
        global_logits = F.interpolate(lr_logits, size=size, mode="bilinear", align_corners=False)
        z0 = global_logits.clone()
        slide = base.hr_slide_inference(size, fused, [])
        for local_logits, box in zip(slide["logits"], slide["boxes"]):
            y1, y2, x1, x2 = box
            z0[..., y1:y2, x1:x2] = base._fuse_logits(
                global_logits[..., y1:y2, x1:x2], local_logits)
        if self.private is None:
            return {"logits": z0, "base_logits": z0, "hr_loss": hr_loss}
        result = self.private(image, features, z0)
        return {"logits": result["logits"], "d_hat_u": result["d_hat_u"],
            "d_hat_s": result["d_hat_s"], "base_logits": z0,
            "hr_loss": hr_loss, "residual": result["residual"]}
