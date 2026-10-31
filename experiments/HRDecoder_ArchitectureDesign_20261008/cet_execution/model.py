"""One HRNet, original HRDecoder, unchanged L/G, and CET action readout."""
import math
import torch
from torch import nn
import torch.nn.functional as F
from private_modules import LocalPath, GlobalPath, Pointwise, ResidualBlock
from fast_objectives import gt_loss
from objectives import transport_logits

class CorrectionHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.project = Pointwise(128, 64)
        self.block = ResidualBlock(64, 3, 2)
        self.output = nn.Conv2d(64, 192, 1, bias=True)
        nn.init.zeros_(self.output.weight)
        with torch.no_grad():
            # PixelShuffle maps each output channel's sixteen contiguous subpixels.
            biases = torch.tensor([math.log(.98), math.log(.01), math.log(.01)]).repeat(4)
            self.output.bias.copy_(biases.repeat_interleave(16))
        assert sum(p.numel() for p in self.parameters()) == 37888
    def forward(self, local, global_path, p0):
        features = torch.cat((local, global_path, F.pixel_unshuffle(p0.detach(), 4)), dim=1)
        raw = F.pixel_shuffle(self.output(self.block(self.project(features))), 4)
        return raw.reshape(raw.shape[0], 4, 3, *raw.shape[-2:])

class CETPrivate(nn.Module):
    def __init__(self):
        super().__init__()
        self.local = LocalPath()
        self.global_path = GlobalPath()
        self.head = CorrectionHead()
        assert sum(p.numel() for p in self.local.parameters()) == 2918816
        assert sum(p.numel() for p in self.global_path.parameters()) == 3459872
        assert sum(p.numel() for p in self.parameters()) == 6416576
    def forward(self, image, features, z0):
        f1, f2, f3, f4 = features
        r = F.pixel_unshuffle(image, 4)
        local = self.local(r, f1, f2, f3)
        global_path = self.global_path(r, f1, f2, f3, f4)
        return self.head(local, global_path, z0.float().sigmoid())

class CETStudent(nn.Module):
    def __init__(self, baseline, *, canvas=2048, private_seed=20261008):
        super().__init__()
        self.base, self.canvas = baseline, canvas
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(private_seed)
            self.private = CETPrivate()
    def private_parameters(self):
        return tuple(self.private.parameters())
    def forward(self, image, label=None, valid=None):
        assert image.shape[-2:] == (self.canvas, self.canvas)
        base, size = self.base, image.shape[-2:]
        features = base.extract_feat(F.interpolate(image, size=base.hr_scale, mode='bilinear', align_corners=False))
        fused = base._transform_inputs(features)
        lr_logits = base.decode_head([fused])
        hr_loss = lr_logits.float().sum()*0
        if self.training and label is not None:
            from mmseg.models.segmentors.HRDecoder import get_crop_bbox, crop
            scale = base.get_random_hr_scale()
            boxes = [get_crop_bbox(size, scale, base.divisible) for _ in range(base.crop_num)]
            crops = torch.cat([base._resizecrop_feats(size, fused, b) for b in boxes])
            crop_logits = base.decode_head([crops])
            crop_labels = torch.cat([crop(label, b) for b in boxes])
            crop_valid = torch.cat([crop(valid, b) for b in boxes])
            crop_logits = F.interpolate(crop_logits.float(), size=crop_labels.shape[-2:], mode='bilinear', align_corners=False)
            hr_loss = gt_loss(crop_logits, crop_labels, crop_valid)
        global_logits = F.interpolate(lr_logits, size=size, mode='bilinear', align_corners=False)
        z0 = global_logits.clone()
        slide = base.hr_slide_inference(size, fused, [])
        for local, box in zip(slide['logits'], slide['boxes']):
            y1, y2, x1, x2 = box
            z0[..., y1:y2, x1:x2] = base._fuse_logits(global_logits[..., y1:y2, x1:x2], local)
        actions = self.private(image, features, z0)
        logits, log_probs = transport_logits(z0, actions)
        return {'logits': logits, 'base_logits': z0, 'action_logits': actions,
                'action_log_probs': log_probs, 'hr_loss': hr_loss}

def inference_probabilities(output, original_hw):
    # Nonlinear transport does not commute with interpolation.
    base = F.interpolate(output['base_logits'].float(), size=original_hw, mode='bilinear', align_corners=False)
    actions = output['action_logits']
    native = F.interpolate(actions.flatten(1, 2).float(), size=original_hw, mode='bilinear', align_corners=False)
    pi = native.reshape(native.shape[0], 4, 3, *original_hw).softmax(dim=2)
    return pi[:, :, 0]*base.sigmoid() + pi[:, :, 1]
