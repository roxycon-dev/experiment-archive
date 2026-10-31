"""One HRNet, original HRDecoder, unchanged L/G, and SACD action readout."""
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
        # Separate nonlinear readouts; all retain the same image and student context.
        self.readouts = nn.ModuleList()
        for _ in range(4):
            output = nn.Conv2d(32, 48, 1, bias=True)
            nn.init.zeros_(output.weight)
            with torch.no_grad():
                biases = torch.tensor([math.log(.98), math.log(.01), math.log(.01)])
                output.bias.copy_(biases.repeat_interleave(16))
            self.readouts.append(nn.Sequential(Pointwise(128, 32), ResidualBlock(32, 3, 2), output))
        assert sum(p.numel() for p in self.parameters()) == 40768
    def forward(self, local, global_path, p0):
        features = torch.cat((local, global_path, F.pixel_unshuffle(p0.detach(), 4)), dim=1)
        # Pack the four independent readouts into grouped kernels. Parameter
        # ownership and each class's 32-channel normalization remain separate.
        def norm(value, modules):
            n, _, h, w = value.shape
            original_dtype=value.dtype
            value=value.reshape(n,4,32,h,w).permute(0,1,3,4,2)
            if value.dtype in (torch.float16,torch.bfloat16):value=value.float()
            weight=torch.stack([m.norm.weight for m in modules]).to(value.dtype)
            bias=torch.stack([m.norm.bias for m in modules]).to(value.dtype)
            value=F.layer_norm(value,(32,),eps=1e-5)
            value=value*weight[None,:,None,None,:]+bias[None,:,None,None,:]
            return value.to(original_dtype).permute(0,1,4,2,3).reshape(n,128,h,w).contiguous()
        points=[head[0] for head in self.readouts]
        blocks=[head[1] for head in self.readouts]
        x=F.conv2d(features,torch.cat([m.conv.weight for m in points]))
        x=F.gelu(norm(x,[m.norm for m in points]))
        y=F.conv2d(x,torch.cat([m.dw.weight for m in blocks]),padding=1,groups=128)
        y=norm(y,[m.norm for m in blocks])
        y=F.gelu(F.conv2d(y,torch.cat([m.expand.weight for m in blocks]),groups=4))
        x=x+F.conv2d(y,torch.cat([m.project.weight for m in blocks]),groups=4)
        raw=F.conv2d(x,torch.cat([head[-1].weight for head in self.readouts]),
                     torch.cat([head[-1].bias for head in self.readouts]),groups=4)
        return F.pixel_shuffle(raw,4).reshape(raw.shape[0],4,3,p0.shape[2],p0.shape[3])

class SACDPrivate(nn.Module):
    def __init__(self):
        super().__init__()
        self.local = LocalPath()
        self.global_path = GlobalPath()
        self.head = CorrectionHead()
        assert sum(p.numel() for p in self.local.parameters()) == 2918816
        assert sum(p.numel() for p in self.global_path.parameters()) == 3459872
        assert sum(p.numel() for p in self.parameters()) == 6419456
    def forward(self, image, features, z0):
        f1, f2, f3, f4 = features
        r = F.pixel_unshuffle(image, 4)
        local = self.local(r, f1, f2, f3)
        global_path = self.global_path(r, f1, f2, f3, f4)
        return self.head(local, global_path, z0.float().sigmoid())

class SACDStudent(nn.Module):
    def __init__(self, baseline, *, canvas=2048, private_seed=20261008):
        super().__init__()
        self.base, self.canvas = baseline, canvas
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(private_seed)
            self.private = SACDPrivate()
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
        private_features = tuple(feature.detach() for feature in features)
        actions = self.private(image, private_features, z0)
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
