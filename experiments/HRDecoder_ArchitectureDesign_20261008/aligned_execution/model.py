"""Half-resolution multiscale W48 correction; same original HRDecoder GT branch."""
import torch
from torch import nn
import torch.nn.functional as F
from private_modules import Pointwise,ResidualBlock
from fast_objectives import gt_loss

class LightAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.features=nn.ModuleList([Pointwise(c,32) for c in (48,96,192,384)])
        self.rgb=Pointwise(12,16)
        self.probability=Pointwise(16,16)
        self.blocks=nn.Sequential(ResidualBlock(64,13,2),ResidualBlock(64,13,2))
        self.readout=nn.Conv2d(64,16,1,bias=True)
        nn.init.zeros_(self.readout.weight);nn.init.zeros_(self.readout.bias)
        assert sum(p.numel() for p in self.parameters())==79504
    def forward(self,image,features,z0):
        target=features[0].shape[-2:]
        values=[F.interpolate(layer(f.detach()),size=target,mode='bilinear',align_corners=False)
                for layer,f in zip(self.features,features)]
        semantic=sum(values)
        rgb=self.rgb(F.pixel_unshuffle(image,2))
        semantic=F.interpolate(semantic,size=rgb.shape[-2:],mode='bilinear',align_corners=False)
        probability=self.probability(F.pixel_unshuffle(z0.detach().float().sigmoid(),2))
        x=self.blocks(torch.cat((semantic,rgb,probability),1))
        return F.pixel_shuffle(self.readout(x),2)

class SACDStudent(nn.Module):
    def __init__(self, baseline, *, canvas=2048, private_seed=20261008):
        super().__init__()
        self.base, self.canvas = baseline, canvas
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(private_seed)
            self.private = LightAdapter()
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
        residual = self.private(image, features, z0)
        # Both direct and feature-mediated private-to-provider routes are cut.
        # Numerically this is a continuous signed logit residual, in FP32.
        logits = z0.float().detach() + residual.float()
        return {'logits': logits, 'base_logits': z0, 'residual_logits': residual,
                'hr_loss': hr_loss}


def inference_probabilities(output, original_hw):
    # Addition is linear; sigmoid follows native-resolution logit interpolation.
    logits = F.interpolate(output['logits'].float(), size=original_hw,
                           mode='bilinear', align_corners=False)
    return logits.sigmoid()
