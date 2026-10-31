"""Auditable v4 reference. No training launcher; v3 runtime is untouched."""
from pathlib import Path
import sys
import torch
from torch import nn
import torch.nn.functional as F

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'training_execution'))
from private_modules import LocalPath,GlobalPath
from v4_objectives import gt_loss,trusted_kl,mixture_logits

WEIGHTS=(.5625,.1875,.25)

class ComponentHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.output=nn.Conv2d(32,64,1,bias=True)
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
    def forward(self,x):return F.pixel_shuffle(self.output(x),4)

class V4Private(nn.Module):
    def __init__(self):
        super().__init__()
        self.local=LocalPath();self.global_path=GlobalPath()
        self.head_u=ComponentHead();self.head_s=ComponentHead()
    def forward(self,image,features):
        f1,f2,f3,f4=features;r=F.pixel_unshuffle(image,4)
        return self.head_u(self.local(r,f1,f2,f3)),self.head_s(self.global_path(r,f1,f2,f3,f4))

class V4Student(nn.Module):
    """One original HRNet call, original decode path, independent component heads."""
    def __init__(self,baseline,canvas=2048,private_seed=20261008):
        super().__init__();self.base=baseline;self.canvas=canvas
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(private_seed)
            self.private=V4Private()
    def component_parameters(self):
        # Component-specific teacher gradients never update shared HRNet.
        return tuple(self.base.decode_head.parameters())+tuple(self.private.parameters())
    def forward(self,image,label=None,valid=None):
        assert image.shape[-2:]==(self.canvas,self.canvas)
        base=self.base;size=image.shape[-2:]
        features=base.extract_feat(F.interpolate(image,size=base.hr_scale,mode='bilinear',align_corners=False))
        fused=base._transform_inputs(features)
        lr_logits=base.decode_head([fused])
        hr_loss=lr_logits.float().sum()*0
        if self.training and label is not None:
            from mmseg.models.segmentors.HRDecoder import get_crop_bbox,crop
            scale=base.get_random_hr_scale()
            boxes=[get_crop_bbox(size,scale,base.divisible) for _ in range(base.crop_num)]
            crops=torch.cat([base._resizecrop_feats(size,fused,b) for b in boxes])
            crop_logits=base.decode_head([crops])
            crop_labels=torch.cat([crop(label,b) for b in boxes]);crop_valid=torch.cat([crop(valid,b) for b in boxes])
            crop_logits=F.interpolate(crop_logits.float(),size=crop_labels.shape[-2:],mode='bilinear',align_corners=False)
            hr_loss=gt_loss(crop_logits,crop_labels,crop_valid)
        global_logits=F.interpolate(lr_logits,size=size,mode='bilinear',align_corners=False)
        za=global_logits.clone();slide=base.hr_slide_inference(size,fused,[])
        for local,box in zip(slide['logits'],slide['boxes']):
            y1,y2,x1,x2=box
            za[...,y1:y2,x1:x2]=base._fuse_logits(global_logits[...,y1:y2,x1:x2],local)
        zu,zs=self.private(image,features)
        return {'logits':mixture_logits((za,zu,zs),WEIGHTS),'member_logits':(za,zu,zs),'hr_loss':hr_loss}

def lambda_k(update):
    if update<=1500:return 0.
    if update<=4000:return .5*(update-1500)/2500
    if update<=32000:return .5
    return .5-.4*(update-32000)/8000

def losses(output,label,valid,a,u,s,update):
    za,zu,zs=output['member_logits'];zm=output['logits']
    target=sum(w*q.detach().float() for w,q in zip(WEIGHTS,(a,u,s)))
    main_gt=.5*gt_loss(zm,label,valid)+.25*gt_loss(za,label,valid)+.125*gt_loss(zu,label,valid)+.125*gt_loss(zs,label,valid)
    lk=lambda_k(update)
    task=main_gt+.1*output['hr_loss']+lk*trusted_kl(zm,target,label,valid)
    components=.5*lk*sum(w*trusted_kl(z,q,label,valid) for w,z,q in zip(WEIGHTS,(za,zu,zs),(a,u,s)))
    return task,components

def routed_backward(task,components,component_parameters,scale=1.):
    parameters=tuple(component_parameters)
    gradients=torch.autograd.grad(components*scale,parameters,retain_graph=True,allow_unused=True)
    (task*scale).backward()
    for p,g in zip(parameters,gradients):
        if g is not None:
            if p.grad is None:p.grad=g.detach()
            else:p.grad.add_(g.detach())

def inference_probabilities(output,original_hw):
    # Preserve the probability-mixture identity and lower bound on native pixels.
    logits=[F.interpolate(z.float(),size=original_hw,mode='bilinear',align_corners=False) for z in output['member_logits']]
    return sum(w*z.sigmoid() for w,z in zip(WEIGHTS,logits))
