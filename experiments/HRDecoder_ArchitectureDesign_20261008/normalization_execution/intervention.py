"""Only teacher KL normalization changes; identical model and GT loss."""
import torch
import torch.nn.functional as F
from fast_objectives import masks,macro

def balanced_kl(logits,teacher,label,valid):
 z=logits.double() if logits.dtype==torch.float64 else logits.float()
 q=teacher.detach().to(z.dtype);y,m=masks(z,label,valid)
 trusted=(q>.5)==y
 kl=(F.binary_cross_entropy_with_logits(z,q,reduction='none')+torch.xlogy(q,q)+torch.xlogy(1-q,1-q)).clamp_min(0)
 positive=m&y;negative=m&~y
 value=.5*(kl*positive*trusted).sum((-2,-1))/positive.sum((-2,-1)).clamp_min(1)
 value=value+.5*(kl*negative*trusted).sum((-2,-1))/negative.sum((-2,-1)).clamp_min(1)
 return macro(value,m)

def apply_intervention(model,arm):
 if arm=='gt_stratified':
  import train as core
  # core.losses resolves its global in objectives, not in train.
  import objectives
  objectives.trusted_kl=balanced_kl
 else:assert arm=='control'
 return model
