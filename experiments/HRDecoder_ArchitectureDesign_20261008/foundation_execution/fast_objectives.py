"""Original squared-denominator Dice, with previously registered valid masking."""
import torch
import torch.nn.functional as F

def masks(logits,label,valid):
    labels=label[:,0] if label.ndim==4 else label
    y=torch.stack([labels==c for c in range(1,5)],1)
    if valid.ndim==3:valid=valid[:,None]
    return y,valid.bool().expand_as(y)&(labels[:,None]!=255)

def macro(value,mask):
    active=mask.sum((-2,-1))>0
    return (value*active).sum()/active.sum().clamp_min(1)

def gt_loss(logits,label,valid):
    z=logits.double() if logits.dtype==torch.float64 else logits.float()
    y,m=masks(z,label,valid); p=z.sigmoid()
    dice=1-(2*(p*y*m).sum((-2,-1))+1e-5)/((p.square()*m).sum((-2,-1))+(y*m).sum((-2,-1))+1e-5)
    # Empty but annotated classes remain included, as in original BinaryDice.
    return macro(dice,m)

def residual_gt(logits,label,valid):
    z=logits.double() if logits.dtype==torch.float64 else logits.float()
    y,m=masks(z,label,valid)
    bce=F.binary_cross_entropy_with_logits(z,y.to(z.dtype),reduction='none')
    return gt_loss(z,label,valid)+.25*macro((bce*m).sum((-2,-1))/m.sum((-2,-1)).clamp_min(1),m)

def trusted_kl(logits,teacher,label,valid):
    z=logits.double() if logits.dtype==torch.float64 else logits.float()
    q=teacher.detach().to(z.dtype);y,m=masks(z,label,valid)
    trusted=(q>.5)==y
    kl=(F.binary_cross_entropy_with_logits(z,q,reduction='none')+torch.xlogy(q,q)+torch.xlogy(1-q,1-q)).clamp_min(0)
    return macro((kl*m*trusted).sum((-2,-1))/m.sum((-2,-1)).clamp_min(1),m)


def balanced_bce(logits,label,valid):
    z=logits.double() if logits.dtype==torch.float64 else logits.float()
    y,m=masks(z,label,valid);pos=y&m;neg=(~y)&m
    dims=(-2,-1)
    positive=(F.softplus(-z)*pos).sum(dims)/pos.sum(dims).clamp_min(1)
    negative=(F.softplus(z)*neg).sum(dims)/neg.sum(dims).clamp_min(1)
    # Full GT group denominators. Missing group stays zero, no renormalization.
    return macro(.5*(positive+negative),m)


def stratified_teacher_kl(logits,teacher,label,valid):
    z=logits.double() if logits.dtype==torch.float64 else logits.float()
    q=teacher.detach().to(z.dtype);y,m=masks(z,label,valid)
    trusted=(q>.5)==y;pos=y&m;neg=(~y)&m;dims=(-2,-1)
    kl=(F.binary_cross_entropy_with_logits(z,q,reduction='none')+torch.xlogy(q,q)+torch.xlogy(1-q,1-q)).clamp_min(0)
    positive=(kl*trusted*pos).sum(dims)/pos.sum(dims).clamp_min(1)
    negative=(kl*trusted*neg).sum(dims)/neg.sum(dims).clamp_min(1)
    return macro(.5*(positive+negative),m)
