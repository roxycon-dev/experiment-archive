"""Same validated GT/KL formulas with vectorized reductions and no hot-path host checks."""
import torch
import torch.nn.functional as F

def masks(logits, label, valid):
    labels = label[:, 0] if label.ndim == 4 else label
    y = torch.stack([labels == c for c in range(1,5)], 1)
    if valid.ndim == 3: valid = valid[:,None]
    m = valid.bool().expand_as(y) & (labels[:,None] != 255)
    return y, m

def macro(values, mask):
    active = mask.sum((-2,-1)) > 0
    return (values*active).sum()/active.sum().clamp_min(1)

def gt_loss(logits, label, valid):
    z = logits.double() if logits.dtype == torch.float64 else logits.float()
    y, m = masks(z,label,valid)
    pos, neg = m & y, m & ~y
    np, nn = pos.sum((-2,-1)), neg.sum((-2,-1))
    p = z.sigmoid()
    dice = 1-(2*(p*pos).sum((-2,-1))+1e-5)/((p.square()*m).sum((-2,-1))+np+1e-5)
    bpos = (F.softplus(-z)*pos).sum((-2,-1))/np.clamp_min(1)
    bneg = (F.softplus(z)*neg).sum((-2,-1))/nn.clamp_min(1)
    return macro(dice*(np>0)+.25*(bpos+bneg),m)

def trusted_kl(logits, teacher, label, valid):
    z = logits.double() if logits.dtype == torch.float64 else logits.float()
    q = teacher.detach().to(z.dtype)
    y, m = masks(z,label,valid)
    trusted = (q > .5) == y
    kl = (F.binary_cross_entropy_with_logits(z,q,reduction='none') + torch.xlogy(q,q) + torch.xlogy(1-q,1-q)).clamp_min(0)
    return macro((kl*m*trusted).sum((-2,-1))/m.sum((-2,-1)).clamp_min(1),m)
