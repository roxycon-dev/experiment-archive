import torch
import torch.nn.functional as F


def masked_loss(logits, target, valid):
    z, g, v = logits.float(), target.float(), valid.float()
    if z.shape != g.shape or g.shape != v.shape or z.shape[1] != 4:
        raise ValueError('Four-channel center loss shape mismatch')
    # Neutralize invalid values before BCE/sigmoid; invalid NaN must not leak.
    z = torch.where(v.bool(), z, torch.zeros_like(z))
    g = torch.where(v.bool(), g, torch.zeros_like(g))
    axes = (0, 2, 3)
    n = v.sum(axes)
    selected = n > 0
    if not selected.any():
        raise ValueError('All-invalid micro-batch')
    p = z.sigmoid()
    bce = (v * F.binary_cross_entropy_with_logits(z, g, reduction='none')).sum(axes) / n.clamp_min(1)
    dice = 1 - (2*(v*p*g).sum(axes)+1) / ((v*p).sum(axes)+(v*g).sum(axes)+1)
    value = .5*bce[selected].mean()+.5*dice[selected].mean()
    if not torch.isfinite(value):
        raise FloatingPointError('Nonfinite masked BCE+Dice')
    return value
