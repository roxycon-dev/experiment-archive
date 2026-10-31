"""FP32 supervision with gradients for saturated false negatives and positives."""
from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def target_masks(label, valid=None):
    if label.ndim == 4:
        label = label[:, 0]
    if label.ndim != 3:
        raise ValueError("Expected BHW labels")
    legal = (label >= 0) & (label <= 4) | (label == 255)
    if not bool(legal.all()):
        raise ValueError("Labels must be 0=BG,1=EX,2=HE,3=SE,4=MA,255=ignore")
    mask = label != 255
    if valid is not None:
        if valid.ndim == 4:
            valid = valid[:, 0]
        mask = mask & valid.bool()
    target = torch.stack([label == c for c in range(1, 5)], 1)
    return target, mask[:, None].expand_as(target)


class SparseLesionLoss(nn.Module):
    def __init__(self, negative_ratio=3, min_negatives=1024, bce_weight=0.5,
                 dense_weight=0.05, auxiliary_weight=0.2):
        super().__init__()
        self.negative_ratio = negative_ratio
        self.min_negatives = min_negatives
        self.bce_weight = bce_weight
        self.dense_weight = dense_weight
        self.auxiliary_weight = auxiliary_weight

    def one_head(self, logits, label, valid=None):
        z = logits if logits.dtype == torch.float64 else logits.float()
        target, mask = target_masks(label, valid)
        if z.shape != target.shape:
            raise ValueError("Loss is evaluated on the full label grid")
        if not bool(torch.isfinite(z).all()):
            raise FloatingPointError("Nonfinite logits")
        values, bces, dices, dense_values = [], [], [], []
        for b in range(z.shape[0]):
            for c in range(4):
                v = z[b, c][mask[b, c]]
                y = target[b, c][mask[b, c]]
                if v.numel() == 0:
                    continue
                positives, negatives = v[y], v[~y]
                positive_loss = F.softplus(-positives).mean() if positives.numel() else v.sum() * 0
                k = min(negatives.numel(), max(self.min_negatives,
                                              self.negative_ratio * positives.numel()))
                # OHEM acts only on background. ALL labelled positives remain.
                negative_loss = (F.softplus(negatives.topk(k, sorted=False).values).mean()
                                 if k else v.sum() * 0)
                # Present groups share mass equally. An empty positive group
                # leaves the complete background loss, not a spurious Dice=1.
                groups = int(positives.numel() > 0) + int(negatives.numel() > 0)
                bce = (positive_loss + negative_loss) / groups
                p = v.sigmoid()
                dice = (1 - (2 * p[y].sum() + 1e-5) /
                        (p.square().sum() + y.sum() + 1e-5)) if positives.numel() else v.sum() * 0
                dense = F.binary_cross_entropy_with_logits(v, y.to(v.dtype))
                values.append(dice + self.bce_weight * bce + self.dense_weight * dense)
                bces.append(bce.detach()); dices.append(dice.detach()); dense_values.append(dense.detach())
        if not values:
            raise ValueError("Entire batch is unannotated; do not consume an optimizer update")
        return torch.stack(values).mean(), {"ohem_bce": torch.stack(bces).mean(),
                                            "squared_dice": torch.stack(dices).mean(),
                                            "dense_bce": torch.stack(dense_values).mean()}

    def forward(self, output, label, valid=None):
        # Recompute pixel-loss intermediates instead of retaining two complete
        # 2048-grid sigmoid/BCE graphs for the main and auxiliary predictions.
        if torch.is_grad_enabled():
            main, metrics = checkpoint(self.one_head, output["logits"], label, valid, use_reentrant=False)
        else:
            main, metrics = self.one_head(output["logits"], label, valid)
        auxiliary = F.interpolate(output["auxiliary_logits"].float(),
                                   size=output["logits"].shape[-2:], mode="bilinear", align_corners=False)
        if torch.is_grad_enabled():
            aux, _ = checkpoint(self.one_head, auxiliary, label, valid, use_reentrant=False)
        else:
            aux, _ = self.one_head(auxiliary, label, valid)
        total = main + self.auxiliary_weight * aux
        return total, {**metrics, "main_loss": main.detach(), "aux_loss": aux.detach(),
                        "loss": total.detach()}

