"""Locked v3 FP32 loss, with full-global-batch denominators for accumulation."""
from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def target_masks(label, valid=None):
    if label.ndim == 4 and label.shape[1] == 1:
        label = label[:, 0]
    if label.ndim != 3 or label.dtype.is_floating_point:
        raise ValueError("Expected integer BHW labels")
    legal = ((label >= 0) & (label <= 4)) | (label == 255)
    if not bool(legal.all()):
        raise ValueError("Labels must be 0=BG,1=EX,2=HE,3=SE,4=MA,255=ignore")
    mask = label != 255
    if valid is not None:
        if valid.ndim == 4 and valid.shape[1] == 1:
            valid = valid[:, 0]
        if valid.shape != label.shape:
            raise ValueError("Validity mask must match label geometry")
        mask = mask & valid.bool()
    target = torch.stack([label == c for c in range(1, 5)], 1)
    return target, mask[:, None].expand_as(target)


def stable_hard_negative_indices(negative_logits, k):
    """Descending logits, then earliest flattened index at the cutoff.

    Inputs already follow ascending valid-negative flattened order. A cutoff
    avoids sorting every background pixel, and specifies tie membership
    independently of the device's topk tie order.
    """
    if k == 0:
        return torch.empty(0, dtype=torch.long, device=negative_logits.device)
    if not 0 < k <= negative_logits.numel():
        raise ValueError("Invalid negative count")
    values = negative_logits.detach()
    threshold = values.topk(k, sorted=False).values.min()
    higher = torch.nonzero(values > threshold, as_tuple=False).flatten()
    tied = torch.nonzero(values == threshold, as_tuple=False).flatten()
    selected = torch.cat((higher, tied[:k - higher.numel()]))
    selected = selected.sort().values
    return selected[torch.argsort(values[selected], descending=True, stable=True)]


class SparseLesionLoss(nn.Module):
    def __init__(self, negative_ratio=3, min_negatives=1024,
                 positive_weight=0.25, negative_weight=0.25,
                 dense_weight=0.05, auxiliary_weight=0.2):
        super().__init__()
        self.negative_ratio = int(negative_ratio)
        self.min_negatives = int(min_negatives)
        self.positive_weight = float(positive_weight)
        self.negative_weight = float(negative_weight)
        self.dense_weight = float(dense_weight)
        self.auxiliary_weight = float(auxiliary_weight)
        if self.negative_ratio < 0 or self.min_negatives < 0:
            raise ValueError("Negative mining counts must be nonnegative")

    @torch.no_grad()
    def batch_denominators(self, label, valid=None):
        target, mask = target_masks(label, valid)
        positive = (target & mask).sum((2, 3))
        valid_per_image = mask[:, 0].sum((1, 2))
        negative = valid_per_image[:, None] - positive
        selected = torch.minimum(negative, torch.maximum(
            torch.full_like(positive, self.min_negatives), self.negative_ratio * positive))
        valid_pixels = valid_per_image.sum()
        if int(valid_pixels) == 0:
            raise ValueError("Entire global batch is unannotated; reject optimizer update")
        return {"positive": positive.sum(0), "selected_negative": selected.sum(0),
                "positive_images": (positive > 0).sum(0), "valid_pixels": valid_pixels,
                "supervision_positive_per_image": positive}

    def one_head(self, logits, label, valid=None, denominators=None):
        z = logits.float()
        target, mask = target_masks(label, valid)
        if z.shape != target.shape:
            raise ValueError("Loss must use full label grid and four independent logits")
        if not bool(torch.isfinite(z).all()):
            raise FloatingPointError("Nonfinite logits")
        if denominators is None:
            denominators = self.batch_denominators(label, valid)
        d = {key: denominators[key].to(device=z.device, dtype=torch.float32).clamp_min(1)
             for key in ("positive", "selected_negative", "positive_images", "valid_pixels")}
        if int(denominators["valid_pixels"]) <= 0:
            raise ValueError("Invalid global batch denominators")
        # Graph-connected empty reduction cannot overflow on ignored logits.
        zero = z[:, :, :0, :].sum()
        pos_loss, neg_loss, dice_loss, dense_loss = zero, zero, zero, zero
        for b in range(z.shape[0]):
            for c in range(4):
                v = z[b, c][mask[b, c]]
                y = target[b, c][mask[b, c]]
                if not v.numel():
                    continue
                positives, negatives = v[y], v[~y]
                pos_loss = pos_loss + F.softplus(-positives).sum() / d["positive"][c] / 4
                k = min(negatives.numel(), max(self.min_negatives,
                                               self.negative_ratio * positives.numel()))
                chosen = stable_hard_negative_indices(negatives, k)
                neg_loss = neg_loss + F.softplus(negatives[chosen]).sum() / d["selected_negative"][c] / 4
                if positives.numel():
                    p = v.sigmoid()
                    dice = 1 - (2 * p[y].sum() + 1e-5) / (p.square().sum() + y.sum() + 1e-5)
                    dice_loss = dice_loss + dice / d["positive_images"][c] / 4
                dense_loss = dense_loss + F.binary_cross_entropy_with_logits(
                    v, y.float(), reduction="sum") / d["valid_pixels"] / 4
        total = (dice_loss + self.positive_weight * pos_loss +
                 self.negative_weight * neg_loss + self.dense_weight * dense_loss)
        return total, {"positive_bce": pos_loss.detach(), "negative_bce": neg_loss.detach(),
                       "squared_dice": dice_loss.detach(), "dense_bce": dense_loss.detach()}

    def forward(self, output, label, valid=None, denominators=None):
        if denominators is None:
            denominators = self.batch_denominators(label, valid)
        def head(z):
            return self.one_head(z, label, valid, denominators)
        if torch.is_grad_enabled():
            main, metrics = checkpoint(head, output["logits"], use_reentrant=False)
        else:
            main, metrics = head(output["logits"])
        auxiliary = F.interpolate(output["auxiliary_logits"].float(), size=label.shape[-2:],
                                  mode="bilinear", align_corners=False)
        if torch.is_grad_enabled():
            aux, _ = checkpoint(head, auxiliary, use_reentrant=False)
        else:
            aux, _ = head(auxiliary)
        total = main + self.auxiliary_weight * aux
        return total, {**metrics, "main_loss": main.detach(), "aux_loss": aux.detach(),
                       "loss": total.detach()}
