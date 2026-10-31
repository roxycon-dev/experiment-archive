import torch
from torch import nn
from torch.nn import functional as F
from mmseg.models.builder import SEGMENTORS
from mmseg.models.segmentors.HRDecoder import HRDecoder


def block(cin, cout, stride=1):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride, 1, bias=False),
                         nn.GroupNorm(8, cout), nn.ReLU(inplace=True))


class SemanticMAResidual(nn.Module):
    def __init__(self):
        super().__init__()
        self.detail = nn.Sequential(block(3, 16, 2), block(16, 32, 2))
        self.semantic = nn.Sequential(nn.Conv2d(720, 32, 1, bias=False),
                                      nn.GroupNorm(8, 32), nn.ReLU(inplace=True))
        self.fusion = nn.Sequential(block(64, 32), block(32, 32), nn.Conv2d(32, 1, 1))
        nn.init.zeros_(self.fusion[-1].weight)
        nn.init.zeros_(self.fusion[-1].bias)

    def forward(self, image, semantics):
        detail = self.detail(image)
        semantic = F.interpolate(self.semantic(semantics), size=detail.shape[-2:],
                                 mode='bilinear', align_corners=False)
        residual = self.fusion(torch.cat([detail, semantic], 1))
        return F.interpolate(residual, size=image.shape[-2:], mode='bilinear',
                             align_corners=False)


def ma_hard_loss(logits, target):
    """MA id=4; keep all positive pixels and a bounded set of hard negatives."""
    target = target.squeeze(1)
    losses = []
    for score, label in zip(logits[:, 3], target):
        positive = score[label == 4]
        negative = score[(label != 4) & (label != 255)]
        count = min(negative.numel(), max(256, 3 * positive.numel()), 65536)
        pos_loss = F.softplus(-positive).mean() if positive.numel() else score.sum() * 0
        neg_loss = F.softplus(negative.topk(count, sorted=False).values).mean() if count else score.sum() * 0
        losses.append((pos_loss + neg_loss) / (2 if positive.numel() else 1))
    return torch.stack(losses).mean()


@SEGMENTORS.register_module()
class HR100Refinement(HRDecoder):
    def __init__(self, semantic_refinement=True, hard_weight=0.02, freeze_bn=True, **kwargs):
        super().__init__(**kwargs)
        self.ma_refiner = SemanticMAResidual() if semantic_refinement else None
        self.hard_weight = hard_weight
        self.freeze_bn = freeze_bn

    def train(self, mode=True):
        super().train(mode)
        if mode and self.freeze_bn:
            for module in self.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    module.eval()
        return self

    def refine_logits(self, logits, img, semantics):
        if self.ma_refiner is None:
            return logits
        ma = logits[:, 3:4] + self.ma_refiner(img, semantics)
        return torch.cat([logits[:, :3], ma], dim=1)

    def refinement_losses(self, logits, target):
        return {'loss_ma_hard': self.hard_weight * ma_hard_loss(logits, target)}
