# E4: DetailBranch + DetailHead (exp/improve)
# A shallow full-resolution feature path (BiSeNet-style detail branch)
# that preserves MA-scale detail lost by the 2x downscale in the LR path
# (HRDecoder resizes the 2048px image to 1024px before the backbone, so
# the native logit grid is 1/8 of the image and the median MA lesion
# (48 px^2) is below one logit cell).

import torch.nn as nn
from mmcv.cnn import ConvModule

from ..builder import HEADS, NECKS


@NECKS.register_module()
class DetailBranch(nn.Module):
    """Stride-4 shallow stem on the ORIGINAL-resolution image.

    Args:
        in_channels (int): image channels (3).
        mid_channels (int): hidden channels (32).
        out_channels (int): output channels (32).
        norm_cfg / act_cfg / conv_cfg: passed to ConvModule.
    """

    def __init__(self,
                 in_channels=3,
                 mid_channels=32,
                 out_channels=32,
                 norm_cfg=None,
                 act_cfg=None,
                 conv_cfg=None):
        super(DetailBranch, self).__init__()
        self.stem = nn.Sequential(
            ConvModule(in_channels, mid_channels, 3, stride=2, padding=1,
                       conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg),
            ConvModule(mid_channels, mid_channels, 3, stride=2, padding=1,
                       conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg),
            ConvModule(mid_channels, out_channels, 3, stride=1, padding=1,
                       conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg),
        )

    def forward(self, x):
        return self.stem(x)


@HEADS.register_module()
class DetailHead(nn.Module):
    """Tiny decoder for the detail branch -> per-class logits at stride 4."""

    def __init__(self,
                 in_channels=32,
                 mid_channels=32,
                 num_classes=4,
                 norm_cfg=None,
                 act_cfg=None,
                 conv_cfg=None):
        super(DetailHead, self).__init__()
        self.convs = nn.Sequential(
            ConvModule(in_channels, mid_channels, 3, padding=1,
                       conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg),
            ConvModule(mid_channels, mid_channels, 3, padding=1,
                       conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg),
        )
        self.cls = nn.Conv2d(mid_channels, num_classes, 1)

    def forward(self, x):
        return self.cls(self.convs(x))
