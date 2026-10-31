# ASPP-lite context module for FCNHead (E3, exp/improve)
# Reference: DeepLabV3 ASPP (Chen et al., CVPR 2017) — light adaptation
# for the HRDecoder simple-CNN decoder to add multi-scale context.

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from ..builder import NECKS


@NECKS.register_module()
class ASPPContext(nn.Module):
    """Multi-scale dilated context module.

    Args:
        in_channels (int): input channels.
        out_channels (int): output channels (default 64).
        dilations (tuple): dilation rates for the 3x3 branches.
        norm_cfg / act_cfg / conv_cfg: passed to ConvModule.
    """

    def __init__(self,
                 in_channels,
                 out_channels=64,
                 dilations=(6, 12, 18),
                 norm_cfg=None,
                 act_cfg=None,
                 conv_cfg=None):
        super(ASPPContext, self).__init__()
        n_branches = len(dilations) + 2  # 1x1 + dilated 3x3s + image pooling
        branch_ch = max(8, out_channels // (2 * n_branches)) * 2
        self.branches = nn.ModuleList()
        # 1x1 branch
        self.branches.append(
            ConvModule(in_channels, branch_ch, 1, conv_cfg=conv_cfg,
                       norm_cfg=norm_cfg, act_cfg=act_cfg))
        # dilated 3x3 branches
        for d in dilations:
            self.branches.append(
                ConvModule(in_channels, branch_ch, 3, padding=d, dilation=d,
                           conv_cfg=conv_cfg, norm_cfg=norm_cfg,
                           act_cfg=act_cfg))
        # image pooling branch (no BN: 1x1 spatial map is batch-size sensitive)
        self.branches.append(
            ConvModule(in_channels, branch_ch, 1, conv_cfg=conv_cfg,
                       norm_cfg=None, act_cfg=act_cfg))
        self.pool = nn.AdaptiveAvgPool2d(1)
        # fuse branches
        self.fuse = ConvModule(branch_ch * n_branches, out_channels, 1,
                               conv_cfg=conv_cfg, norm_cfg=norm_cfg,
                               act_cfg=act_cfg)

    def forward(self, x):
        feats = [b(x) for b in self.branches[:-1]]
        pooled = self.branches[-1](self.pool(x))
        pooled = F.interpolate(pooled, size=x.shape[2:], mode='bilinear',
                               align_corners=False)
        feats.append(pooled)
        return self.fuse(torch.cat(feats, dim=1))
