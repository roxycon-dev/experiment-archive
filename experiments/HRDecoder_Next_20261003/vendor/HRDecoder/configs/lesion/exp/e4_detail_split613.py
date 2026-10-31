_base_ = './baseline_split613.py'

# ============================================================
# exp/e4_detail_split613.py — E4: full-resolution detail path.
# Adds a shallow stride-4 DetailBranch on the ORIGINAL 2048px
# image (the backbone only sees a 1024px downscale) plus a tiny
# DetailHead; its logits are added to the fused LR+HR logits
# with a learnable scalar `detail_scale` initialized at 0
# (training starts exactly at baseline behavior).
# Hypothesis: MA (median 48 px^2, below one native logit cell
# of the LR path) is missed because the detail is destroyed by
# the 2x downscale; the detail path restores it, raising MA
# recall / AUPR without hurting larger classes.
# ============================================================

model = dict(
    detail_branch=dict(
        type='DetailBranch',
        mid_channels=32,
        out_channels=32,
        norm_cfg=dict(type='BN', requires_grad=True),
        act_cfg=dict(type='ReLU'),
    ),
    detail_head=dict(
        type='DetailHead',
        in_channels=32,
        mid_channels=32,
        num_classes=4,
        norm_cfg=dict(type='BN', requires_grad=True),
        act_cfg=dict(type='ReLU'),
    ),
)
