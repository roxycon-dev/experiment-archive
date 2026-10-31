_base_ = './baseline_split613.py'

# ============================================================
# exp/e4v2_detail_split613.py — E4-v2: detail branch WITH direct
# auxiliary loss on the detail head.
# E4 (detail branch only) raised MA recall (+5.8pp) but hurt
# precision (-7.7pp): the detail head is only trained indirectly
# through the fused loss, so its logits are noisy. E4-v2 adds a
# direct dice loss (weight 0.3) on the upsampled detail logits,
# training the detail head to be precise on its own.
# Hypothesis: direct supervision fixes the FP problem -> MA
# precision/IoU/AUPR improve without losing the recall gain.
# ============================================================

model = dict(
    hr_settings=dict(
        detail_loss_weight=0.3,
    ),
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
