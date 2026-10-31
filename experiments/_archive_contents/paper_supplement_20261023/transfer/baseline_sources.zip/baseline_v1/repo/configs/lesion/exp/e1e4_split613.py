_base_ = './baseline_split613.py'

# ============================================================
# exp/e1e4_split613.py — E1 (Tversky alpha=0.7, FN-penalizing)
# + E4 (full-res detail branch). Both target the MA bottleneck
# (recall via loss surface, resolution via detail features).
# Used for the stacked ablation row (E1+E4).
# ============================================================

model = dict(
    decode_head=dict(
        loss_decode=dict(
            type='BinaryLoss',
            loss_type='tversky',
            loss_weight=1.0,
            smooth=1e-5,
            loss_hyper=dict(alpha=0.7),
        )
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
