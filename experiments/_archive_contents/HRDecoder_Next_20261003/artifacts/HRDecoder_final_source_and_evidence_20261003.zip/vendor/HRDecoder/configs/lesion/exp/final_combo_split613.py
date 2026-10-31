_base_ = './baseline_split613.py'

# ============================================================
# exp/final_combo_split613.py — FINAL combination (filled in after
# individual results): E4 detail branch + E1v2 CE+Dice
# (+ E2E3 if it wins standalone). Kept as a template that the
# researcher edits based on the ablation results.
# ============================================================

model = dict(
    decode_head=dict(
        loss_decode=dict(
            type='BinaryLoss',
            loss_type='ce_dice',
            loss_weight=1.0,
            smooth=1e-5,
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
