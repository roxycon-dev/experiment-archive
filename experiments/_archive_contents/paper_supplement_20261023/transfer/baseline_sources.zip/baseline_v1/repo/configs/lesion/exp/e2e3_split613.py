_base_ = './baseline_split613.py'

# ============================================================
# exp/e2e3_split613.py — E2 (learnable LR/HR logit fusion) +
# E3 (ASPP-lite context in decoder). Orthogonal changes; used
# for the stacked ablation row (E2+E3) and as a building block
# for the final combination.
# ============================================================

model = dict(
    hr_settings=dict(
        fuse_mode='learnable',
    ),
    decode_head=dict(
        context_cfg=dict(
            type='ASPPContext',
            dilations=(6, 12, 18),
        )
    )
)
