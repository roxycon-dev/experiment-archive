_base_ = './e4v2_detail_split613.py'

# ============================================================
# exp/combo2_split613.py — COMBO-2: pure-dice framework stack.
# E4v2 (detail branch + direct aux loss) + E2E3 (learnable
# LR/HR logit fusion + ASPP-lite context in decoder).
# No CE+Dice (twice shown negative on test). All components
# keep the baseline dice loss; E2E3's val trajectory showed
# no harm (38.3-39.2%), E4v2 targets the MA precision problem.
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
    ),
)
