_base_ = './baseline_split613.py'

# ============================================================
# exp/e3_context_split613.py — E3: ASPP-lite context module in
# the FCN decoder head (multi-scale dilated context after the
# local convs). Addresses the paper's admitted weakness: the
# simple CNN decoder lacks context modeling.
# Hypothesis: EX/HE (large lesions) benefit from broader context,
# reducing boundary over/under-segmentation and false positives.
# ============================================================

model = dict(
    decode_head=dict(
        context_cfg=dict(
            type='ASPPContext',
            dilations=(6, 12, 18),
        )
    )
)
