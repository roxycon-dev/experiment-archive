_base_ = './final_combo_split613.py'

# ============================================================
# exp/combo_v2_split613.py — COMBO-v2: E4 detail branch +
# CE+Dice + direct auxiliary loss on the detail head.
# Extends the current best candidate (E4+CE+Dice) with the
# E4v2 fix for the detail path's precision problem.
# ============================================================

model = dict(
    hr_settings=dict(
        detail_loss_weight=0.3,
    ),
)
