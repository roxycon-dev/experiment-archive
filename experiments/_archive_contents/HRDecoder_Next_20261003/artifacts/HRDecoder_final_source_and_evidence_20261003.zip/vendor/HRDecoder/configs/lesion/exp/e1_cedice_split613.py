_base_ = './baseline_split613.py'

# ============================================================
# exp/e1_cedice_split613.py — E1-v2: CE + Dice combined loss.
# (Tversky alpha=0.3 and 0.7 both collapsed to all-positive at
# iter 4000 — recorded in EXPERIMENT_LOG; mechanism: Tversky's
# numerator TP (vs Dice's 2TP) halves/weakens the background
# suppression gradient, so the model cannot escape the
# all-positive attractor.)
# CE+Dice is the codebase-native combination
# (binary_ce_dice_loss): CE gives strong per-pixel gradients
# (background suppression + score calibration -> helps AUPR),
# Dice keeps the class-sum structure of the original loss.
# Hypothesis: stronger background suppression improves MA/HE
# precision-recall balance without the Tversky collapse.
# ============================================================

model = dict(
    decode_head=dict(
        loss_decode=dict(
            type='BinaryLoss',
            loss_type='ce_dice',
            loss_weight=1.0,
            smooth=1e-5,
        )
    )
)
