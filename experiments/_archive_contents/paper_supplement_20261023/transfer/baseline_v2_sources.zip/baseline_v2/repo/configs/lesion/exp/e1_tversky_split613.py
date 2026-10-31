_base_ = './baseline_split613.py'

# ============================================================
# exp/e1_tversky_split613.py — E1: replace per-class Dice with
# Tversky loss. alpha weights FN (den = TP + alpha*FN + beta*FP,
# beta = 1-alpha). alpha=0.7 -> FN penalized more -> recall push.
# (First attempt used alpha=0.3 = FP-penalizing; that run
# collapsed to all-positive predictions at iter 4000 — recorded
# in EXPERIMENT_LOG. This alpha=0.7 variant is the recall-targeted
# setting; iter-4000 val is checked for the same collapse.)
# Hypothesis: MA/HE recall is limited by the balanced Dice
# surface; Tversky with alpha>beta reduces false negatives without
# the class-weight coupling problem that made 板块三 weighting fail.
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
    )
)
