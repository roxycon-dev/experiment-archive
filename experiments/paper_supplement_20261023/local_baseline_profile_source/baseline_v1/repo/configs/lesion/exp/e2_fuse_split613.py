_base_ = './baseline_split613.py'

# ============================================================
# exp/e2_fuse_split613.py — E2: learnable LR/HR logit fusion.
# The official HFF is a hardcoded element-wise average of LR and
# HR crop logits (fuse_mode is unused in upstream code). We make
# the fusion weights learnable per class (softmax-normalized,
# initialized at 0.5/0.5), applied identically in train & test.
# Hypothesis: the optimal LR/HR trade-off differs per class
# (e.g. HR crops help tiny MA more than large HE); a learned
# per-class balance improves both mIoU and mAUPR.
# ============================================================

model = dict(
    hr_settings=dict(
        fuse_mode='learnable',
    )
)
