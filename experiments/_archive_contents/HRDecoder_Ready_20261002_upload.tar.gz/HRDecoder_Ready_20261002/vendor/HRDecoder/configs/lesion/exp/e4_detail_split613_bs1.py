_base_ = './e4_detail_split613.py'

# Fallback if E4 OOMs at bs=2: single sample per GPU with halved lr
# (same compromise documented in the repo's 方案 B comment).
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
)
optimizer = dict(type='SGD', lr=0.0025, momentum=0.9, weight_decay=0.0005)
