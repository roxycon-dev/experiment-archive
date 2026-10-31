_base_ = [
    '../_base_/models/hrdecoder_fcn_hr48.py',
    '../_base_/datasets/hr_ddr_2048.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/sgd.py',
    '../_base_/schedules/poly10warm.py',
]

# ==================== baseline config for DDR 6:1:3 split (板块一) ============
# Custom experimental split (not the original paper split):
#   data_root -> ./data/DDR_split_613  (train 454 / val 76 / test 227, seed 42)
# No data augmentation beyond the official pipeline, no copy-paste / sliding
# window oversampling, no class weighting -> this is the baseline for all
# 板块二 / 板块三 comparisons.

model = dict(
    type='HRDecoder',
    hr_settings=dict(
        hr_scale=(1024, 1024),
        scale_ratio=(0.75, 1.25),
        divisible=8,
        lr_loss_weight=0,
        hr_loss_weight=0.1,
        fuse_mode='simple',
        crop_num=4,
    ),
    # single-GPU BN overrides (same as the working single-GPU config)
    backbone=dict(
        norm_cfg=dict(type='BN', requires_grad=True)
    ),
    decode_head=dict(
        norm_cfg=dict(type='BN', requires_grad=True)
    )
)

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        data_root='./data/DDR_split_613',
        img_dir='images/train',
        ann_dir='labels/train'),
    val=dict(
        data_root='./data/DDR_split_613',
        img_dir='images/val',
        ann_dir='labels/val'),
    test=dict(
        data_root='./data/DDR_split_613',
        img_dir='images/test',
        ann_dir='labels/test'),
)

optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0005)

seed = 42
runner = dict(type='IterBasedRunner', max_iters=40000)
checkpoint_config = dict(by_epoch=False, interval=1000, max_keep_ckpts=1)
evaluation = dict(interval=4000, metric='mIoU', priority='LOW', save_best='mIoU')
