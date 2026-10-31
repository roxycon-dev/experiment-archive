_base_ = '../exp/hr100_a.py'

# HR+ patch (dice-only): student + frozen reference both strict-loaded from
# the B checkpoint inside the model constructor; load_from stays None.
custom_imports = dict(imports=['hrplus_patch'], allow_failed_imports=False)

img_norm_cfg = dict(
    mean=[81.205, 50.636, 21.216], std=[76.252, 48.798, 21.625], to_rgb=True)
image_scale = (2048, 2048)
crop_size = (1024, 1024)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=image_scale, keep_ratio=False),
    dict(type='RandomRotate', prob=1.0, pad_val=0, seg_pad_val=255,
         degree=(-45, 45), auto_bound=False),
    dict(type='ForegroundMixedCrop', crop_size=1024, fg_prob=0.5),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]

model = dict(
    type='HRPlusRefinement',
    pretrained=None,
    backbone=dict(pretrained=None),
    semantic_refinement=False,
    hard_weight=0,
    freeze_bn=True,
    source_checkpoint='weights/b_weights.pth',
    lovasz_weight=0.0,
    test_cfg=dict(compute_aupr=True))

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        data_root='./data/DDR_split_613',
        pipeline=train_pipeline),
    val=dict(data_root='./data/DDR_split_613'),
    test=dict(data_root='./data/DDR_split_613'))

# Both weights come from source_checkpoint strict-loaded in the constructor.
load_from = None
resume_from = None

optimizer = dict(type='SGD', lr=0.0005, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict(grad_clip=dict(max_norm=5.0, norm_type=2))
lr_config = dict(policy='poly', warmup='linear', warmup_iters=200,
                 warmup_ratio=0.1, power=1.0, min_lr=0.00005, by_epoch=False)
runner = dict(type='IterBasedRunner', max_iters=20000)
checkpoint_config = dict(by_epoch=False, interval=2000, max_keep_ckpts=2)
evaluation = dict(interval=2000, metric='mIoU', priority='LOW',
                  save_best='mIoU')
log_config = dict(interval=20,
                  hooks=[dict(type='TextLoggerHook', by_epoch=False)])
seed = 42
