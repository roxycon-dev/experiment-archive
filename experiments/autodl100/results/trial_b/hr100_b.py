norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='HR100Refinement',
    use_sigmoid=True,
    pretrained=None,
    backbone=dict(
        type='HRNet',
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        extra=dict(
            stage1=dict(
                num_modules=1,
                num_branches=1,
                block='BOTTLENECK',
                num_blocks=(4, ),
                num_channels=(64, )),
            stage2=dict(
                num_modules=1,
                num_branches=2,
                block='BASIC',
                num_blocks=(4, 4),
                num_channels=(48, 96)),
            stage3=dict(
                num_modules=4,
                num_branches=3,
                block='BASIC',
                num_blocks=(4, 4, 4),
                num_channels=(48, 96, 192)),
            stage4=dict(
                num_modules=3,
                num_branches=4,
                block='BASIC',
                num_blocks=(4, 4, 4, 4),
                num_channels=(48, 96, 192, 384))),
        pretrained=None),
    hr_settings=dict(
        hr_scale=(1024, 1024),
        scale_ratio=(0.75, 1.25),
        divisible=8,
        lr_loss_weight=0,
        hr_loss_weight=0.1,
        fuse_mode='simple',
        crop_num=4),
    decode_head=dict(
        type='FCNHead',
        in_channels=720,
        in_index=0,
        channels=64,
        kernel_size=7,
        num_convs=1,
        compress=True,
        concat_input=False,
        dropout_ratio=-1,
        num_classes=4,
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='BinaryLoss', loss_type='dice', loss_weight=1.0,
            smooth=1e-05)),
    train_cfg=dict(work_dir='/root/autodl-tmp/hr100/trial_b'),
    test_cfg=dict(mode='whole', compute_aupr=True),
    detail_branch=dict(
        type='DetailBranch',
        mid_channels=32,
        out_channels=32,
        norm_cfg=dict(type='BN', requires_grad=True),
        act_cfg=dict(type='ReLU')),
    detail_head=dict(
        type='DetailHead',
        in_channels=32,
        mid_channels=32,
        num_classes=4,
        norm_cfg=dict(type='BN', requires_grad=True),
        act_cfg=dict(type='ReLU')),
    semantic_refinement=False,
    hard_weight=0.02,
    freeze_bn=True)
dataset_type = 'LesionDataset'
data_root = './data/DDR'
img_norm_cfg = dict(
    mean=[81.205, 50.636, 21.216], std=[76.252, 48.798, 21.625], to_rgb=True)
image_scale = (2048, 2048)
crop_size = (2048, 2048)
palette = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128]]
classes = ['bg', 'EX', 'HE', 'SE', 'MA']
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(2048, 2048), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=(2048, 2048), cat_max_ratio=0.75),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='RandomRotate',
        prob=1.0,
        pad_val=0,
        seg_pad_val=0,
        degree=(-45, 45),
        auto_bound=False),
    dict(
        type='Normalize',
        mean=[81.205, 50.636, 21.216],
        std=[76.252, 48.798, 21.625],
        to_rgb=True),
    dict(type='Pad', size=(2048, 2048), pad_val=0, seg_pad_val=0),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2048, 2048),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),
            dict(
                type='Normalize',
                mean=[81.205, 50.636, 21.216],
                std=[76.252, 48.798, 21.625],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        img_dir='images/train',
        ann_dir='labels/train',
        data_root='/root/autodl-tmp/hr100/data/DDR_split_613',
        classes=['bg', 'EX', 'HE', 'SE', 'MA'],
        palette=[[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                 [0, 0, 128]],
        type='LesionDataset',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(
                type='Resize', img_scale=(2048, 2048), ratio_range=(0.5, 2.0)),
            dict(
                type='RandomCrop', crop_size=(2048, 2048), cat_max_ratio=0.75),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(
                type='RandomRotate',
                prob=1.0,
                pad_val=0,
                seg_pad_val=0,
                degree=(-45, 45),
                auto_bound=False),
            dict(
                type='Normalize',
                mean=[81.205, 50.636, 21.216],
                std=[76.252, 48.798, 21.625],
                to_rgb=True),
            dict(type='Pad', size=(2048, 2048), pad_val=0, seg_pad_val=0),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_semantic_seg'])
        ]),
    val=dict(
        img_dir='images/val',
        ann_dir='labels/val',
        data_root='/root/autodl-tmp/hr100/data/DDR_split_613',
        classes=['bg', 'EX', 'HE', 'SE', 'MA'],
        palette=[[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                 [0, 0, 128]],
        type='LesionDataset',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(2048, 2048),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=False),
                    dict(
                        type='Normalize',
                        mean=[81.205, 50.636, 21.216],
                        std=[76.252, 48.798, 21.625],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        img_dir='images/test',
        ann_dir='labels/test',
        data_root='/root/autodl-tmp/hr100/data/DDR_split_613',
        classes=['bg', 'EX', 'HE', 'SE', 'MA'],
        palette=[[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                 [0, 0, 128]],
        type='LesionDataset',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(2048, 2048),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=False),
                    dict(
                        type='Normalize',
                        mean=[81.205, 50.636, 21.216],
                        std=[76.252, 48.798, 21.625],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]))
log_config = dict(
    interval=20, hooks=[dict(type='TextLoggerHook', by_epoch=False)])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = '/root/HRDecoder/work_dirs/exp_e4_detail_split613/best_mIoU_iter_36000.pth'
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
optimizer = dict(type='SGD', lr=0.0005, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict(grad_clip=dict(max_norm=5.0, norm_type=2))
lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=200,
    warmup_ratio=0.1,
    power=1.0,
    min_lr=5e-05,
    by_epoch=False)
seed = 42
runner = dict(type='IterBasedRunner', max_iters=8000)
checkpoint_config = dict(by_epoch=False, interval=500, max_keep_ckpts=2)
evaluation = dict(
    interval=1000, metric='mIoU', priority='LOW', save_best='mIoU')
custom_imports = dict(imports=['hr100_refinement'], allow_failed_imports=False)
work_dir = '/root/autodl-tmp/hr100/trial_b'
gpu_ids = range(0, 1)
