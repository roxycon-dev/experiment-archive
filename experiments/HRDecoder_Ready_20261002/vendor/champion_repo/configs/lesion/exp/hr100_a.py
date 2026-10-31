_base_ = './e4_detail_split613.py'
custom_imports = dict(imports=['hr100_refinement'], allow_failed_imports=False)
model = dict(type='HR100Refinement', pretrained=None, backbone=dict(pretrained=None),
             semantic_refinement=True, hard_weight=0.02, freeze_bn=True,
             test_cfg=dict(compute_aupr=True))
data = dict(workers_per_gpu=4,
    train=dict(data_root='/root/autodl-tmp/hr100/data/DDR_split_613'),
    val=dict(data_root='/root/autodl-tmp/hr100/data/DDR_split_613'),
    test=dict(data_root='/root/autodl-tmp/hr100/data/DDR_split_613'))
load_from = '/root/HRDecoder/work_dirs/exp_e4_detail_split613/best_mIoU_iter_36000.pth'
resume_from = None
optimizer = dict(type='SGD', lr=0.0005, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict(grad_clip=dict(max_norm=5.0, norm_type=2))
lr_config = dict(policy='poly', warmup='linear', warmup_iters=200, warmup_ratio=0.1, power=1.0, min_lr=0.00005, by_epoch=False)
runner = dict(type='IterBasedRunner', max_iters=8000)
checkpoint_config = dict(by_epoch=False, interval=500, max_keep_ckpts=2)
evaluation = dict(interval=1000, metric='mIoU', priority='LOW', save_best='mIoU')
log_config = dict(interval=20, hooks=[dict(type='TextLoggerHook', by_epoch=False)])
seed = 42

