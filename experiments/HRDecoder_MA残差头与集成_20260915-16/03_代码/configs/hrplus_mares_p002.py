_base_ = './hrplus_mares.py'

# LR 探针：lr=0.002，2000 步，每 250 步验证（头训练很快，密集验证成本可接受）
optimizer = dict(type='SGD', lr=0.002, momentum=0.9, weight_decay=0.0005)
lr_config = dict(policy='poly', warmup='linear', warmup_iters=200,
                 warmup_ratio=0.1, power=1.0, min_lr=1e-05, by_epoch=False)
runner = dict(type='IterBasedRunner', max_iters=2000)
checkpoint_config = dict(by_epoch=False, interval=1000, max_keep_ckpts=2)
evaluation = dict(interval=250, metric='mIoU', priority='LOW', save_best='mIoU')
