_base_ = './hrplus_patch_dice.py'

# P1: 语义条件 MA 残差头（冻结基准主干，只训头）
# 相对 hrplus_patch_dice.py 的唯一变化 = model.type 换成 HRPlusMAResidual，
# 外加「只训头」所需的学习率/步数调整（主干冻结，头是新建的，需要更大 lr）。
custom_imports = dict(imports=['hrplus_patch', 'ma_residual'],
                      allow_failed_imports=False)

model = dict(
    type='HRPlusMAResidual',
    source_checkpoint='weights/b_weights.pth',
    ma_hard_weight=0.0,       # 第一版按 R1 设计只用 Dice 主目标
    lovasz_weight=0.0,
)

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
)

# 只训头：lr 比端到端微调大一个量级
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict(grad_clip=dict(max_norm=5.0, norm_type=2))
lr_config = dict(policy='poly', warmup='linear', warmup_iters=300,
                 warmup_ratio=0.1, power=1.0, min_lr=0.0001, by_epoch=False)
runner = dict(type='IterBasedRunner', max_iters=6000)
# 头很小，密集验证成本可接受；interval 就是步数（IterBasedRunner，by_epoch=False）
checkpoint_config = dict(by_epoch=False, interval=1000, max_keep_ckpts=2)
evaluation = dict(interval=500, metric='mIoU', priority='LOW', save_best='mIoU')
log_config = dict(interval=20,
                  hooks=[dict(type='TextLoggerHook', by_epoch=False)])
seed = 42
