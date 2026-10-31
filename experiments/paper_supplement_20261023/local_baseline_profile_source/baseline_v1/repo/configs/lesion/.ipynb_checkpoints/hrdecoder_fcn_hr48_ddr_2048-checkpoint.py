_base_ = [
    '../_base_/models/hrdecoder_fcn_hr48.py',
    '../_base_/datasets/hr_ddr_2048.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/sgd.py',
    '../_base_/schedules/poly10warm.py',
]

model = dict(
    type= 'HRDecoder',
    hr_settings=dict(
        hr_scale = (1024,1024),
        scale_ratio = (0.75, 1.25),
        divisible = 8,
        lr_loss_weight = 0,
        hr_loss_weight = 0.1,
        fuse_mode = 'simple',
        crop_num = 4,
    ),
    # 👇 强行覆盖底层继承来的配置，将 backbone 中的归一化层全部改为单卡 BN
    backbone=dict(
        norm_cfg=dict(type='BN', requires_grad=True)
    ),
    # 👇 强行覆盖底层继承来的配置，将 decode_head 中的归一化层全部改为单卡 BN
    decode_head=dict(
        norm_cfg=dict(type='BN', requires_grad=True)
    )
)

# ==================== 单卡优化配置 ====================
# 方案 A（极力推荐）：如果 4090 显存允许，单卡 samples_per_gpu 设为 2，学习率 lr 设为 0.005
data=dict(
    samples_per_gpu=2, 
    workers_per_gpu=2,
)
optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0005)

# 方案 B（若报错显存溢出使用）：如果显存不足，samples_per_gpu 设为 1，学习率 lr 设为 0.0025
# data=dict(
#     samples_per_gpu=1, 
#     workers_per_gpu=1,
# )
# optimizer = dict(type='SGD', lr=0.0025, momentum=0.9, weight_decay=0.0005)
# ====================================================

seed = 14
runner = dict(type='IterBasedRunner', max_iters=40000)
checkpoint_config = dict(by_epoch=False, interval=1000, max_keep_ckpts=1)
evaluation = dict(interval=4000, 
                  metric='mIoU',
                  priority='LOW',
                  save_best='mIoU')