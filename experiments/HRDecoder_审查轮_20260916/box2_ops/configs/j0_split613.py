_base_ = './hrplus_patch_dice.py'

# ============================================================================
# J0：从 D 模型（iter_20000）续训，冻结全部参数，只解冻 student 的 decode_head。
#
# 协议的三处关键决定（均由 2026-09-17 子代理审计修正，附理由）：
#  1) 起点 = iter_20000（D 的收敛终态），不用 best_mIoU_iter_2000（实测只到 iter1999/epoch9，欠训）
#     —— 理由：J0/J1 要回答"在收敛的 D 之上微调有何增益"，起点必须与 D 基线同 horizon。
#  2) 学习率 = 固定 5e-5（poly + warmup_ratio=1.0 + min_lr=5e-5 ⇒ 实际恒定）
#     —— 禁止用原 config 的 5e-4 重启：对已收敛模型等于冷启动，会破坏已训特征。
#     注：两臂共享同一 LR，故绝对 LR 不破坏 J0 vs J1 的内部有效性。
#  3) 加载方式必须是 load_from（只载权重），**不能** resume_from
#     —— D 的 optimizer 里没有新解冻参数的 state，resume 会结构不匹配。
#     且 load_from 的 ckpt 已剔除 reference.*（见 prep_D_ckpt.py），保证 reference 恒为 B。
# ============================================================================

custom_imports = dict(imports=['hrplus_patch', 'j0j1_freeze_hook'],
                      allow_failed_imports=False)

# 起点：D 的 student 部分（reference.* 已剔除）
load_from = '/root/box1_sync/D_iter20000_student_only.pth'
resume_from = None

runner = dict(type='IterBasedRunner', max_iters=8000)
checkpoint_config = dict(by_epoch=False, interval=2000, max_keep_ckpts=4)

# 评估只在 4000 步（节省：本模型的推理要跑 tiled student + 整图 reference，很贵）
evaluation = dict(interval=4000, metric='mIoU', priority='LOW', save_best='mIoU')

optimizer = dict(type='SGD', lr=0.00005, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict(grad_clip=dict(max_norm=5.0, norm_type=2))
lr_config = dict(policy='poly', warmup='linear', warmup_iters=200,
                 warmup_ratio=1.0, power=1.0, min_lr=0.00005, by_epoch=False)

# priority 必须 >= NORMAL：OptimizerHook 是 ABOVE_NORMAL(40)，
# 若本 hook 排在它之前，首个 iter 读到的 grad 会是被清空后的 None → 断言假失败
custom_hooks = [dict(type='FreezeScopeHook', arm='J0', expect_trainable=8,
                     priority='NORMAL')]
