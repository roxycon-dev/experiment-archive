_base_ = './j0_split613.py'

# ============================================================================
# J1 = J0 的全部设置，**唯一差异**：额外解冻 HRNet 的最后一个 stage（本仓库 = stage4）。
#   J0 trainable 张量 = 8（全在 decode_head）
#   J1 trainable 张量 = 440（8 + 432，全部在 backbone.stage4）
# 其余（起点 iter_20000 / 固定 lr 5e-5 / 8000 步 / seed 42 / batch 2 / 1024 crop /
# BN 恒 eval / reference 冻 B / load_from 而非 resume_from）完全继承，保证单变量。
#
# 说明：resume_from 与 load_from 均继承 J0（load_from 指向同一个已剔除 reference.* 的 D）。
# ============================================================================

custom_hooks = [dict(type='FreezeScopeHook', arm='J1', expect_trainable=440,
                     priority='NORMAL')]
