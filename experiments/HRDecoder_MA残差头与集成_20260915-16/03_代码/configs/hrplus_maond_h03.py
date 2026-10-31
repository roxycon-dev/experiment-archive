_base_ = './hrplus_maond.py'

# 第二阶段（R1 设计）：Dice + 低权重 MA 难负例项。
# 相对 hrplus_maond.py 的唯一变化 = ma_hard_weight 由 0 -> 0.3
model = dict(ma_hard_weight=0.3)

runner = dict(type='IterBasedRunner', max_iters=4000)
evaluation = dict(interval=500, metric='mIoU', priority='LOW', save_best='mIoU')
