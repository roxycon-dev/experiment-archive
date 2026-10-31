_base_ = './hrplus_mares.py'

# 正确的实验：以 **D@2000（项目最好模型，val 41.615 / test 42.58）** 为冻结基准，
# 只训练语义条件 MA 残差头。reference 仍用 B（保持 D 原有的 0.5 概率融合不变）。
#
# 这样：EX/HE/SE 逐位保持 D 的水平（结构隔离），MA 由残差头提升。
model = dict(
    type='HRPlusMAOnBase',
    source_checkpoint='/root/autodl-tmp/hrplus/D_0913_retry/best_mIoU_iter_2000.pth',
    reference_checkpoint='weights/b_weights.pth',
    ma_hard_weight=0.0,
)
