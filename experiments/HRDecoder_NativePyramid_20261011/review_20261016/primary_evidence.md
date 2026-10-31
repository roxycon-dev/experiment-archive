# 原始论文与官方实现证据

核查原则：仅用原始研究、官方源码或框架文档解释具体运算定义。引用支持某个方法属性，不把其它数据集或类别的分数当成DDR 454/76/227协议上的预测。

|原始来源|可用于本方案的证据|不能推出的结论|
|---|---|---|
|HRNet 原论文，[Deep High-Resolution Representation Learning for Visual Recognition](https://arxiv.org/abs/1908.07919)|HRNet并行保留高至低分辨率流并反复交换信息；解释W48四尺度为何可同时含空间与上下文信息。|不能证明ImageNet预训练W48的每个通道都含病灶语义，也不说明新decoder能在40k内保持各尺度贡献。|
|HRNetV2 原论文，[High-Resolution Representations for Labeling Pixels and Regions](https://arxiv.org/abs/1904.04514)|HRNetV2把多个并行分辨率的上采样表示聚合到高分辨率表示；支持“多尺度上采样融合”是原方法特征。|不等价于v3逐尺度各做4类线性读出，也不证明720→4旁路最优。|
|HRDecoder 原论文和项目实现，[MICCAI开放论文页](https://papers.miccai.org/miccai-2024/390-Paper4020.html)、[论文PDF](https://papers.miccai.org/miccai-2024/paper/4020_paper.pdf)、[官方GitHub](https://github.com/CVIU-CSU/HRDecoder)|论文描述HRL高分辨率局部学习和HFF多尺度局部/全局融合；实现配置将 HRNetV2-W48 四尺度合成720通道，再由FCNHead解码；全图与高分辨率裁块以带权损失组合。项目保存代码的配置更精确表明 `compress=True` 下是720→64 1×1、再64→64 7×7，当前 `hr_loss_weight=.1`、`lr_loss_weight=0`。|H0候选适配器关闭`compress`，720→64的7×7 head不是同一原始结构。论文数据/阈值与本轮融合标签、处理、划分不同，不能替代本任务基线或v3成绩。|
|ICNR 原论文，[Checkerboard artifact free sub-pixel convolution](https://arxiv.org/abs/1707.02937)|ICNR通过重复子像素卷积核初始化，使初始子像素卷积对应resize-convolution式行为，论文报告初始化时没有checkerboard伪影。|只涉及特定初始化时的相位伪影；不说明零末层后D支路的梯度、40k稳定性或mIoU。v3的D权重/偏置全零会令初始D恒为零，是更强的“零初始残差”属性，不是采用ICNR或获得其所有实验结果。|
|Group Normalization 原论文，[Group Normalization](https://arxiv.org/abs/1803.08494)|GN不沿batch维估计统计量，论文在小batch检测/分割/视频情形提供了替代BN的实验依据。|GN按每个样本分组，对NCHW特征仍跨组内通道和空间聚合；小batch动机不能推出其在本任务的空间pad完全不变或能消除崩塌。|
|Layer Normalization 原论文，[Layer Normalization](https://arxiv.org/abs/1607.06450)|LN对单个样本的神经元输入求统计，不用mini-batch统计。|原论文并未证明在这里的高分辨率卷积分割、极稀疏病灶上优于GN。归一化范围由实现的`normalized_shape`决定。|
|PyTorch官方[LayerNorm文档](https://docs.pytorch.org/docs/main/generated/torch.nn.LayerNorm.html)与[GroupNorm文档](https://docs.pytorch.org/docs/stable/generated/torch.nn.GroupNorm.html)|`LayerNorm(normalized_shape)`对最后D维求统计；图像NCHW上`LayerNorm([C,H,W])`会跨通道与空间，需先转NHWC再用`LayerNorm(C)`才得到每个像素的channel-only统计；GN给每个通道可学习affine，LN(C)也有每通道γ/β，因此若两者通道数C相同，affine参数数都是2C。|文档的轴定义不保证autocast时内部FP32临时分配或总显存峰值；执行版本应明确FP32统计并实测。|
|PyTorch官方[interpolate文档](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html)与[PixelShuffle文档](https://docs.pytorch.org/docs/stable/generated/torch.nn.PixelShuffle.html)|`interpolate`定义空间重采样与`align_corners`选项；PixelShuffle把`r²`倍通道重排为空间倍率r。它们支持检查v3的shape和相位映射。|不同 resize 顺序通常不是等价操作；单tensor大小也不是实际峰值。|
|Focal Loss原论文，[Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)|以降权易分类样本聚焦稀疏难例，支持讨论稠密预测前景/背景数量不平衡动机。|本方案的top-k softplus负例、全batch分母和Dice权重不是该Focal Loss；Focal目标检测结果不验证自定义稀疏分割loss。|
|Generalized Dice原论文，[Generalised Dice overlap as a deep learning loss function for highly unbalanced segmentations](https://arxiv.org/abs/1707.03237)|提出稀少类别的不平衡分割loss重平衡动机。|v1/v3的每图正类Dice与其广义按类体积加权定义不同，不能用该论文声称自定义系数或梯度稳定。|

源码对照取自已读的本地官方派生代码而非动态运行：`autodl100/work/HRDecoder/mmseg/models/segmentors/HRDecoder.py`、`.../decode_heads/fcn_head.py`、`.../losses/binary_loss.py`、`configs/_base_/models/hrdecoder_fcn_hr18.py`和`hrdecoder_fcn_hr48.py`；另以FullCompetition H0候选`trainer/models/hrdecoder_adapter.py`比对 `compress` 参数默认值。论文属性只支持本表所述机制边界，不替本地运行身份、数据口径或当前v3实证背书。
