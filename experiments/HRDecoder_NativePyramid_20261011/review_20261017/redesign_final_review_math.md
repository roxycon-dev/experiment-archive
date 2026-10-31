# v3 损失与优化证据：反方终审

范围：复核原 HRDecoder Dice、v3 正常训练日志与新设计中的损失、clip、BN 数学边界。未运行模型或训练，未读取 val/test 像素或 checkpoint。完整日志投影身份与覆盖见 [manifest](recovered_logs/train_metrics_all40000.manifest.json)，逐 2k 汇总见 [CSV](recovered_logs/full_train_metrics_2k_summary.csv) 和 [JSON](recovered_logs/full_train_metrics_2k_summary.json)。

## 结论

v3 没有表现为训练崩塌：40k 更新完整，train 总 loss 按 2k 桶从1.147降至0.450；主干与头部模块范数持续非零。验证 Z 在36k达到40.277%，40k降至39.426%。日志支持“训练目标继续下降而 val 没有跟上”，但无法在过拟合、阈值校准、目标错配或架构视野变化中单独定因。

也没有证据支持“MA长期没有监督”或“MA直接分类器梯度最弱”：MA平均1,902正像素/update，仅7.13%更新为零阳性；F1–F4直接分类器参数梯度中位数，MA=.133/.117/.118/.115，高于SE=.038/.061/.058/.061。相反，SE约48–50%的更新没有正像素。上述权重梯度合并了 Z 和0.2S的梯度，不是 logit 梯度，也不能代表验证收益。

全程clip触发26.77%更新，系数中位数1、均值.9225；首2k触发67.3%、系数均值.772，末2k触发19.8%、均值.939。它确实缩小部分更新，但全局范数裁剪用同一标量，不选择性压某一类别。B0参考配置没有clip，v3使用max_norm=5；新设计取消clip是恢复历史优化器的选择，不是此run已证明的修复根因。

## 损失与优化的精确边界

原 B0 对每张图、每个前景类使用 sigmoid 平方 Dice，`ε=1e−5`：

```text
D_ic = 1 − (2 Σ(p·y) + ε) / (Σp² + Σy² + ε)
L_B0 = mean_{image,class}(D_full_fused)
     + 0.1 × mean_{image,4 crops,class}(D_crop)
```

同一 W48 先编码缩至1024的整图；M=4 crop 来自低分特征图裁切/缩放，不是四张原分辨率 RGB crop 重编码。`lr_loss_weight=0`只关闭 LR 单独辅助损失，full-canvas 融合输出仍以系数1监督。原 BinaryLoss 把0和255都映射成全零前景标签，没有显式 ignore mask；当前 B0 DDR 配置的旋转/pad标签值为0，所以不能把255处理定为旧 run 的已证故障。新方案显式屏蔽255时，所有交叠和分母必须只在有效像素求和；有效像素存在但该图该类无阳性时仍保留 Dice 背景FP惩罚。

空阳性类的平方 Dice 不是强背景损失。令 `S=Σp²,y≡0`，则 `D=1−ε/(S+ε)`，其 logit 导数为 `2εp²(1−p)/(S+ε)²`。当 `S≫ε` 或高置信 `p→1` 时梯度可小；这是理论风险，不证明 v3 的 FP 或新方案的结果。

v3 每个 head 的类宏平均为：阳性 BCE 按全batch该类阳性数归一、权重.25；按当前 logits 选 `k=min(N,max(1024,3P))` 困难负例并按所选数归一、权重.25；只在阳性图计算平方 Dice；valid dense BCE 按全batch有效像素归一、权重.05。四类再平均；`L=L_Z+0.2L_S`。loss日志的主head加权标量份额从首2k约正项30.5%、难负12.5%、Dice56.8%、dense BCE0.20%，到末2k约29.7%、15.5%、54.7%、0.05%。这是标量损失拆分，不是梯度份额。日志没有 S 的四项拆分，也没有 Z/S 单独梯度，因此无法确定动态 OHEM、Dice、BCE 哪项支配共享参数更新；也不能把 v3 混合损失代入纯加权 BCE 的闭式后验/阈值推导。

在双视图 `Z=(G+N)/2` 下，融合 logit 的局部梯度分别为 `∂L/∂G=0.5∂L/∂Z` 和 `∂L/∂N=0.5∂L/∂Z`；共享参数梯度则为 `.5 J_Gᵀg + .5 J_Nᵀg`，若另有 G 的0.1 crop Dice，再加 `.1 J_cropᵀh`。因此不能说共享主干梯度整体减半。主损失是 `Dice((G+N)/2)`，不是两路 Dice 均值。

BatchNorm训练态按当前前向 batch 统计归一化：`x̂=(x−μ_B)/sqrt(σ²_B+ε)`；PyTorch默认`eps=1e−5,momentum=0.1`，running均值按`r_new=0.9r_old+0.1μ_B`更新（方差也更新相应batch统计）。若输入、物理batch、训练状态与前向顺序完全相同，仅把 running buffers 分成 G/N 银行不会改变该次训练输出或任务梯度；它改变的是各域的 eval-time EMA 统计。由于 G/N 输入视图不同，独立银行可分别跟踪其 eval 分布。真实物理batch=2统计不同于两个非同步 microbatch=1分别归一化后累积，即使共享完整batch loss分母，两者的 BN 输出和反传也不同。另，v3 run 将 backbone BN 保持 eval、affine仍训练，decoder 使用 LayerNorm；B0参考为训练态 BN、batch2。故跨run结果还混有归一化策略差异。新双视图、独立buffer及额外计算面积均无本run性能因果证据。

v3 全局梯度范数中位数3.624、p90 7.543、最大113.274；clip系数<.5占4.545%。`lr×clip`只近似裸梯度项幅度，不能代替含动量与 weight decay 的实际参数位移。C(aux)端点、D(readout)端点、D stem/detail、共享语义干的裁前范数在每个2k桶都有非零中位数；末桶为.081/.037/.115/.826。这只排除了探针显示的“端点整体零梯度”，不说明 C/D 的贡献方向、与另一分支的冲突或各损失项的因果作用。

## 对当前定稿的审稿约束

恢复 B0 `Dice_full + 0.1 Dice_M4` 和无 clip SGD，最大优点是减少多个未经验证的目标改动，并有可追溯参考配置。它不证明 1024+2048双视图、logit固定平均、独立 BN 状态或显式255 mask会超过45.32%融合目标；原 B0函数也不是新模型的初始化恒等函数。不要从 v3 scalar loss、gradient norm 或 clip频率追加新的 BCE系数，也不要把clip定为已知根因。后续若要判断某个优化因素，先核对并比较同口径 B0 与 v3 的日志/数据/评估协议；本报告不指定新训练或加模块。
