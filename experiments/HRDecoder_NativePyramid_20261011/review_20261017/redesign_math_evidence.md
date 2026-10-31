# 重新设计的损失与采样证据

范围：复用 `review_20261016` 的运行/源码审计，精确复核原 HRDecoder 官方实现与当前 v3 的损失和采样数学；不运行训练、forward/backward 或新评估，不读取新像素/checkpoint。以下把源码事实、实验观察和推论分开。

## 1. 原 HRDecoder 与 v3 的目标函数

### B0 原损失（源码事实）

归档配置指定 `BinaryLoss(type='dice', smooth=1e-5, loss_weight=1)`，类别权重为空。对 batch 中每张图 i、前景类 c，源码先 sigmoid，再将空间维展平，计算

```text
D_ic = 1 - (2 Σ_hw p_ichw y_ichw + ε)
             / (Σ_hw p_ichw² + Σ_hw y_ichw² + ε),  ε=1e-5
L_B0 = (1 / (B·4)) Σ_i Σ_c D_ic
```

也就是每图/每类一项 Dice，先四类等权平均，再对 batch 取均值；它不是把所有图的像素先合并成一个 Dice。无阳性图/类仍计算平方 Dice，预测出的概率会受惩罚，不能在复刻时整类跳过。源码的 one-hot 把 0 和 255 都映射成四个前景通道全零，但没有对 255 建有效像素 mask；当前 B0 DDR 配置的旋转和 pad 标签值为 0，所以现有证据不能把“255 的处理”说成该 run 的已证失败原因。新实现若允许 255，必须显式 mask。

空阳性类的惩罚不等于有很强的背景梯度。令 `S=Σ p²` 且 `y≡0`，则 `D=1−ε/(S+ε)`，逐像素 logit 导数为 `∂D/∂z_j=2ε p_j²(1−p_j)/(S+ε)²`。当许多预测已使 `S≫ε` 或单点 `p→1` 饱和时，梯度可很小；因此“Dice仍计入 absent class”不能解释成“必然足以快速压制高置信假阳性”。相反，若该类整图全被 ignore 而没有有效像素，正确 masked 实现应跳过该图/类，不能让空掩码伪造背景监督。

整次训练不是单一“局部 crop loss”：

```text
L_B0,total = Dice(fused full-canvas logits, full labels)
           + 0.1 × Dice(random HR-crop logits, crop labels)
```

`lr_loss_weight=0` 关闭仅 LR global logits 的独立辅助损失；全画布的 global/local 融合 logits 仍以系数 1 受监督。每图另取4个随机 crop，crop边长为 1024×U[0.75,1.25] 并取整至8的倍数，位置在8像素离散网格均匀抽样。HR crop 特征来自同一次“整图先缩放到1024→W48”的特征图，再裁切/缩放；不是把2048原图 patch 重新送进 W48。推理在整图上下文 logits 与 tile logits 间用固定0.5融合。因而 B0 已实测“共享 W48、整图上下文、局部 tile 解码”，但它没有验证“W48 原尺度 patch 重编码 + 整图分支”。[官方 HRDecoder 代码](https://github.com/CVIU-CSU/HRDecoder) 与本地 [BinaryLoss](../../HRDecoder_Ready_20261002/vendor/champion_repo/mmseg/models/losses/binary_loss.py)、[DDR 2048 配置](../../HRDecoder_Ready_20261002/vendor/champion_repo/configs/lesion/hrdecoder_fcn_hr48_ddr_2048.py)、[HRDecoder 训练路径](../../HRDecoder_Ready_20261002/vendor/champion_repo/mmseg/models/segmentors/HRDecoder.py) 一致。

### v3 当前损失（运行源码事实）

每个 head（Z 主输出、S 辅助语义输出）使用相同定义。设每图/类有效阳性数为 P_ic，负像素数 N_ic，按该 head 当前 logit 取前 k_ic 个困难负例，`k_ic=min(N_ic,max(1024,3P_ic))`；batch 分母为 `P_c=Σ_i P_ic`、`K_c=Σ_i k_ic`、`A_c=# {i:P_ic>0}`、`V=Σ_i 有效像素数`。每 head 的第 c 类项是：

```text
D_ic = 1 - (2 Σ p y + ε)/(Σ p² + Σ y + ε)，仅 P_ic>0 时计入
L_c = Σ_i D_ic / max(A_c,1)
    + 0.25 Σ_positive softplus(-z) / max(P_c,1)
    + 0.25 Σ_selected-hard-negative softplus(z) / max(K_c,1)
    + 0.05 Σ_valid BCEWithLogits(z,y) / max(V,1)
L_head = (1/4) Σ_c L_c
L_total = L_Z + 0.2 L_S
```

四类权重相同；正/挖掘负项分别按 batch 内各自像素数归一；Dice 只在该类有阳性的图上算，因此 absent-class 的假阳性惩罚由困难负项/密集 BCE承担，而非 Dice。Z/S 分开挖负例，分母跨 microbatch 共享；255 在 v3 `target_masks` 中显式排除。损失是 Dice + 动态 hard-negative + 全有效像素 BCE 的混合，不是固定权重 BCE。它与 B0 的关键差异是图级 Dice改为阳性图条件 Dice，并新增困难负和 dense BCE 项；不是“B0未平衡、v3首次平衡类别”——B0 本来就对四类取宏平均。

### 固定阈值与组平衡 BCE：有限的数学结论

对只含固定样本分布的纯加权 BCE，给定自然后验 `η=P(Y=1|x)` 和正/负逐样本权重 `a,b`，点态最优概率为

```text
q* = aη / (aη + b(1−η));   q*>0.5  ⇔  η > b/(a+b).
```

所以“逐像素权重相等” `a=b` 对应自然后验阈值0.5。若损失是正负组均值等权 `0.5 E[ℓ+|Y=1]+0.5 E[ℓ−|Y=0]`，换回自然像素分布后逐像素系数分别与 `0.5/π`、`0.5/(1−π)` 成正比（π 是正例率），固定 q=0.5 对应自然后验阈值 η>π。这解释了组均值可能改变分数校准，不能把“组贡献各0.25”误读为自然分布上的 `a=b`。

此闭式不适用于当前 v3 总损失：挖掘负分布随模型 logit 变化且依赖输入，`K_c/P_c` 随 batch 和类别变化；同时存在非 pointwise 的 Dice、dense BCE、Z/S 共享参数与全局裁剪。因此该数学只提示严格固定 `sigmoid(z)>0.5` 评估可能受训练目标校准影响；当前没有因果证据说它导致低召回/低 IoU，也不能从 loss 系数推出该 run 的有效阈值。

## 2. 采样期望与可以保证的范围

令预先固定的有限 tile 集合为 T，定义 tile 损失 `L_t(θ)`。均匀抽一个 tile 的损失，其期望恰为

```text
E_{t~Unif(T)}[L_t(θ)] = (1/|T|) Σ_{t∈T} L_t(θ).
```

若希望用训练集阳性偏置的 proposal `q(t)=α p(t)+(1−α)r(t)` 采样、但优化目标仍是均匀/指定分布 `p(t)` 下的**这个 tile 损失期望**，只要每个目标 tile 都有 `q(t)>0`，可用未自归一化权重 `p(t)/q(t)`：`E_q[(p/q)L_t]=E_p[L_t]`。应保留准确抽样概率；对单个 batch 把权重再除以其随机和，有限样本下变成自归一化估计，不再严格无偏。阳性 proposal 只能使用训练标注，且会有权重方差/有效样本量代价。

保证仅针对所定义的 `L_t`。若 `L_t` 是 crop Dice，期望就是“均匀 crop-Dice 目标”，不等于整图 Dice：ratio-of-sums 与 sum/expectation-of-ratios 不交换；同理，不能据此声称全图 IoU 无偏或保证全图指标提高。对 B0 而言，随机 HR crop 项是辅助目标，另有 fused full-canvas Dice 主目标。

当前 v3 每次均匀抽训练图、随机缩放裁 2048 画布；只尝试至多10次让任一有效标签类别不超过75%，没有“每个 batch 必有 MA/病灶阳性”约束。即使 crop 有多个类别也可能没有 MA。B0 的 RandomCrop `cat_max_ratio=.75` 与4个均匀 HR crop 同样不保证每类阳性。MA 训练信号出现频率需要用逐 update 正像素/阳性图日志测量，不能从宏平均 loss 名称推定。

## 3. 已有正负实验能支持什么

- **B0 是最强的单 W48、Dice-only 正参考。** ImageNet W48、原 HRDecoder FCN、40k 上限、global batch2、SGD .005；报告 val best 39.390%@32k、test42.20%。它证明原配方能训练出可用基线，不证明必超融合目标。B0 full checkpoint/精确源码运行 SHA 和逐类计数不全。当前 v3 的 val Z@40k 39.4256%，比 B0 best 仅 +0.0356pp，且比锁定融合目标45.3218%低5.8963pp；两者差异同时涉及2048整图直接进W48 vs B0 backbone 输入1024、解码器、BN/LN、损失与S/Z双监督，不能归因单因子。40k“同预训练”既不锁定算力，也不锁定函数或优化轨迹。
- **局部细节和上下文都可能有用，力度失衡会伤另一类。** 原论文描述的 HRDecoder 正是共享低分辨率整图 W48 特征+局部 crop decoder+融合，并报告 crop 数过多偏小病灶、会忽略较大 SE；HR loss 权重过大会偏局部、0.1 较合适。官方论文在 IDRiD 上报告简单0.5加权融合优于其两种 scale-attention 对照。这是“局部与上下文要一起衡量”的主来源证据，不是本地 DDR checkpoint 的因果归因。[MICCAI 2024 原论文](https://papers.miccai.org/miccai-2024/390-Paper4020.html)
- **MA 的背景/假阳性抑制可提升 IoU，却可能牺牲召回。** H4 冻结 D+reference 底座，仅训约2k的 MA 校正：测试 MA TP/FP/FN 69,270/90,073/120,773→80,641/96,545/109,402，IoU 24.729→28.138%；其它类计数不变。它支持窄输出校正可同时加 TP、但 FP 也上升；不证明 W48 在线训练效果，且测试集已复用。Detail 同 checkpoint 的 private/base 4k 头对照中，MA FP 1,326,940→63,322、IoU 1.81→16.45%，但 TP 25,245→21,810、FN相应增加；Detail 后续16k整体退化且 SE 零检出。它支持一次强假阳性抑制的 precision/recall trade-off，不是独立训练的因果消融。
- **BCE/采样调整没有可直接移植的胜利配方。** 12个 train-only batch 的旧梯度探针观察到高置信正例 Dice logit 梯度约1e−13–1e−11、平衡 BCE 约1e−6–1e−5；无 SGD 更新，不能说明共享 W48 参数方向或最终指标。500-update stratified teacher-KL normalization arm 的 mIoU 比 control 高1.632pp，但 MA precision 约−3.13pp、recall约+11.26pp，没过预定门槛。IoU/Lovasz+BCE 组合 screen 的 FP下降伴随多类 TP下降，且同时换了损失，不能断言任何单一分量有益/有害。详见 [历史运行审计](../review_20261016/runs_findings.md)。

当前融合 val 参考逐类 IoU EX/HE/SE/MA=50.37/45.04/59.84/26.03%，而 v3 Z@40k 为46.62/40.94/49.64/20.50%；最大差额是 SE（10.20pp），不是 MA（5.54pp）。这不证明每类都必须达到融合系统的类别 IoU 才能满足45.32宏平均目标；它只反对把损失改造单押在 MA 上。监督需同时观察类级 TP、FP、FN，不能仅以正类召回或背景精度判优。

## 4. 双分辨率双路径提案的数学边界

对候选固定 logit 平均 `z=(z_G+z_N)/2`，融合处对每条路的局部导数恰为1/2。主损失梯度随后通过两次不同输入视图反传到共享参数，两个路径梯度按链式法则累加；这既不是全局梯度简单减半，也不是把旧学习率折半。平均的是 logit，所以 `sigmoid(z)` 的赔率是两支赔率的几何平均 `odds(z)=√(odds(z_G)·odds(z_N))`，不是两支概率算术平均；一支输出很低会拉低融合结果。固定系数避免学习一个自由分支门，但不保证分支置信度校准相同。

若保留只施加到 G 的 `0.1 L_crop(G)`，令 `g=∂L_full/∂z`、`h=∂L_crop/∂z_crop`，`J_G/J_N/J_crop` 分别表示各输出对共享 W48 参数 θ 的 Jacobian，则该主干梯度为

```text
∇θ L = 0.5 J_Gᵀ g + 0.5 J_Nᵀ g + 0.1 J_cropᵀ h。
```

因此两次全图分辨率监督共享权重，但 G 另有 crop 辅助信号；SGD 梯度范数可以显示两个视图都通到参数，却不能单靠范数判断两路梯度方向抵消或哪个分支改善验证。

主损失应在已融合、完整标签网格上只计算一次 B0 同定义的 image/class Dice；因 Dice 是非线性比值，`D((z_G+z_N)/2)` 不等于 `.5D(z_G)+.5D(z_N)`。如有255 ignore，公式中的交叠和两个分母都先仅在有效像素上求和。若当前图的某类没有正像素但仍有有效像素，照常计算平方 Dice 以保留背景 FP 惩罚；仅该图完全没有有效像素时才排除。用整图 Dice+0.1个 M=4 随机低分特征 crop Dice 是对原 HRDecoder 目标的数值/归约复用，但编码器视图数、归一化状态与监督路径已改变，因此不等于重现了 B0 训练。

独立 global/native BN running-state bank 只能避免同一 buffer被不同分辨率前向轮流覆盖；训练模式中 BatchNorm 输出仍使用本次 microbatch 的统计量，buffer 隔离不使微批1等价于 global batch2，也不确保验证时分支统计校准。相同W48参数在两种分辨率共享，不代表各分支特征/logit已对齐。按输入像素面积粗估，B0单1024编码为1份、1024 global+2048 native为5份；相对单2048编码约1.25份。此比例只是输入面积量级，不是实测 FLOPs、显存或 wall time。

## 5. 可保留基线函数的约束与审稿问题

数学上可以定义 `Z_θ(x)=Z_B0(x)+R_θ(x)` 并令 `R_θ` 初始化严格为零（包括其最终输出层），在相同输入、预处理、归一化状态、插值、权重和阈值下，初始化输出与 B0 函数一致。冻结 B0 路径也可在任何时刻保留一个显式基准输出。两者只保证函数起点/基准可用，不保证更新后仍更好；训练可把 logits 推离该函数，val 与 train 分布有限，单点初始化也不建立泛化界。若新模型换了 1024/2048 backbone 视野、标签掩码或损失，这个等式也不成立。

设计/审稿时应能回答：

1. 正类 BCE、困难负 BCE、dense BCE 各自如何归一？阳性图缺失时 Dice 是否仍处罚背景假阳性？255 是否从每个分子和分母都排除？微批是否复用同一整 batch 分母？
2. 固定 `>0.5` 下，损失是否保留概率校准；正/负组权重指自然像素权重还是条件组均值？hard-negative 采样概率是否随 logit 变、是否存在所有目标 tile 的支持？
3. 训练采样是否记录每类正像素、正图、零阳性更新比例？MA 单靠一个稀疏 batch 的梯度范数不能证明欠拟合；要连同 train/val 曲线、FP/FN、正常 SGD 梯度一起判断。
4. “局部tile”是重新编码原始2048图像 crop，还是裁切全图1024 backbone feature？全图上下文分支及其融合权重是否与局部分支同时受 full-image 监督？
5. 性能门槛比较的是单 W48 raw@40k，还是多模型融合？是否使用相同图像/标签字节、阈值、metric aggregation 和选点？目标系统的45.3218%不能转写成 W48 单模型的基准保证。

## 6. 全量 SGD 日志：能判定与不能判定的优化风险

现在已取得原始日志的只读全量投影。远端原文件为 `/root/autodl-tmp/nativepyramid_sp_v3_20261016/run_seed42_launch02/train.jsonl`，3,473,259,370 bytes，SHA-256 `2b286aee99ed75d894c8d60a239c92018fed158bc98d0def7eed9e61fcf3a63f`；投影覆盖 40,000 条连续唯一 update，无缺行/异常，projected JSONL SHA-256 `a72d77846ace86e429fd4120689ed814aa67668e6dd627c9274216835a5ac3a9`。只省略了 `gradients_before_clip/after_clip.parameter_modules` 两个巨型映射，尺度×类别梯度、模块范数、loss、lr、clip、训练阳性计数都保留；完整回收清单和校验在 [投影manifest](recovered_logs/train_metrics_all40000.manifest.json)。运行正常完成 40,000 update，checkpoint 持久化到40k；没有 pause 或 in-flight update。代码每次成功更新记录完整的主 head四项 loss、总/主/辅 loss、裁剪前全局范数、clip系数、裁前后模块/直读分尺度×类别范数与阳性监督量。

训练目标持续下降：2k桶总 loss 从首桶 1.1472 依次降至末桶 0.4503，主 loss 从 0.9559 降至 0.3750；与此同时验证 Z 在 36k 达最佳40.2770%，至40k回落到39.4256%。这是“训练目标持续改善但验证未同步”的证据，兼容过拟合、阈值/校准变化或训练目标与验证mIoU不匹配；日志本身不能区分它们，不能仅凭末段val下降断言过拟合。逐项 loss 只记录 Z 主 head；S 的 `aux_loss` 有标量，但没有 S 的四项分解或 Z/S 梯度拆分。

裁剪阈值为全局梯度范数5。全40k梯度范数中位数3.624、p90 7.543、最大113.274；clip在26.77%的更新触发，系数中位数1、均值0.9225。首2k触发率67.3%、系数均值0.772；末2k触发率19.8%、系数均值0.939。系数<0.5占全程4.545%；极端点包括update34052范数113.27/系数.044、33143范数83.11/.060、21271范数63.92/.078。clip使用一个公共标量，触发时不会选择性压某个类别或模块；它确实缩小了部分更新的反传梯度，并改变其相对于动量/权重衰减的影响。`lr×clip`只是无动量、无衰减时的梯度步长代理，不是实际参数位移。因B0的 `optimizer_config=dict()` 未设置grad_clip，而v3锁了max_norm=5，禁用clip是恢复参考配方的决定；这些日志还不足以把clip判为失败原因。

MA正像素覆盖不支持“多数更新没有MA目标信号”：平均1,902像素/update，中位976，7.13%更新无MA阳性；20个2k桶均值为1,853–1,967，零阳性率6.4–8.2%。SE每update均值约4,800–5,300像素，但约48–50%更新没有SE阳性，这主要反映一个batch两张图只有部分含SE，不等于训练集不覆盖SE。direct classifier权重梯度的全程中位数（F1–F4）MA=.133/.117/.118/.115，均高于SE=.038/.061/.058/.061，因此MA不是测得最弱的直接分类器类。该范数混合了Z与0.2S的贡献且不是logit梯度，也不表示相同更新方向或验证收益。

各2k桶的梯度探针中，C endpoint(auxiliary)与D endpoint(readout)、D stem/detail、shared semantic trunk的中位范数均持续非零；末2k分别为.081、.037、.115、.826。它们是组合loss的参数梯度范数，不是各支路对共享主干的独立因果贡献。主head的加权scalar loss份额从首桶约正BCE30.5%、困难负BCE12.5%、Dice56.8%、dense BCE0.20%，到末桶约29.7%、15.5%、54.7%、0.05%；scalar loss份额不等于梯度份额，不能据此说dense BCE梯度可以忽略或Dice支配更新。

因此，完整日志削弱了“训练完全崩塌”“MA正监督长期缺失”“全程被clip强压”“C/D末端断梯度”这些解释，但未给出优化/泛化根因。当前设计恢复 B0 的平方 Dice、0.1 crop 系数和不裁剪SGD，是有历史参考的受控配方选择，不是由这次v3日志证明其可达到目标；空阳性平方Dice的背景导数可能很小这一数学风险仍然存在，尤其高置信假阳性饱和时。后续如判断某个具体优化因子，应先与同口径B0日志/配置逐项比较；不要从 loss 数值或范数相关性推出新 BCE 系数或单一根因。
