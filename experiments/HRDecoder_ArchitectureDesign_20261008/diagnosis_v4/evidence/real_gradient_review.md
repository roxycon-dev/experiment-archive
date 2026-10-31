# 20k 真实输出梯度与 GT loss 修订审查

## 判定

现有证据足以把“在 20k 冻结输出上，EX/SE 的 GT 阳性像素没有得到有效的 Dice 正向压力”定位为一个**已测到的近端失效机制**；它不能唯一定位训练曲线退化的因果来源。真实样本上，EX/SE GT 区域 student 概率约 `1e-8–1e-7`，旧 Dice 对该区域的梯度和仅约 `5e-8–2.8e-7`；换成阳性像素均值 softplus 后，同一输出张量的直接梯度和约 `−0.25`。老师在同一实际增强几何下对这些 GT 区域有高概率，且对 4 张所选训练/验证图切换到 BN batch stats 仍未出现 EX/SE 阈值预测。三类证据共同排除了“只是 teacher 未对齐”或“只因 BN running stats”这类简单解释。

这仍是输出张量级反事实，不是完整优化轨迹。`real_objectives.json` 明确记录 6 张刻意筛选的训练图、固定 augmentation seed、formal updates=0，且只求输出梯度，没有完整参数梯度回放；其 `old_main_without_hr` 省略 HR loss，也未纳入 signed auxiliary 的实际 private 路由。实际模型参数 Jacobian、优化器动量/学习率、其他病例分布和 20k 后更新均未由该 JSON 证明。因此可说 GT-Dice 正梯度不足在所选冻结输出上已被直接证实；不能说这是 20k EX/SE 全部失效的唯一原因。

## 真实梯度证据

证据来自 `diagnosis_v4/evidence/real_objectives.json`，元数据为 checkpoint 20000、CUDA FP32、TF32 关闭、formal updates=0、实际 `TrainCacheDataset` 增强；限制字段说明是 6 张刻意选择的训练图和输出梯度，而非全参数因果回放。实现见 `diagnosis_v4/probe_real_objectives.py:50-74`。

三张选中的 both-positive 训练图上，GT 有效面积为 1,157,156 像素。EX GT 阳性像素数为 367、3,573、2,212；SE 为 777、355、3,126。旧 Dice 的 GT 区域平均概率：EX 为 `1.13e-7–1.39e-7`，SE 为 `2.54e-8–1.11e-7`。旧 Dice 在 GT 阳性像素上的梯度和分别只有 EX `−2.26e-7…−2.78e-7`、SE `−5.07e-8…−2.23e-7`，loss 接近 1。`repaired_GT` 的梯度和在每个样本/类别都是约 `−0.25`，正好体现阳性像素均值 `softplus(-z)` 在低概率处不饱和；这是约百万倍的输出梯度差异，不是参数更新量差异。

同一三张图上，混合 teacher `T` 的 GT 阳性均值为 EX `0.636–0.761`、SE `0.772–0.872`；阈值 0.5 的 teacher GT recall 分别为 EX `0.654–0.790`、SE `0.811–0.913`。这证明这几张图上的蒸馏 teacher 并未在 GT 区域整体缺信号。实际 teacher 几何/缓存的全数据正确性仍超出六张抽样能覆盖的范围。

不能把“旧 Dice 正向信号几乎为零”误读为“旧主目标完全没有上抬信号”：JSON 的 `weighted_old_KL` 已包含 0.5 权重。在这六个 EX/SE 阳性样本上，其公共 logit shift 梯度约为 `−3.0e-4…−1.76e-3`，方向是上抬，明显强于旧 Dice，但仍比 `repaired_GT` 的 `−0.25` 小约 140–830 倍。由于该数字只对输出 logits 求导，不能用此比例推定 backbone/head 参数更新的比例。

三张 both-absent 训练图上，EX/SE 的旧 Dice 空类公共 shift 梯度是向下的：EX `+1.83e-4…+2.34e-4`，SE `+4.41e-5…+4.98e-5`。候选 `repaired_GT` 在这些空类上的梯度仅为 EX `+4.46e-9…+5.06e-9`、SE `+2.05e-9…+2.13e-9`。因此它明显削弱了空类输出压力。需要留意，所读到的六条 EX/SE 空类记录并没有出现 `0.27` 量级的公共 shift 梯度；若 `1e-4–0.27` 指其他类别、逐像素范数或其他统计定义，应在引用时注明统计口径，不能将上限归到本 JSON 的 EX/SE `common_shift_gradient`。

## BN 反事实的证据边界

`diagnosis_v4/evidence/normalization_probe_cuda.json` 状态为 COMPLETE，含 20k checkpoint 的 2 张 train + 2 张 val 有阳性图，分别用 running stats 和仅切换 BatchNorm 为 batch stats 的单图推理。4 张图的 EX/SE 在两种模式下阈值 0.5 的 TP 都为 0；最接近的一张 val 图 batch-stats 下 EX 最大概率也只有约 `9.68e-4`。这使“20k 选中图的漏检仅由 running BN 统计导致”很不可信。该探针是每次一个样本的推理反事实、关闭 dropout，不是训练时多批次 BN 轨迹，更不是全验证集统计，故不能完全排除 BN 与其他数据/尺度条件交互。

## 对建议的 GT loss 逐项核验

建议按每个样本、每类定义 `m=valid`、`y=该类 GT`、`pos=m*y`、`neg=m*(1-y)`，先算每个图像/类别标量，再对固定的 active-valid 图像/类别对做宏平均：

\[
L_{i,c}=D_{i,c}+0.25(P_{i,c}+N_{i,c}),\quad
P=\operatorname{mean}_{pos}\operatorname{softplus}(-z),\quad
N=\operatorname{mean}_{neg}\operatorname{softplus}(z).
\]

其中 `D=0` 若该图该类没有 GT 阳性；否则沿用 squared Dice，分子和分母都只对 `m=1` 的像素求和。若 `pos` 或 `neg` 组为空，只把对应均值项设成图连接的零，不把另一个组重新乘 2。macro 分母固定使用 active-valid 图像/类别对数；全 batch 没有 active-valid 对时返回 `z.sum()*0`，避免空均值产生 NaN。无效像素不能在构造 target 时重新标成有效背景，否则它们会错误进入 Dice 分母或 negative BCE。

在单个有效图像/类别对上，阳性 BCE 对公共 logit shift 的梯度和是 `−0.25*mean_pos(sigmoid(−z))`，低概率时约 `−0.25`；阴性 BCE 是 `+0.25*mean_neg(sigmoid(z))`。再做固定 active-valid macro 时，两项都除以同一个有效图像/类别对数，不按组有无重新缩放。若一个 batch 有阳性组的对占 active-valid 对比例 `r`，阳性 BCE 的 batch 梯度贡献自然约为 `−0.25r`（再乘 `mean_pos(sigmoid(−z))`）；若该 batch 没有此类阳性，当前 batch 就没有该阳性项。

`probe_real_objectives.py:48,68-74` 的逐样本 `repaired_GT` 与上述公式一致：有效 mask 同时用于 Dice 分母和正负 BCE 组，缺失组由 `clamp_min(1)` 得零，无正类时 Dice 使用 `z.sum()*0`。但这个探针没有批量 macro；要实现训练版，应在每个样本/类别先算 `L_i,c`，再按同一 active-valid 对集合求平均，不能按 positive-present 组另行归一化。

该式合理地修复了已测到的阳性梯度饥饿，但有明确代价：

- **空类 FP 抑制会弱很多。** 在选中 EX/SE 空图上，新 loss 的 negative-mean BCE 输出梯度只有约 `2e-9–5e-9`，旧 Dice 是 `4e-5–2.3e-4`。负组均值会把少量高概率假阳性的惩罚按百万级背景像素稀释；新目标有提高稀疏 FP 的风险。正类 mean 归一化也让只有少数标注阳性的图与大片阳性的图获得相近的总阳性项，可能放大极小/含噪标注的影响。
- **概率不再按真实像素患病率校准。** 当一个图像/类别同时有正负组时，两个条件均值给正、负组固定系数；它不再是逐像素 prevalence-weighted Bernoulli NLL。常数 logit 的 BCE 部分在正负组都存在时趋向 `z=0`，不会按实际阳性面积给出校准概率；Dice 又会改变该折中。不能默认沿用概率阈值 0.5，应在独立验证集做阈值/校准检查，并关注 precision-recall 而非只看 recall。
- **KD 会与 GT 负组明显冲突。** 在三张 both-absent 图上，现有 `weighted_old_KL`（已经乘 0.5）对 EX 的 shift 梯度是 `−9.16e-5…−1.41e-4`，对 SE 是 `−1.81e-4…−1.88e-4`，都把输出往上推；候选 GT negative BCE 的梯度只有 `+2e-9…+5e-9`，往下压。这里 teacher 的空 GT 概率质量高于学生的极低概率，KL 梯度会压过新 GT 背景校准。阳性病例上 KD 通常与 GT 上抬方向相同，但它在 GT-negative 像素上也可能推高概率。先验证 GT-only 的新 loss；再逐类记录 `GT-positive`、`GT-negative`、KD 的输出梯度范数/夹角，按实际冲突幅度设蒸馏权重或门控，不能直接把旧 `lambda_k` 原样带入。

## 机制已定位到哪一步

可以成立的结论是：冻结 20k 模型在所选真实增强图上对 EX/SE 阳性像素给出的概率极低；旧 Dice 在这些像素上的直接梯度已接近零；存在 GT 的阳性归一化 logit loss 会把直接输出梯度提升到明显非饱和量级；仅切换 BN 统计未恢复这些图的 EX/SE 检出。这已把正样本梯度饥饿从合成推测提升为真实输出上的可复现近端机制。

尚不能声称它是唯一训练根因。样本是为同时含/不含 EX、SE 而选的 6 张训练图，固定一个 augmentation seed；验证集只在 BN 反事实中抽了 2 张。探针未回放完整参数梯度、HR loss、signed auxiliary private 路由或 SGD 状态；`repaired_GT` 也只是同一输出上的局部梯度，不证明采用后一定恢复验证召回。另一个仍可影响结果的明确信号是 teacher KL 在空 GT 上与新负组 BCE 对抗。下一次最有判别力的实现验证应是冻结 checkpoint 上比较 GT-only 新 loss 的输出梯度，再做受控训练/验证对照；该审查本身没有改训练代码或启动训练。
