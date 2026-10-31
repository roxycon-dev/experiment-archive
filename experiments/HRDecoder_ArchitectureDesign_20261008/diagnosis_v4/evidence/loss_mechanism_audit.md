# 20k EX/SE 漏检：loss 机制审计

审计范围是 `training_execution/objectives.py` 的 Dice、KL、signed auxiliary，以及历史 HRDecoder 的 BinaryLoss/Dice 实现。独立探针只在 CPU 上对 8×8 合成张量运行 autograd，并对 2048² 使用闭式公式；没有载入模型、真实图像、缓存或 checkpoint，也没有优化器步骤。脚本与完整数值在本目录的 `probe_loss_mechanism.py` 和 `evidence/loss_mechanism_probe.json`。

## 代码事实

- 当前 `masked_dice` 先对 logits 做 sigmoid，构造四类 foreground one-hot，然后对每个样本、每类都计算
  \[
  L=1-\frac{2\sum mpy+\epsilon}{\sum m(p^2+y^2)+\epsilon},\qquad \epsilon=10^{-5}.
  \]
  空间尺寸不同时会先双线性 resize。`macro` 的 `active` 只看该样本是否有 valid 像素；它**不看该类是否有正像素**。因此一个有有效像素但该类全背景的样本仍进入四类宏平均。对应 `objectives.py:5-22`。
- 当前 `masked_kl` 是逐有效像素 Bernoulli KL，先减 teacher entropy，再按每个样本/类别的有效像素数做均值，最后宏平均；logit 梯度逐像素为 `p-q`。这是全像素平均，稀有 teacher-positive 区域的总信号随其像素占比缩小。对应 `objectives.py:24-29`。
- `signed_aux` 对有效像素使用 `1+4|d|` 权重，再除以这些权重之和；损失是 beta=0.1 的 SmoothL1，监督目标是 teacher 概率差，不是 GT。零差目标和零预测在该像素梯度为零。对应 `objectives.py:31-45`。
- update 20k 时 `lambda_k=0.5`、`lambda_d=0.1`；主 task 是 `Dice + 0.1*hr_loss + lambda_k*KL`，signed auxiliary 独立作为第二 loss。private 路由只把 auxiliary 梯度加到指定 private 参数，task 梯度才正常回传。对应 `objectives.py:47-52,62-92`。
- 当前训练画布为 2048²，增强后 `valid` 排除了标签 255 和几何填充；因此每样本实际 Dice 有效像素数 `V` 可能小于 2048²。对应 `data.py:84-103`。下面的整画布数值取 `V=4,194,304`，不是对真实增强样本有效面积的断言。
- 原 HRDecoder `binary_dice_loss` 也使用 `1-(2Σpy+smooth)/(Σ(p²+y²)+smooth)`，按样本展平；`binary_loss` 对四个 foreground 类求和再除以类别数。2048 配置明确设 `loss_type='dice'`、`smooth=1e-5`。对应 `binary_loss.py:46-62,72-118` 与 `efficient-hrdecoder_fcn_hr48_ddr_2048.py:33-34`。当前 masked Dice 的空类机制因此不是新引入的公式。
- 原 one-hot 转换把 label 0 和 255 都映射到被丢弃的最后一通道，所以这些像素的 target 在四个 foreground 通道全为零；但预测 `p²` 仍在 Dice 分母里。DecodeHead 会把 `ignore_index` 传给 BinaryLoss，但 BinaryLoss 的 Dice 路径没有将它转成 prediction mask。若原训练标签中确有 255，这些位置并未被真正忽略；目前是否存在这类像素需要数据证据。对应 `binary_loss.py:11-37,46-62,72-118`、`decode_head.py:226-245`。

## Dice 的空类偏置和阳性救援梯度

对一个有效像素数为 `V`、所有 logit 相同为 `z` 的空类别，令 `p=sigmoid(z)`、`S=Vp²`，则
\[
L_{empty}=S/(S+\epsilon),\qquad \partial L/\partial z=2\epsilon S(1-p)/(S+\epsilon)^2>0.
\]
梯度下降会降低 logit；当 `S≪epsilon` 时，空类损失和梯度一起趋近 0。对有 `K>0` 个阳性、其余背景的同一均匀 logit，当 `p` 很小时，损失趋近 1，整体 logit-shift 救援梯度约为 `-2p`，随 logit 下降呈指数衰减。这是“空类可形成低损失区域、漏检阳性救援变弱”的直接公式证据。

CPU autograd 复现的 8×8 当前 Dice 和历史 Dice 完全一致；loss 与闭式值在 FP32 舍入误差内相同。全画布闭式探针显示：

| 均匀 logit | p | 空类 Dice loss | 空类 shift 梯度 | 阳性占 0.1% 时 Dice loss | 阳性 shift 梯度 | 空类与阳性样本混合时，阳性样本比例需超过 |
|---:|---:|---:|---:|---:|---:|---:|
| -10.55 | 2.62e-5 | 0.99654 | +6.90e-3 | 0.999948 | -5.24e-5 | 99.25% |
| -12.34 | 4.37e-6 | 0.88916 | +1.97e-1 | 0.999991 | -8.75e-6 | 99.9956% |
| -15 | 3.06e-7 | 0.03777 | +7.27e-2 | 0.9999994 | -6.12e-7 | 99.9992% |
| -20 | 2.06e-9 | 1.78e-6 | +3.56e-6 | 0.99999999 | -4.12e-9 | 99.8845% |

“阳性比例阈值”是把同一类别的病例分成空类和阳性两种、两者 logits 与特征梯度都假设相同后，令宏平均公共 logit 梯度为零解出的比例；不是数据集统计，也不是网络参数梯度的精确阈值。在 `z=-10.55`、阳性面积 0.1% 的例子里，若阳性病例占 10%，空类与阳性样本混合 shift 梯度仍是 `+6.21e-3`（梯度下降压低 logit）；阳性病例占 99% 时仍为 `+1.72e-5`。这说明在该合成机制下，空类 Dice 分量可盖过稀有阳性的 logit 救援信号。

在 `z=-10.55`，阳性像素从 4（全画布约 0.000095%）增至 419,430（10%）时，阳性 loss 仍约 0.99995，公共 shift 梯度都约 `-5.24e-5`；`z=-12.34` 时对应梯度约 `-8.75e-6`。对少量阳性，Dice 公共梯度并不会因其占比变小而增强；对同一低概率值，阳性梯度大小主要随 `p` 而不是 `K` 改善。

低 loss 的“吸引区域”出现在更负的均匀 logits，而不是上述均匀最大值处。例如 z=-20 时，空类 loss 为 `1.78e-6`；若 0.1% 阳性面积的病例只占 1% 或 10%，合成平均 Dice 分别约 `0.0100/0.1000`，而公共梯度仍分别约 `+3.53e-6/+3.21e-6`，方向继续压低 logit。这里的病例率只是灵敏度例子，不是该数据集的实际患病病例比例。

给出的 EX/SE 检查值是 `maxlogit=-10.55/-12.34`。最大值本身无法确定 Dice 的空间和：若所有像素都等于最大值，整画布空类 loss 分别为 0.997/0.889；若仅一个像素处于该最大值、其余像素为 -20，同一最大值下空类 loss 分别只有 `7.04e-5/3.69e-6`。因此“全背景低 loss 区域存在”已由公式确证，但该检查的 maxlogit 不能说明当前模型是否落在该区域。有效面积也改变 epsilon 区间：z=-12.34 时，V 为整画布 25%、50%、100% 的空类 loss 分别为 0.667、0.800、0.889；增强后 `valid` 面积需要按样本实测。

## KL 与 signed auxiliary 能否救援

若 teacher 在阳性区域比例 `f` 上给常值 `q=0.9`、其他位置为 0，student logit 均匀为 `z`，当前全像素 KL 对公共 logit 的梯度是 `p-0.9f`。在 `z=-10.55`，只有当 `f>0.00291%` 时该例的公共梯度才转为抬高 logit；在 `z=-12.34` 阈值为 `0.000486%`。例如 f=0.1% 时 KL 梯度方向是上抬，分别为 `-8.74e-4/-8.96e-4`（尚未乘 update 20k 的 0.5）；若 teacher 全图 q=0，则梯度分别为 `+2.62e-5/+4.37e-6`，方向是继续压低。全像素平均本身是标准概率蒸馏，不是实现 bug；它不能保证 teacher 在稀有 GT 区域有足够概率质量，也会在 teacher 同样漏检时跟随低输出。

signed auxiliary 的监督来源是两组 teacher 差值。它按 `1+4|d|` 加权，但再除以整张有效图的权重和；支持区稀少时整体梯度仍随支持比例缩小。零预测下，若一支差值图在 0.1% 像素上为 0.01，另一支目标为零，则 probe 的该头 shift 梯度经 update 20k `lambda_d=0.1` 后约 `5.2e-6`；差值为 0.1 时约 `7.0e-5`。若两个老师在 GT 漏检处一致，signed auxiliary 在这些点没有 GT 救援信号；此外其 auxiliary 梯度按当前路由不直接回传共享主干/基准 logits。因此它可能训练 private contrast head，但不应被当成主输出的阳性监督替代品。

## 确证与尚未证实

**确证的 loss 机制：**当前 Dice 将空类别纳入平均；其目标在全背景预测时随概率降低而降至 0，空类梯度最终也消失。低概率阳性样本的 Dice loss 接近 1，公共 logit 梯度约 `-2p`，继续降低 logit 会快速削弱阳性救援。历史 HRDecoder 的 Dice 使用相同 eps、平方概率分母和类平均，因而具有同类机制。全像素 KL 和 teacher-difference auxiliary 的空间均值也使稀有区域信号按面积缩小；KL 的方向还取决于 teacher 在这些位置的概率。

**不能据此确认的实际因果：**没有 20k checkpoint 的逐像素 logit 分布、每类 GT 阳性病例率和像素占比、逐样本 `valid` 数、Dice 分子/分母、teacher 在漏检 GT 上的均值/总概率质量，也没有 task Dice/KL/aux 对分类 bias 或末层权重的实际梯度分解。`maxlogit` 不是这些统计的替代。故上述机制足以构成明确的优化风险，但还不能断言它造成了 20k 的 EX/SE 零检出。

## 最小损失修复建议

先只改监督目标，不加模型分支：

1. Dice 的宏平均只纳入 `sum(valid*target)>0` 的样本/类别对，避免空类别 Dice 在稀有阳性任务中持续施加全图向下梯度。
2. 加两个有效像素掩码内的 logits BCE 分量，并分别按正像素和负像素归一化：阳性项为 `mean_pos softplus(-z)`，阴性项为 `mean_neg softplus(z)`；对无阳性类别跳过阳性项。这样阳性 logit 很低时 `dL+/dz≈-1`，而不是 Dice 的 `≈-2p`；阴性项保留背景校准。正负权重应在真实训练 batch 上按每类梯度范数校准，不从本合成探针硬编码。

仅把原 HRDecoder 的普通像素均值 BCE 加到现有 Dice 上，不是足够有把握的修复：在相同 uniform probe、z=-10.55、阳性病例率 10%、阳性面积 0.1% 时，均值 BCE 的公共梯度约 `-7.38e-5`，而现有 Dice 混合梯度为 `+6.21e-3`，仍由空类 Dice 决定方向。需要同时移除 Dice 的空类病例项，并让 positive BCE 按阳性像素归一化。先用只含 GT 的 B0/无 KD 基线验证 20k 前的每类召回，再逐一加入 KL 与 signed auxiliary；按 batch 记录正/负梯度范数和 teacher 阳性质量，才能区分是否还需要改蒸馏权重。

## 可复现产物

- `probe_loss_mechanism.py`：仅依赖 PyTorch CPU，复现当前与历史 Dice、小张量 autograd、2048² Dice/KL/signed-aux 闭式探针。
- `evidence/loss_mechanism_probe.json`：上述完整输入、梯度、病例阳性率、有效面积敏感性、相同 maxlogit 不同空间分布及解释限制。
