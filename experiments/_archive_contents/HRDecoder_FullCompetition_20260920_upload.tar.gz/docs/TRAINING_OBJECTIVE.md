# TRAINING_OBJECTIVE — 训练目标与指标的精确定义

本文件是**损失与指标口径的唯一文字权威**。`trainer/loss.py` 与 `trainer/metrics.py`
逐条实现本文；`tools/preflight_training.py` 用合成数据对手算值做一致性断言。
任何实现与本文冲突时，先改本文（并说明理由），再改代码——不允许两边各说一套。

日期：2026-09-20 ｜ 状态：已实现并通过合成手算校验（CPU）

---

## 1. 任务形式

对一张眼底图 `x ∈ R^{H×W×3}`，模型输出 **4 通道独立 logits**
`z ∈ R^{4×H×W}`，通道序固定

```
通道 0 = EX（硬性渗出）   通道 1 = HE（出血）
通道 2 = SE（软性渗出）   通道 3 = MA（微血管瘤）
```

- 4 个通道是**互相独立的二分类**，没有背景通道，也没有 softmax。
- 通道序的权威定义在 `tools/contract_common.py:LESION_CHANNEL_ORDER`，
  与 DDR 融合标签的 class id 对齐关系为 **通道 i ↔ class id i+1**
  （`EX=1, HE=2, SE=3, MA=4`，逐字来自 `D:\HRDecoder\tools\convert_dataset\ddr.py`）。
- 评估端对齐规则来自 `mmseg/core/evaluation/lesion_metric.py` 的
  `sigmoid_confused_matrix`：`pred = pred_logit[i-1] > thresh`，`label = raw_label == i`。
  两侧一致 ⇒ 训练通道序无需置换。

## 2. 监督信号的三元组 `(target, valid)`

每个样本带两个同形状张量：

| 张量 | 含义 | 取值 |
|---|---|---|
| `target[c,y,x]` | 该像素是否属于病灶 c | 0/1 |
| `valid[c,y,x]` | 该像素的类别 c **是否被标注** | 0/1 |

`valid` 的三种来源（`data_contract/manifest.schema.json`）：

- `full`：该类全图已标注 ⇒ `valid ≡ 1`（DDR 四类都属于这一种）；
- `file`：逐像素有效栅格 ⇒ `valid` 取栅格非零处；
- `none`：该类**完全未标注** ⇒ `valid ≡ 0`，`target ≡ 0`。

**核心科学约束：`valid=0` 的像素既不是正样本、也不是负样本，必须整类屏蔽。**
"数据集里没有这个病灶的标注" ≠ "这张图没有这个病灶"。把缺类当阴性会让模型
学会输出 0，从而在外部数据上系统性地压低该类召回。

## 3. 损失：masked BCE + masked soft Dice

记 `p = σ(z)`（sigmoid 只做一次）、`smooth = ε = 1e-5`、`n_c = Σ_{y,x} valid[c,y,x]`。

### 3.1 每类 masked BCE（先按该类有效像素数归一）

```
BCE_c = ( Σ_{y,x} BCEWithLogits(z[c,y,x], target[c,y,x]) · valid[c,y,x] ) / max(n_c, 1)
```

- 使用 `binary_cross_entropy_with_logits`（数值稳定，不显式 sigmoid）；
- `valid=0` 位置乘 0 从分子剔除；
- 分母是**该类自己的有效像素数**，不是 `H·W`，也不是 batch 内像素总数。

### 3.2 每类 masked soft Dice（valid 只作用一次）

```
p̃ = p · valid        t̃ = target · valid
Dice_c = 1 − ( 2·Σ p̃·t̃ + ε ) / ( Σ p̃ + Σ t̃ + ε )
```

`valid` 在分子分母各乘一次，等价于"只在有效区域内算 Dice"。
`ε` 取旧已验证定义 `1e-5`（与 `BinaryLoss` 的 dice 分支同值，保证 H/U 两臂口径一致）。

### 3.3 有效类别等权平均（全无效类不进分母）

```
has_valid_c = 1[ n_c > 0 ]        N_active = Σ_c has_valid_c
L = ( w_ce · Σ_c has_valid_c·BCE_c  +  w_dice · Σ_c has_valid_c·Dice_c ) / max(N_active, 1)
```

- **分母是有效类别数，不是常数 4。** 若某类在整批里都未标注，它既不贡献损失
  也不稀释其他类的权重；等价于"这类这一批不参与训练"。
- 全无效类**梯度恒为 0**（分子分母同时为 0 且乘 `has_valid_c=0`），不会引入 NaN。
- 若**整批 4 类全无效**：显式返回 `L = Σ 0·z`（数值 0、保留计算图、梯度为 0）。
  绝不静默丢弃该 batch，也绝不返回 NaN。
- `w_ce = w_dice = 1.0`（配置 `loss.ce_weight` / `loss.dice_weight`）。

### 3.4 与旧实现（`BinaryLoss`）的差异

| 项 | 旧 `BinaryLoss` | 本实现 |
|---|---|---|
| 缺类处理 | 无 `valid` 概念 ⇒ 缺类被当阴性 | `valid=0` 整类屏蔽 |
| 归一化 | 逐图/逐 batch 平均 | 先按每类有效像素归一，再对有效类别等权平均 |
| 全无效 batch | 会产生 NaN/0 除 | 显式零损失零梯度 |

> 差异是有意的：`configs/*.yaml` 的 `loss.per_class_valid_masking` 已置 `true`，
> 表示"新入口必须实现 valid 屏蔽"，而不是沿用旧语义。

## 4. 指标：整图还原后累计

协议名 `metrics.protocol = whole_image_accumulated_counts`，实现在 `trainer/metrics.py`，
整图推理路径在 `trainer/eval.py`。

### 4.1 像素级累计（每类独立）

记阈值 `τ = 0.5`，`V = valid > 0.5`，`P = (p > τ) ∧ V`：

```
TP_c = Σ [ P ∧ (target > 0.5) ∧ V ]
FP_c = Σ [ P ∧ (target ≤ 0.5) ∧ V ]
FN_c = Σ [ ¬(p > τ) ∧ (target > 0.5) ∧ V ]
```

- **累计跨整张验证集**（不是逐图求指标再平均）——这是与旧
  `sigmoid_confused_matrix` + `total_tp/total_fn` 一致的口径；
- **阈值是严格 `p > 0.5`**：`p == 0.5` 判为阴性（与旧 `pred_logit > thresh` 一致）；
- `V=0` 的像素不计入任何桶（既不产生 FP 也不产生 FN）。

派生指标：

```
IoU_c = TP/(TP+FP+FN)      F1_c = 2TP/(2TP+FP+FN)
PPV_c = TP/(TP+FP)         Recall_c = TP/(TP+FN)
```

主指标 `mIoU_lesion4 = mean(IoU_c, c ∈ {EX,HE,SE,MA})`，**背景不参与**。
分母为 0 的类记 `NaN` 并从均值中剔除，同时写入 `excluded_classes` 显式记录，
**不冒充 0**（冒充 0 会人为压低 mIoU，冒充 1 会人为抬高）。

### 4.2 AUPR

每类维护 256-bin 概率直方图（正像素/负有效像素分开累计），按阈值自高到低扫出
PR 阶梯曲线，取 **recall 递增方向** 的右黎曼阶梯积分：

```
AP_c = Σ_i (R_i − R_{i−1}) · P_i        R_{−1} = 0
```

无正样本像素的类返回 `NaN`（并记入 `excluded_classes.aupr`）。
这是对连续 PR 曲线的分箱近似，输出中带 `aupr_note` 如实标注。

> ⚠️ 历史 bug：旧实现把 `precision`/`recall` 反转之后再按 `R` 递增方向累加，
> 等价于在 **recall 递减** 方向做增量 → 完美分类器只能得 ≈0.2（真值 1.0）。
> 已修正，并在 `tools/preflight_training.py::check_eval_path` 与
> `tests/run_training_smoke.py::T11` 加了"完美分类器 AUPR == 1.0"的断言。

### 4.3 为什么验证必须是整图（或滑窗还原）

1. 最终 DDR test 评估是整图口径，训练期验证必须同口径才可比较；
2. 训练 crop 由"前景混合采样"挑出（`fg_prob=0.5`），病灶像素占比被显著抬高，
   用 crop 算 PPV/IoU 会系统性高估召回；
3. 因此 `trainer/eval.py::require_whole_image_evaluator` 在 **formal 模式**下
   缺 evaluator 或缺类型不匹配时直接 fail closed，训练一个 update 都不跑。

整图路径的实现细节：

- **绝不 resize 输入**：先按 `mean/std` 归一化，再 pad 到 32 的倍数
  （`PAD_DIVISOR`），pad 值为 0（等价补"均值色"，避免黑边扰动 HRNet 的 BN 统计）；
- 推理后把 logits 裁回原始 `H×W` 再累计；pad 区域在指标里由 `valid=0` 排除；
- `metrics.eval_mode = sliding_window` 时按配置给出的 `window_size`/`window_stride`
  滑窗推理，重叠区域对 **logits** 取算术平均（覆盖计数归一），
  窗口/步长**必须显式配置**，不猜默认值，且必须是 32 的倍数。

## 5. 优化目标相关约定（与损失同属"训练目标"）

| 约定 | 取值 | 说明 |
|---|---|---|
| 参数组 | `backbone` / `decoder` 两组 LR | H：HRNet 主干 vs FCN 头；U：ConvNeXt 编码器 vs U-Net 解码器 |
| 优化器 | AdamW | `optimizer != adamw` 直接报错（其他优化器需另行验证） |
| 梯度累积 | 以 **optimizer update** 计数 | 累积窗口跨 epoch 连续，不丢半窗、不补步 |
| LR 日程 | linear warmup + cosine，**按 update** | `warmup_updates = round(warmup_ratio × updates)` |
| BN 策略 | HRNet 主干统计量冻结、affine 可训练 | 每次 `model.train()` 后立刻重新冻结 |
| AMP | `mixed_precision=true` 时启用 GradScaler | 只在 CUDA 上启用；裁剪在 `unscale_` 之后 |
| 随机性 | `seed` 固定 python/numpy/torch + cudnn deterministic | 全部 RNG 态写入 checkpoint |

## 6. 本目标**不**声称的事情

- 不声称该目标会使 mIoU 上升。它只保证"监督信号与评估口径自洽"，
  架构/预训练是否带来增量由实验数据回答（H0/H1、U0/U1 配对）。
- 不声称 loss 数值与非等权/带背景的方案可比。
- 不声称 256-bin AUPR 等同精确 AP（是分箱近似，已在输出中标注）。
