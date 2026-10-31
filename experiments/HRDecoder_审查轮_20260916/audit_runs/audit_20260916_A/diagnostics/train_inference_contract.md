# train_inference_contract.md —— 训练目标与部署预测的逐项对照（prompt §9.1）

> run_id: `audit_20260916_A`　生成：2026-09-16（CST）
> 结论定性：**已证实存在结构性不一致（事实）**；它是否是性能瓶颈的**主因**仍是**待检验假设**。
> 全部数字来自 `diagnostics/val_blend_decomposition.json`（val 76 张，本轮实测）与
> `hrplus_patch.py` 的源码（已逐行阅读）。

---

## 1. 逐项对照表（§9.1 要求）

| 项目 | 训练（`forward_train`） | 验证/部署（`inference`） |
|---|---|---|
| 图像尺寸 | **1024×1024 crop**（`ForegroundMixedCrop(crop_size=1024)`，在 Resize(2048) 之后裁） | **2048×2048 整图**（pipeline `Resize(keep_ratio=False)`） |
| 裁剪 | 50% 前景锚定 / 50% 均匀随机裁块 | 无裁剪（整图） |
| 归一化 | train 管线 `Normalize`（同一组 mean/std） | test 管线 `Normalize`（同组） |
| **semantic 来源** | 该 crop 经 `encode_decode` 得到的 `lr_feat`（**crop 局部上下文**） | 每个 tile 各自的 `lr_feat`（**tile 局部上下文**） |
| **MA 头位置** | `encode_decode` 内 `refine_logits` 之后（融合**前**） | 同左（融合**前**） |
| **监督对象** | **单块 crop 的 logits**（`decode_head.losses(dice_logits, gt)`） | — |
| **reference 是否参与** | **完全不参与**（`forward_train` 只调用 `self.encode_decode`） | **参与**：`0.5·P_ref + 0.5·P_student` |
| **tile / stride** | 1024 输入 → `hr_slide_inference` 仅产生 **1 个 tile**（无重叠） | tile=1024、**stride=768** → **多个重叠 tile** |
| **重叠归一化** | 无（单 tile） | `tiled_forward` 内部对重叠区做平均 |
| resize / 激活 / 融合顺序 | `encode_decode` → loss（`torch.where(gt==255,-1e4,·)` 后送 Dice） | `tiled_forward` → 裁剪到 `img_shape` → `resize_logit(ori_shape)` → `sigmoid` → flip 逆变换 → 与 reference 概率融合 |
| 翻转 | RandomFlip（训练增广） | 仅 TTA 时；模型内部 `_student_prob` 做逆翻 |

**两处结构性不一致：**

- **不一致 A（融合）**：训练时学生**从未见过 reference**，但部署时它的输出被 0.5/0.5 混合。
- **不一致 B（重叠平均）**：训练时是单 tile 的原始 logits；部署时是 stride=768 重叠平均后的 logits。学生从未在"被平均过"的表示上被监督。

---

## 2. 融合的解析后果（在已证实的同坐标线性概率融合前提下）

`HRPlusRefinement.inference` 的实现是：

```python
return blend_probs(prob_r, prob_s, weight_a=0.5)     # = 0.5*prob_r + 0.5*prob_s
```

`blend_probs` 是**逐元素线性**的，两侧都已 sigmoid 到概率，且都经同一条
`resize_logit → ori_shape` 路径回到原图坐标。**因此下面的推导成立。**

在 τ = 0.5、比较符为**严格 `>`** 时：

```
P_final = 0.5·P_s + 0.5·P_r > 0.5   ⟺   P_s > 1 − P_r
```

- **若 P_r = 0** → 需要 `P_s > 1`。sigmoid 的值域是开区间 (0,1)，
  **该像素在数学上无法被判为阳性**。
- **若 P_r = 1** → 需要 `P_s > 0`，**恒成立**，学生无法纠正该像素的阳性判定。
- **若 P_r = 0.9** → 学生只需 `P_s > 0.10`；reference 主导。
- **若 P_r = 0.1** → 学生必须 `P_s > 0.90` 才能翻盘。

---

## 3. 在 val 上的实测（本轮，76 张，401,111,888 像素）

### 3.1 三种判定路径的指标

| 判定路径 | mIoU | EX | HE | SE | MA |
|---|---:|---:|---:|---:|---:|
| student-only | 0.3934 | 0.4024 | 0.4305 | 0.5127 | **0.2278** |
| reference-only | 0.3921 | **0.4425** | 0.4155 | 0.5162 | 0.1940 |
| **blend 0.5/0.5（实际部署）** | **0.4161** | 0.4404 | **0.4473** | **0.5508** | 0.2262 |

**读数：**
- 融合整体**是增益**（0.4161 vs 0.3934 / 0.3921），且在 **HE 与 SE 上同时超过两个单体**
  （HE 0.4473 > max(0.4305, 0.4155)；SE 0.5508 > max(0.5127, 0.5162)）→ 存在真实协同。
- **但融合对 MA 是负担**：0.2262 < student-only 的 0.2278。
  → 在 MA 这一类上，"融合"没有增益，反而略微拖后腿。

### 3.2 不可纠正性（核心量化）

对 MA 的每一类错误像素，计算它要翻盘所需的 `required_P_student = 1 − P_reference`：

| 错误类型 | 像素数 | 所需 P_s 中位数 | ≥0.9 的比例 | ≥1.0 的比例 |
|---|---:|---:|---:|---:|
| **MA 漏检（FN）** | **45,505** | **0.970** | **70.9%** | 0.000 |
| MA 误检（FP） | 35,852 | 0.392 | 1.2% | 0.000 |

`reference` 输出精确等于 0 或 1 的像素数均为 **0**（共 4.01 亿像素），
所以**不存在数学上绝对不可达的像素** —— 但：

> **70.9% 的 MA 漏检像素，需要学生把该像素的概率推到 ≥0.9 才能在 0.5 融合下翻盘。**

这是一个**可量化的结构性瓶颈**，而且它解释了为什么"改进 MA 头"这类做法收益有限：
学生即使在自己那一侧显著变好，只要 `P_r` 极低，最终判决仍由 reference 拉回。

**对照**：MA 误检只需 `P_s` 中位数 0.392 即可纠正 → **误检可纠、漏检难纠**，
而历史上的 MA 问题主要被描述为"漏检/敏感度不足"（E4 把召回推高却 IoU 不动）。

---

## 4. 这**不**能证明什么（严格遵守 §0.2 的三种判断）

- ❌ 不能证明"融合是 MA 差的根因"。本节只证明**存在结构性不一致 + 一个可量化的约束**。
  仍未排除的替代解释：MA 标签本身的可分性上限、优化问题、`P_r` 低是因为 B 本身在 MA 上弱。
- ❌ 不能证明"去掉融合就能涨"。**未做**该实验，且 §10.2 明确要求不得把多个改动混在一起。
- ❌ 不能把 val 结论外推到 test。**test 在本轮只做过标签值域审计，未做任何性能评估。**
- ❌ 不能声称 `P_r` 是"硬上限"。如 §9.1 所述：`P_r` 非零时理论可达性与
  sigmoid 饱和/优化困难是不同问题。

---

## 5. 由本节产生的待检验假设（用于 §10.1 的 GO/NO-GO）

**H1（首选）**：若把 MA 的监督改为"部署一致的输出"（即在训练时就按部署的
融合与重叠平均形式计算 MA loss），同样的更新预算下 MA 的 val 表现会变好。

- **支持证据**：§1 的两处不一致是**代码级事实**；§3.2 显示 70.9% 的 MA 漏检落在
  "student 需要 ≥0.9"的区域，说明 student 被监督的目标与它被判决的方式不同。
- **需要先解决的技术障碍**（§10.2 已列明）：
  1. 整图 reference 的上下文与 crop reference 不等价 → 不能用"相同 crop 跑 B"冒充；
  2. tile 内独立算 loss ≠ 对重叠融合后完整预测算 loss → 近似必须写清；
  3. 若用 `no_grad` 包住整个可训练计算会切断 MA 梯度。**（当前实现里
     `_reference_prob` 内部有 `torch.no_grad()`，但它只包住 reference，学生侧不经过它，
     所以现有实现不违反这一条；C1 若改动需重新核对。）**
- **否决条件**：若在 train 小样本过拟合探针里，部署一致监督**不能比原监督更好地
  降低它自己监督的目标**，则 H1 不成立。

**H2（次选）**：HE 的局部纠错。本轮观测到 blend 在 HE 上产生协同（0.4473），
说明 HE 存在可利用的互补性；但**尚未做** §9.2 要求的逐图 TP/FP/FN 与连通域诊断，
故**证据不足，暂不提出**。

---

## 6. 复算命令

```bash
cd <repo> && export PYTHONPATH=$PWD
python ops/diag_train_inference_contract.py \
  --config configs/lesion/exp/hrplus_patch_dice.py \
  --ckpt <D@2000 完整 checkpoint> \
  --out-dir <audit_runs/audit_20260916_A>
# 期望：val 76 张；student mIoU≈0.3934 / reference≈0.3921 / blend≈0.4161
#       MA_FN median required_P_student ≈ 0.970，≥0.9 比例 ≈ 0.709
```
