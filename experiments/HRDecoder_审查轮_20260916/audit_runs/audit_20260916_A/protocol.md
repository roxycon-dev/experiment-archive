# protocol.md —— 评价协议冻结（Phase A3）

> 生成时间：2026-09-16（CST）
> 依据：`mmseg/core/evaluation/lesion_metric.py`、`mmseg/datasets/lesion_dataset.py` 的真实实现（已逐行阅读，非按名字猜测）
> 源码 SHA-256：
> - `metric_lesion` = `2d8ff78ec6ffd4a2…`（6918 字节）
> - `dataset_lesion` = `96e697e26f15c477…`（4901 字节）
> 完整哈希见 `manifest.json`

---

## 1. 主指标定义（从真实代码反推）

### 1.1 输出通道与激活

- 模型是 **4 个独立 sigmoid 通道**（不是 softmax）。`num_classes = 5`（含背景），
  但 `pred_logit` 只有 **4 个通道**，代码用
  `assert pred_logit.shape[0] == num_classes - 1` 强制这一契约。
- 背景**没有独立 logit 通道**；4 个通道分别对应 `EX / HE / SE / MA`。
- 代码注释（第 9-18 行）说明标签取值 `{0,1,2,3,4}`，`0 = 背景`。

### 1.2 逐类混淆矩阵（真实公式）

`sigmoid_confused_matrix(pred_logit, raw_label, num_classes=5, thresh)` 对
`i ∈ {1,2,3,4}`（即 EX/HE/SE/MA）执行：

```
pred  = pred_logit[i-1] > thresh          # ← 严格大于，不是 >=
label = raw_label == i
TP_i  = Σ_pixels (label ∧ pred)
P_i   = Σ_pixels pred                     # 即 TP+FP，供该类使用
FN_i  = Σ_pixels label − TP_i             # 即 TP+FN−TP = 漏检数
TN_i  = Σ_pixels ¬(label ∨ pred)
```

### 1.3 聚合方式：**全数据集累计后相除**（不是逐图平均）

`sigmoid_metrics` 对每张图调用上面的函数，然后把 `P / TP / FN / TN`
**跨全部图像累加**，最后一次性相除：

```
total_P[c]  = Σ_i P_i,c
total_TP[c] = Σ_i TP_i,c
total_FN[c] = Σ_i FN_i,c

ppv[c] = total_TP[c] / total_P[c]
s[c]   = total_TP[c] / (total_TP[c] + total_FN[c])
f1[c]  = 2·total_TP[c] / (total_P[c] + total_TP[c] + total_FN[c])
IoU[c] = total_TP[c] / (total_P[c] + total_FN[c])
```

等价形式：`IoU[c] = TP / (TP + FP + FN)`，其中 `FP = P − TP`。

**这是数据集级累计 IoU，不是"逐图 IoU 再平均"。** 两者差值可观测（见 §4 的手算样例 S2）。

### 1.4 阈值

- 模块级常量 `SIGMOID_THRESH = 0.5`。
- `compute_aupr=True` 时阈值集合是 `np.linspace(0, 1, 11)`（11 个点），
  并取 `index = argmax(threshs == 0.5)`（= 第 6 个点，index 5）作为**主指标所用阈值**。
- `compute_aupr=False` 时阈值集合是 `[0.5]`，`index = 0`。
- **两种情况主指标用的都是 0.5**，且比较运算都是**严格 `>`**。

> **冻结结论：阈值 = 0.5，比较符 = 严格 `>`。不得更改。**

### 1.5 mIoU 的类别范围与 NaN 处理

`lesion_dataset._evaluate`：

```
iou, f1, ppv, s, aupr, ae = lesion_metrics(results, gt_seg_maps, num_classes,
                                           ignore_index=self.ignore_index)
mIoU = np.nanmean(np.nan_to_num(iou[-4:], nan=0))
```

- `iou` 长度 5（含背景），**`iou[-4:]` 取最后 4 个，即 EX/HE/SE/MA，背景被排除**。
- 某类为 NaN（该类的 `TP+FP+FN == 0`，发生 `0/0`）时**置 0**，而不是丢弃该类。
- `np.nanmean` 对 4 个元素求平均 → **四个类权重相同**。

### 1.6 单位与返回范围

- `lesion_metrics` 返回的是 **0–1** 的 IoU。
- `_evaluate` 把它原样存进 `eval_results['mIoU']`（**0–1**），
  仅**打印** `summary_str` 时 `'{:.2f}'.format(mIoU * 100)`。
- `tools/test.py` 输出的 `global 42.58` 是**百分数**（打印值）。
- **冻结结论：内部量纲 0–1，报告口径百分数。混用会造成 100 倍错误。**

### 1.7 冻结后的完整公式

```
对 c ∈ {EX, HE, SE, MA}（对应通道 0,1,2,3；对应标签值 1,2,3,4）:

  TP_c = #{(img, px) : gt[img,px] == c  ∧  prob[img, c, px] > 0.5}
  FP_c = #{(img, px) : gt[img,px] != c  ∧  prob[img, c, px] > 0.5}
  FN_c = #{(img, px) : gt[img,px] == c  ∧  prob[img, c, px] <= 0.5}

  IoU_c = TP_c / (TP_c + FP_c + FN_c)          # 分母为 0 时记 0（经 nan_to_num）

  mIoU = (IoU_EX + IoU_HE + IoU_SE + IoU_MA) / 4
```

**注意 `FP_c` 的真实定义**：代码写的是 `P_c = Σ pred`，其中 `pred = prob[c] > 0.5`，
而 `FN_c = Σ label − TP_c`。于是 `P_c + FN_c = TP_c + FP_c + FN_c`。这两条路径等价
**仅当** `Σ pred` 覆盖所有像素（含 `gt==255`）。见 §2。

---

## 2. ⚠️ 发现的实际实现问题：`ignore_index` 被传入但从未使用

**事实（源码级证据）：**

- `lesion_metrics(results, gt_seg_maps, num_classes, ignore_index=None, nan_to_num=None)`
  —— 第 173 行定义了 `ignore_index` 形参。
- 但**函数体内一次都没有引用 `ignore_index`**（第 178–205 行只有
  `compute_aupr` / `use_sigmoid` 的分支与 `ppv/s/f1/iou` 的计算）。
- 它只是被原样透传的名字；下游 `sigmoid_metrics` / `sigmoid_confused_matrix`
  的签名里**都没有**这个参数。
- 调用侧 `mmseg/datasets/lesion_dataset.py` 第 71-72 行确实传了
  `ignore_index=self.ignore_index`。

**推论：**

- `label = raw_label == i` 对 `raw_label == 255` 恒为 `False` → 255 像素**不贡献 FN**。
- 但 `pred = pred_logit[i-1] > thresh` 与 `raw_label` 无关 →
  **如果模型在 255 区域预测为阳性，该像素会计入 `P_i` 从而计入 FP。**

**后果**：`gt==255` 的区域被隐式当作**负样本**（而非"忽略"）。

**影响范围（待实测确认）：**

- 若 val/test 标签中不存在 255，则对已报告分数**无影响**。
- 训练时的 `RandomRotate` 填充使用 `seg_pad_val=255`，所以**训练路径**会大量出现 255；
  但训练用的是 `binary_loss.BinaryLoss`（另一条实现），不走 `lesion_metrics`。
- **需要在 Phase C4 用真实标签实测 val/test 是否存在 255**，才能判定历史分数是否需要重算。
  → 记为 **需验证项 V1**，未验证前不得声称"历史成绩无误"或"历史成绩作废"。

> 按 prompt §3.3：「如果真实实现和历史声明不同：先报告不一致，保留 legacy 结果，
> 另设 corrected 结果」。**本项现在只报告不一致，不改代码、不重算。**

---

## 3. 其他需要记录的实现细节

### 3.1 AUPR 的计算路径存在待核点

```python
maupr = np.zeros((num_classes,), dtype=np.float64)      # 长度 5
...
ppv_list = np.nan_to_num(total_tp_list / total_p_list, nan=1)
s_list   = np.nan_to_num(total_tp_list / (total_tp_list + total_fn_list), nan=0)
if compute_aupr:
    for i in range(1, len(maupr)):          # i = 1..4
        x = s_list[:, i]                     # 召回率，随阈值递增而单调递减
        y = ppv_list[:, i]
        maupr[i] = auc(x, y)
```

- `x`（召回率）随阈值从 0→1 是**单调递减**的。`sklearn.metrics.auc` 要求 `x`
  单调；递减时是否被接受取决于 sklearn 版本。
- `aupr[0]`（背景）保持 0，且 `_evaluate` 只取 `aupr[-4:]`，所以背景不参与平均。
- **本条只作记录**：`mAUPR` 是辅助指标，不是主指标。冻结结论是
  **主指标只用 `mIoU`，AUPR 不得用于任何选择或比较**。
  → 记为 **需验证项 V2**。

### 3.2 输出回原图的路径（影响指标的其他环节）

在 `lesion_encoder_decoder.py` 的 `inference()` 中：

```python
output = logit_activation(seg_logit, self.use_sigmoid)   # 先激活成概率
flip = img_meta[0]['flip']
if flip:
    flip_direction = img_meta[0]['flip_direction']
    if flip_direction == 'horizontal':  output = output.flip(dims=(3,))
    elif flip_direction == 'vertical':  output = output.flip(dims=(2,))
```

**顺序是：先输出到原图尺寸/激活，再做逆翻。** 这条路径与 `HRPlusRefinement`
覆写的 `inference()`（tiled student + reference 融合）**不是同一个函数**，
必须在 Phase C3 用 hooks 分别验证（见 §5 的 R1）。

### 3.3 mMAE 是辅助指标

`mMAE = np.nanmean(np.nan_to_num(ae[-4:], nan=0))`，其中
`class_ae[j] = mean_absolute_error(label, pred_logit[j-1])`，
`label` 是 bool（`gt==j`），`pred` 是 float 概率。报告里以 `MAE(e-2)` 打印。
**不参与主指标。**

---

## 4. 独立 oracle 必须覆盖的手算样例（prompt §8.1）

`lesion_metric.py` **不能自己验证自己**。下表是实现 `tools/oracle_metrics.py`
（纯 NumPy，不 import 被测模块）必须通过的最小集合：

| ID | 样例 | 期望 | 说明 |
|---|---|---|---|
| S1 | TP=1, FP=1, FN=1 | IoU = 1/3 | 基本定义 |
| S2 | 图1: TP=1,FP=0,FN=0；图2: TP=0,FP=9,FN=0 | 累计 IoU = 1/10 | **不是**逐图平均的 1/2 → 验证 §1.3 |
| S3 | GT 有阳性、预测全阴性 | IoU = 0 | |
| S4 | GT 全阴性、预测有阳性 | IoU = 0（TP=0） | |
| S5 | GT 与预测全阴性 | 分母 0 → NaN → 经 `nan_to_num` 记 0 | 冻结空并集规则 |
| S6 | 单个 MA 像素且预测完全正确 | MA IoU = 1 | 不能被 resize/类别映射吞掉 |
| S7 | 同一像素两个独立通道均为阳性 | 两通道分别计数（本协议支持多标签） | 因为各通道独立 sigmoid |
| S8 | 概率恰好 0.5 | **判为阴性**（`>` 不是 `>=`） | 验证 §1.4 |
| S9 | 概率 0.5+ε / 0.5−ε | 分别判阳/阴 | 阈值邻域 |
| S10 | 四类分别只在四个不同位置阳性 | 四类 IoU 均为 1；交换类序后应失败 | 捕获通道/HWC 错位 |
| S11 | ignore 位置（若确认为 255）预测任意变化 | **当前实现下 TP/FN 不变、FP 会变** | §2 的直接后果 |
| S12 | 图像顺序重排 | 累计计数不变 | 验证聚合与顺序无关 |

**S11 是本轮最重要的 oracle 用例**：它把 §2 的静态推断变成可复现的实验事实。

---

## 5. 本轮必须冻结、任何实验都不得改动的清单

| 项目 | 冻结值 | 依据 |
|---|---|---|
| 主指标 | `mIoU` = 4 类 IoU 的算术平均 | `_evaluate` 的 `nanmean(iou[-4:])` |
| 类别顺序 | EX, HE, SE, MA ↔ 通道 0,1,2,3 ↔ 标签 1,2,3,4 | `CLASSES` 与 `pred_logit[i-1]` |
| 背景 | **不计入平均** | `iou[-4:]` |
| 阈值 | **0.5** | `SIGMOID_THRESH` |
| 比较符 | **严格 `>`** | `pred_logit[i-1] > thresh` |
| 聚合 | **全数据集累计** | §1.3 |
| 量纲 | 内部 0–1，报告 ×100 | §1.6 |
| 空并集 | NaN → 置 0 | `nan_to_num(iou[-4:], nan=0)` |
| 数据划分 | 454 / 76 / 227，不可合并 train+val | 用户约束 + `phaseA4` PASS |
| 分辨率管线 | 沿用原配置（2048,2048），tile=1024, stride=768 | 原 `test_cfg` / `hr_settings` |
| TTA | 由本轮协议另行登记，不得事后挑 | prompt §11.1 |

## 6. 未验证项（不得写成 PASS）

| ID | 内容 | 为何现在不能结论 | 计划在哪个阶段验证 |
|---|---|---|---|
| V1 | val/test 标签是否含 255（决定 §2 是否影响历史分数） | 需解压数据、实测标签值域 | C4 |
| V2 | `auc(x,y)` 在 x 单调递减时是否按预期工作 | 依赖服务器 sklearn 版本 | C4（非主指标，低优先） |
| V3 | `num_classes` 在真实调用中是 5 还是 4 | 本地无 mmcv，无法解析 config 与构建 dataset | C1 |
| V4 | `HRPlusRefinement.inference` 与 `LesionEncoderDecoder.inference` 的真实调用点 | 需在真实 runner 里用 hooks 捕获 | C3 |
| V5 | `compute_aupr=True` 时 `index=argmax(...)` 命中的确实是 0.5 | 静态可推（linspace(0,1,11) 第 6 点是 0.5），但仍需运行时断言 | C1 |

---

## 7. 一条命令复算关键结论

```bash
# 独立 oracle（纯 NumPy，不 import 被测模块）
python tools/oracle_metrics.py --out tests/loss_ignore_metrics.json
# 期望：12 个用例全部 PASS；S2 必须给出 0.1 而不是 0.5
```
