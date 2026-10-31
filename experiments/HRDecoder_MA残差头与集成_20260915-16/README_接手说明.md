# 接手说明：HRDecoder MA 残差头与集成实验

> **作者**：前一位 AI 助手（本次会话）
> **时间**：2026-09-15 22:20 → 2026-09-16 10:55（北京时间）
> **目标**：把 D:\dataset\experiment 下既有工作的 test mIoU 从 **42.20** 提到 **45.35**（+3.15）
> **结果**：**未达成。最好 43.229，缺口 2.12。**
> **约束（用户明确要求）**：**不允许引入任何外部数据集**，必须沿用现有 454/76/227 划分

---

## 0. 三十秒摘要

- 我承接的是前一轮诊断报告（`HRDecoder_诊断调研_20260910`）里 Codex 押注的方向：**「冻结基准网络 + 只训练语义条件 MA 残差头」**。
- **我把这个设计第一次真正实现了**（此前只跑过一个被稀释的近似版），并用硬性结构不变量证明它按设计工作：**MA 类 +5.27 点，其余三类逐位零变化**。
- 但它**加在已经充分微调的模型上没有增量**——说明 MA 类存在约 **val 22.6 / test 25** 的平台，微调与残差头两条路都收敛到那里。
- 我另外验证了 7 条数据无关的杠杆：翻转 TTA（有效）、多模型集成（有效，最大增益来源）、多尺度 TTA（单模型有效但不能与集成叠加）、BN 统计重估（有害）、Lovasz 损失（有害）、难负例损失（无效）。
- **结论：43.2 是当前架构 + 这份数据的现实天花板。** 要跨过 2.1 点，需要换骨干、重新审视 MA 评估口径，或扩大验证集。

---

## 1. 我要解决的困难

### 1.1 目标与现实的量级差距

| 项目 | test mIoU |
|---|---:|
| 历史 baseline（报告数字，**从未重跑**） | 42.20 |
| 项目此前最好单模型（记为 D@2000） | 42.575 |
| **目标** | **45.35** |
| 缺口 | **+2.78**（相对历史最好单模型） |

过去全部实验里，**单次最大 test 提升是 +0.38**（第三轮 D 方案）。目标要求的是它的 7 倍。

### 1.2 验证集不可靠，选择信息传不到测试集

- val 只有 **76 张**，且各类像素比例在 train/val/test 之间差异极大（EX 像素占比 train/val/test = 0.2466% / 0.0728% / 0.1337%）。
- 已知两次 val→test 排序反转：
  - 逐类路由：val 39.90 → test **41.82**（HE 模型优劣反转）
  - D 方案：val **+2.23** → test 只有 **+0.38**
- **后果**：任何在 val 上做的模型选择，都无法可靠外推。这也是为什么本项目的 test 已经被多次查看（至少 4 次）——这个事实必须在论文里披露。

### 1.3 MA 类极难，但它是 4 倍杠杆

```
mIoU = (EX + HE + SE + MA) / 4
当前最好：53.86 / 44.51 / 49.82 / 24.73 → 43.23
```

MA 只有 24.7，全场最低。**MA 每涨 1 点，mIoU 涨 0.25**（其他类也是 0.25，但它们已在 44-54 高位，空间小）。所以"要不要押注 MA"是理性的，问题是"押注能否奏效"。

### 1.4 既有代码里的两个隐藏陷阱

- **标准 TTA 路径在这个仓库里是坏的**（详见 §5.3）
- **`HRPlusRefinement` 的 `source_checkpoint` 同时决定 student 和 reference**，所以不能直接用它以 D 为基准跑实验（详见 §5.1、§5.2）

---

## 2. 我做了什么

### 2.1 第一步：多子代理交叉审查代码（动 GPU 前）

派出 3 个独立子代理分别审查核心算子、模型结构、评估指标链路。

**报告了 2 个"阻断项"，经实证全部是误报：**

| 子代理的"阻断项" | 实证结论 |
|---|---|
| `load_state_dict(strict=True)` 会因 checkpoint 含 `ma_refiner.*` 而崩溃 | ❌ 误报。直接读 checkpoint：顶层键是 `backbone/decode_head/detail_branch/detail_head/detail_scale`，**`ma_refiner` 相关键数量 = 0**（B 的训练配置就是 `semantic_refinement=False`） |
| DDP `find_unused_parameters=False` 会崩 | ❌ 误报。`tools/train.py` 默认 `--launcher none` → `distributed=False` → 走 **MMDataParallel，根本不走 DDP** |

**真实存在的遗留项（已核实为非阻断）：**

| 疑点 | 核实结论 |
|---|---|
| `self.align_corners` 可能为 None | ✅ 实测为 `False` |
| `seg_fields` 未同步裁剪辅助标签 | ✅ 训练管线只有 `gt_semantic_seg` 一个标签字段，且永远被显式裁剪 |
| 评估侧 `ignore_index` 未透传，255 被当 FP | ✅ 抽样 7466 万像素，**原始数据 0 个 255**（只来自训练时旋转填充） |
| Dice `smooth=1e-5` 梯度弱 | ⚠️ 真实存在，但 D/L 相同，不影响单变量对照 |

> **方法论教训：子代理的结论必须用实证证伪，不能直接采信。** 如果当时采信了那两个"阻断项"，会去做无用的修复。

### 2.2 第二步：建立可信的评估器（本次最重要的方法论资产）

原仓库的 `tools/test.py` 只能测单模型、单尺度、无 TTA。我自建了 `03_代码/ops_eval_ens.py`：
- N 模型加权概率融合
- 翻转 TTA
- 多尺度 TTA（`--scales`）
- 复用仓库自带 `dataset.evaluate()` → `lesion_metrics`，**保证指标口径与官方完全一致**

**三重验证**（这是判断我所有数字是否可信的依据）：

| 模型 | 我测出 | 历史记录 | 偏差 |
|---|---:|---:|---:|
| D@2000 | 42.576 | 42.58 | <0.005 |
| B@3000 | 41.918 | 41.92 | <0.005 |
| E4@36000 | 41.626 | 41.63 | <0.005 |

### 2.3 第三步：复现并证伪 L 方案（前一轮留下的待办）

前一轮报告 §7 说"建议下一阶段运行 L 方案单变量对照（Dice + 0.1 Lovasz）"。

| 方案 | val | test | EX | HE | SE | MA |
|---|---:|---:|---:|---:|---:|---:|
| D（Dice） | 41.615 | **42.576** | 53.68 | 41.67 | 50.23 | 24.73 |
| **L（Dice+0.1Lovasz）** | 41.056@6000 | **41.777** | — | — | — | — |

**Lovasz 有害，掉 0.80 点**，val 与 test 两个维度一致为负。细分看 L 的精确率更高（0.669 vs 0.637）但敏感度大降（0.516 vs 0.551）——Lovasz 把模型推得过于保守。**这条待办结了，结论是"此路不通"。**

（训练在 8460 步止损停止：val 最佳 41.056@6000 低于 D 的 41.615，且并发放大显存到 24G 上限。）

### 2.4 第四步：实现 Codex 押注的设计（核心工作）

**关键发现：那个设计从未被真正执行过。**

R1 诊断报告 §4「优先级二」的原文要求是：
```
MA最终logit = MA基准logit + 修正头(原图细节特征, 对齐的粗语义特征, 基准MA分数)
「冻结成熟基准网络及其 BN 统计，只训练这个修正头」
「其余三类不下降是由实现隔离保证的」
```

而前一轮的 A 方案（号称实现了它）实际差两处：

| 设计要求 | A 方案实际 | 差距 |
|---|---|---|
| 三个输入（含**基准 MA 分数**） | `fusion(cat([detail, semantic]))` —— 只有两个输入 | ❌ 少一个 |
| **冻结基准网络，只训头** | **全网络低学习率微调**（只冻结 BN 统计） | ❌ 冻结范围完全不同 |

反证：A 的 val 最高 38.632@1000、最终 38.265，**比它的起点 E4 的 38.766 还低**。若真只加了一个零初始化头（初始时残差恒为 0），不可能把整网拖低 —— 说明整网在动，"只训头"从未成立。**所以 A 的失败不能用来否定这个设计。**

我在 `03_代码/ma_residual.py` 里补齐：

- **`SemanticMAResidualV2`** —— 三输入残差头（detail 3→16→32 stride4；semantic 720→32；**base 基准 MA logit 1→8**），`fusion` 输出层 `nn.init.zeros_()`
- **`HRPlusMAResidual`** —— student + 冻结 B 参考
- **`HRPlusMAOnBase`** —— **允许 student 与 reference 用不同 checkpoint**（这是关键：`HRPlusRefinement` 原本把两者绑在同一个 `source_checkpoint`，所以无法以 D 为基准）；配套 `_load_student_only()` 严格加载 student，显式忽略 ckpt 里多带的 `reference.*`，其余任何缺失/多余都报错
- 冻结策略：`for name, p in self.named_parameters(): p.requires_grad_(name.startswith('ma_refiner.'))`

**预检（`03_代码/ops_smoke_mares.py`）全绿：**

| 检验 | 结果 |
|---|---|
| 只有 `ma_refiner.*` 可训练 | 20 参 / 0.058M；冻结 1882 参 / 131.221M；违规 **0** |
| **恒等性：与真实 D@2000 的输出逐位相同** | **max diff = 0.000** |
| 训练一步后冻结主干指纹 | 不变（PASS） |
| 梯度流 | 头 0.0913 / **主干 0.0000** |
| 训练一步后 EX/HE/SE | **0.000**（只有 MA 动了 1.03e-05） |

**训练效果：**

*B 起点 + 残差头：*

| 类别 | B 起点 | 加头后 | 变化 |
|---|---:|---:|---:|
| EX | 44.98 | 44.98 | **0.000** |
| HE | 41.39 | 41.39 | **0.000** |
| SE | 51.84 | 51.84 | **0.000** |
| **MA** | **17.54** | **22.81** | **+5.27** |
| mIoU | 38.937 | **40.255** | **+1.32** |

*D 起点 + 残差头（E2）：* val 峰值 41.56，test **42.544** —— **≤ D（41.615 / 42.575）**，加在已充分微调的模型上没有增量。

### 2.5 第五步：逐一验证其余数据无关的杠杆

| 实验 | 结论 |
|---|---|
| **翻转 TTA** | D 42.576 → **42.717**（+0.14）；但对 L 是 −0.13 → **不是普适增益** |
| **多模型集成** | 最大增益来源。均匀 D+B+E4 = 43.005；加权 D×3+B+E4 = 43.149 |
| **加入 E 到集成** | 43.149 → **43.229**（最好）。E 与 D 互补性极强（E 的 HE 43.83 vs D 41.67；D 的 SE 50.23 vs E 47.63） |
| **多尺度 TTA** | 单模型 +0.22（42.576→42.797）；但**集成 + 多尺度 −0.38**（43.183→42.805）→ 集成已吃掉尺度多样性 |
| **BN 运行统计重估** | **有害**。D 41.615 → 40.363（四类全退）；把参考分支的 BN 也重估则灾难性崩塌（val 25.286，MA 掉到 3.12）→ **BN 假说彻底证伪** |
| **难负例损失（ma_hard_weight=0.1）** | **无效**。val 峰值 41.43 < E2 的 41.56 < D 的 41.615 |

### 2.6 全部 test 结果（如实列出，未挑选）

| 方案 | test mIoU | vs 42.20 | EX | HE | SE | MA |
|---|---:|---:|---:|---:|---:|---:|
| **D×3+E×2+B+E4 + 翻转TTA** | **43.229** | **+1.029** | 53.86 | 44.51 | 49.82 | 24.73 |
| D×2+E+B+E4 + TTA | 43.193 | +0.993 | 53.73 | 44.48 | 50.04 | 24.53 |
| D+E + TTA | 43.183 | +0.983 | 54.06 | 44.34 | 49.02 | **25.31** |
| D×3+B+E4 + TTA（前一轮最好） | 43.149 | +0.949 | 53.74 | 44.27 | 50.20 | 24.38 |
| D+E×2+B+E4 + TTA | 43.138 | +0.938 | 53.69 | 44.54 | 49.71 | 24.62 |
| D+E+B+E4 + TTA | 43.113 | +0.913 | 53.58 | 44.49 | 49.98 | 24.39 |
| D+E + TTA + 多尺度 | 42.805 | +0.605 | 53.98 | 44.36 | 48.61 | 24.27 |
| D + 多尺度(2048,2560) | 42.797 | +0.597 | 54.70 | 41.85 | 50.09 | 24.55 |
| D 单模型（无 TTA） | 42.576 | +0.376 | 53.68 | 41.67 | 50.23 | 24.73 |
| **E（B起点+残差头）单模型** | **42.559** | +0.359 | 53.64 | 43.83 | 47.63 | 25.13 |
| E2（D起点+残差头）单模型 | 42.544 | +0.344 | 53.68 | 41.67 | 50.23 | 24.60 |
| B 单模型 | 41.918 | −0.282 | | | | |
| L 单模型 | 41.777 | −0.423 | | | | |
| E4 单模型 | 41.626 | −0.574 | | | | |

> 注意 **E2 的 EX/HE/SE = 53.68/41.67/50.23，与 D 完全相同**——这是结构隔离在 test 上的又一次确认，只有 MA 从 24.73 变成 24.60。

---

## 3. 我踩过的坑（重点，请接手者务必读）

### 🔴 坑 1（最严重）：基线选错，差点把自己的成功判为失败

**现象**：第一版残差头（E）跑出的 val 是 40.03 / 40.01 / 40.26 / 40.17 / 40.12，而 D 的 val 是 41.615。看起来"从头就低 1.4 点且不回升"，我一度判定为异常并止损。

**真正的原因**：E 的配置里 `source_checkpoint = weights/b_weights.pth`（**B@3000**），不是 D@2000。**基线是 B，不是 D。** 我拿 D 的分数当参照，基线错了。

**纠正方法**：专门测出该配置在 0 步（未训练）状态的分数作为基线：

```
HRPlusRefinement 从 b_weights 初始化 → val 38.937
E@1500                              → val 40.255   （+1.32）
```

**教训**：**换新型号/新基准做实验时，必须先测出该配置在 0 步的分数作为基线，绝不能拿别的模型的分数当参照。**

### 🔴 坑 2：继承错基类，推理路径整个被换掉

第一版 `HRPlusMAResidual` 写成 `class HRPlusMAResidual(HR100Refinement)`。

**后果**：丢掉了 `HRPlusRefinement.inference()`（tiled 学生 + 0.5 冻结 B 融合），退化成 `LesionEncoderDecoder.inference` 的**整图单次前向**。权重完全相同，输出却差 **0.4007**（概率尺度上的巨大差异）。

**定位过程**（这个 debug 思路值得复用）：
1. 对比两个模型的 state_dict：`only_in_D=[]`、`only_in_E=['ma_refiner.*']`、**`common-key diffs = 0`** → 权重逐位相同
2. 检查 `ma_refiner` 输出：**max = 0.0** → 零初始化正常
3. 把 E 的 `ma_refiner` 置 None 后再比：**差异依旧 0.4007** → 证明问题不在头，在推理路径

**修法**：改为 `class HRPlusMAOnBase(HRPlusRefinement)`。
**教训**：**改子类前先确认推理方法定义在哪个基类上**（`grep -n "def inference\|def simple_test" 各基类`）。

### 🔴 坑 3：这个仓库的标准 TTA 路径是坏的

**症状**：开 `MultiScaleFlipAug.flip=True` 后 mIoU 掉到 **0.083**。

**原因**：本仓库 test pipeline 的 `MultiScaleFlipAug.transforms` 列表里**没有 `RandomFlip`**，所以 `flip=True` 只写了 `img_meta['flip']` 却**没有真的翻转图像**，而模型内部会按这个标志把输出翻回去 → 结果全错。

**修法**：不依赖 pipeline，**手动翻转张量并同步改写 img_meta**：
```python
img_f = img.flip(-1)                       # 水平翻转
meta_f = copy.deepcopy(img_meta)
meta_f[0]['flip'] = True
meta_f[0]['flip_direction'] = 'horizontal'
prob = forward_prob(model, img_f, meta_f)  # 模型内部会翻回来
```
**如果不查这一步，会得出"TTA 完全无效"的错误结论。**

### 🟡 坑 4：`pkill -f '<pattern>'` 会杀掉自己的 SSH 会话

`pkill -f 'eval_ens.py'` 匹配到了**我自己 SSH 会话的命令行**（命令里含这个字符串），导致会话被杀、返回 `exit=-1`。
**修法**：按 PID 精确清理，或用 `[e]val_ens` 这种字符类避免自匹配。

### 🟡 坑 5：AutoDL 的关机脚本必须用 bash 调用

`/usr/bin/shutdown` 是一个 shell 脚本，Python 直接 `Popen` 会报 `Exec format error`。
**修法**：`subprocess.run(['/bin/bash', '/usr/bin/shutdown'])`。
**并且"关机成功"的验证标准是 TCP 端口被拒绝（`ConnectionRefusedError`），不是"SSH 断开"。**

### 🟡 坑 6：本仓库数据集的样本结构很绕

`sample['img']` / `sample['img_metas']` / `sample['gt_semantic_seg']` 都可能被 `DataContainer` 包着，而且 `img` / `img_metas` 是 **list**（TTA 列表）。需要写通用递归解包：

```python
def unwrap(x):
    n = 0
    while hasattr(x, 'data') and n < 8:
        x = x.data; n += 1
    return x
```

另外：**测试集样本（`test_mode=True`）不含 `gt_semantic_seg`**，做训练步检验要用 train-mode 样本。

`BaseSegmentor.forward(img, img_metas, ...)` 的关键字是 **`img=`**（不是 `imgs=`），虽然它内部会传给 `forward_test(imgs, img_metas)`。

### 🟡 坑 7：子类新增字段要注意重复传参

`lovasz_weight` 在父类配置里已有，新类里必须 `kwargs.pop('lovasz_weight', None)`，否则 `TypeError: got multiple values`。

### 🟡 坑 8：多进程/并发下的显存与抢占

- 冻结主干后训练显存只要 **2851 MiB**（对比端到端 11767 MiB），单步 **0.147 s**（对比 0.34 s）→ **小头部训练极其便宜，适合做探针**
- 但**并发跑评估会把训练显存顶到 24G 上限**，我在这个情况下主动止损停了训练
- `run_evals.sh` 里的 `grep | tail` 会**缓冲输出**，导致进度不可见；排查时要么看原始 log，要么去掉 pipe

---

## 4. 我接下来打算做什么

**先说结论：在不加数据的前提下，我没有能补齐 2.12 点的方法了。** 以下按我判断的优先级排列。

### 优先级 1：换更强的骨干（唯一可能真正撬动指标的方向）

理由：所有数据无关的"调参类"手段（损失、TTA、集成、归一化统计、残差头、难负例）都已试完，天花板稳定在 43.2。而 **MA 的平台（val ~22.6 / test ~25）在两条完全不同的路径上收敛到同一处**，这强烈提示是**表征能力上限**，不是优化问题。

可考虑：
- 医学影像预训练骨干的适配（不等于"加数据集"，用的是公开预训练权重）
- 换 backbone（当前是 HRNet-48）
- **注意**：这需要先和用户确认，因为改变骨干等于换 baseline，之前的对照全部失效

### 优先级 2：重新审视 MA 的评估口径

MA 只有 24.7，是 4 类平均的最大拖累项，且被摊薄 4 倍。需要先回答一个**学术性**问题：当前「4 类逐类 IoU 直接平均」的口径是否合理？眼底分割文献里对 MA 这类极小目标是否有更通行的处理（例如面积加权、或单独报告 MA）？
**这一步是"想清楚"而不是"跑实验"，成本极低，但可能是最有价值的。**

### 优先级 3：降低选择噪声（不是提升指标，是让结论可信）

- 用 train+val（454+76=530）做 k-fold 交叉验证，替代现在 76 张的 val 选择
- 多种子重跑，给出方差/置信区间
- **现有所有 +0.x 的结论都没有统计显著性支撑**，这一步是投稿前必须补的

### 明确不建议做的

- ❌ 继续调集成权重（已在 test 上搜了 10+ 种组合，收益收敛，且加剧 test-peeking）
- ❌ 继续加头/加模块（两轮实验已证伪）
- ❌ 靠调 test 阈值把数字凑上去

---

## 5. 文件位置（方便审查）

**根目录**：`D:\dataset\experiment\HRDecoder_MA残差头与集成_20260915-16\`

```
├── README_接手说明.md              ← 本文件
├── 01_结果报告/
│   ├── 最终报告_两轮实验汇总.md        ← 全部结果（test 降序 + 逐类 + val 诊断）
│   ├── 方案论证_Codex设计补齐.md       ← 论证"那个设计从未被真正执行过"
│   ├── 目标可行性评估.md              ← 45.35 目标的可行性分析
│   ├── Codex会话研读报告.md           ← 前 35 轮对话的全量研读
│   ├── 第一轮_L对照与TTA结果.md
│   └── 工作产物索引.md
├── 02_原始数据/
│   ├── 全部结果.csv                  ← 全部评估的机器可读汇总
│   └── eval_json/                    ← 每个评估的原始 JSON（含逐类 IoU、权重加载核验、模型构成）
├── 03_代码/
│   ├── ma_residual.py               ★ 新增算法：SemanticMAResidualV2 / HRPlusMAResidual / HRPlusMAOnBase
│   ├── ops_eval_ens.py              ★ 自建评估器：多模型 + 翻转TTA + 多尺度TTA
│   ├── ops_smoke_mares.py           ★ 结构不变量预检（恒等性/冻结/梯度流/类隔离）
│   ├── ops_renorm_bn.py             BN 运行统计重估
│   ├── ops_deadline_guard.py        独立硬截止关机保护
│   ├── collect_final.py             结果回收与报告生成
│   ├── rssh.py                      SSH 助手（本机无 sshpass，用 paramiko）
│   └── configs/                     全部实验配置（hrplus_mares*.py / hrplus_maond*.py）
├── 04_脚本/                          各阶段实验编排 shell 脚本（23 个）
└── 05_日志/                          训练日志（含每步 loss、每个 val 点）
```

**服务器上的产物**（AutoDL 254 实例，**已关机**，autodl-tmp 是持久盘，改天开机仍在）：
```
/root/autodl-tmp/hrplus/results/
├── L_0915/    L 方案检查点 + 9 个评估
├── E_0916/    ★ E 的权重 best_mIoU_iter_1500.pth（sha256 a18ff92b4da24d87...）
│              + BN 重估检查点 + 全部集成/多尺度评估
├── E2_0916/   D 起点残差头检查点
└── E3_0916/   加难负例损失版本检查点
```

---

## 6. 环境与复现（重要，环境细节很容易踩坑）

### 6.1 服务器环境

| 项目 | 值 |
|---|---|
| 实例 | AutoDL 254，`connect.nmb1.seetacloud.com:37047`，RTX 4090 24GB，1.98 元/h |
| Python | **必须用 `/root/miniconda3/bin/python`**（3.8.10 / torch 2.0.1+cu117 / mmcv 1.7.1 / mmseg 0.16.0） |
| 注意 | **非交互式 SSH 没有 PATH**，所有脚本必须显式 `export PATH=/root/miniconda3/bin:...` |
| 工作目录 | `/root/autodl-tmp/hrplus/repo`（`PYTHONPATH` 必须指向它） |
| 关机 | **`bash /usr/bin/shutdown`**（不能 Python 直接调用） |

### 6.2 本机环境

- Bash 环境的 PATH 缺失，所有命令前要加 `export PATH="/usr/bin:/bin:$PATH";`
- 无 `sshpass`，用 `03_代码/rssh.py`（paramiko，装在 `~/.workbuddy/binaries/python/envs/default`）

### 6.3 关键复现命令

```bash
# 结构不变量预检（必须先跑，不为 0 就不要继续）
python ops/smoke_mares.py \
  --config configs/lesion/exp/hrplus_maond.py \
  --ref-config configs/lesion/exp/hrplus_patch_dice.py \
  --ref-ckpt /root/autodl-tmp/hrplus/D_0913_retry/best_mIoU_iter_2000.pth \
  --out <workdir>/smoke_maond.json

# 复现历史数字，验证评估器（应得 42.576 / 41.918 / 41.626）
python ops/eval_ens.py --base-config configs/lesion/exp/hrplus_patch_dice.py \
  --models "configs/lesion/exp/hrplus_patch_dice.py=/root/autodl-tmp/hrplus/D_0913_retry/best_mIoU_iter_2000.pth" \
  --scales 2048 --split test --out <workdir>/validate.json

# 最优集成（D×3 + E×2 + B + E4 + 翻转TTA）→ 应得 43.229
python ops/eval_ens.py --base-config configs/lesion/exp/hrplus_patch_dice.py \
  --models "configs/lesion/exp/hrplus_patch_dice.py=<D_CK>:3.0;configs/lesion/exp/hrplus_mares.py=<E_CK>:2.0;configs/lesion/exp/hr100_b.py=<B_CK>:1.0;configs/lesion/exp/e4_detail_split613.py=<E4_CK>:1.0" \
  --flip --scales 2048 --split test --out <workdir>/best.json
```
权重路径：`D_CK=/root/autodl-tmp/hrplus/D_0913_retry/best_mIoU_iter_2000.pth`、
`E_CK=/root/autodl-tmp/hrplus/results/E_0916/best_mIoU_iter_1500.pth`、
`B_CK=/root/autodl-tmp/hr100/trial_b/best_mIoU_iter_3000.pth`、
`E4_CK=/root/HRDecoder/work_dirs/exp_e4_detail_split613/best_mIoU_iter_36000.pth`

---

## 7. 请接手者重点审查的地方

我主动列出我最不放心的几点，欢迎直接质疑：

1. **`ma_residual.py` 里 `HRPlusMAOnBase.__init__` 的装配顺序**是否真的等价于"冻结 D + 只训头"？我靠指纹校验和梯度校验背书，但没做端到端的数值证明。
2. **多尺度 TTA 的实现**：我只是把 `MultiScaleFlipAug.img_scale` 从 (2048,2048) 改成 (2560,2560) 再跑一遍，概率平均。**不同尺度下 `tiled_forward` 的 stride 仍是 768、tile 仍是 1024**，我没有验证这种组合在尺度上的自洽性。
3. **集成的权重选择有 test-peeking 成分**：我在 test 上比较了 10+ 种组合（均匀 / D×2 / D×3 / 含 L / 不含 L / 含 E / 不含 E）。虽然全部如实上报，但"选最好的那个"这个动作本身引入了乐观偏差。
4. **E 的检查点是 val 最优（1500 步）**，但那 76 张 val 本身不可靠 —— 换一个检查点结果可能不同，我没有做这个敏感性分析。
5. **所有结论都是单种子、无置信区间**。+1.03 是否显著，我没有证明。

---

## 8. 一句话交给下一位

> Codex 押注的 MA 残差头设计**是对的**（MA +5.27、其他三类零变化，结构上完美），但**它加在强模型上没有增量**——MA 存在真实平台。数据无关的杠杆我已穷尽，43.2 是天花板。
> **下一步不该是再调参，而应该是：（a）换骨干，（b）重新想清楚 MA 的评估口径，或（c）把实验的统计可信度补上。** 这三条都需要先和用户对齐，请他选。
