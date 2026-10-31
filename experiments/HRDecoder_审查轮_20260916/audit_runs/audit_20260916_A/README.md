# README.md —— 审计轮 Phase A 报告 + 首条回复

> **run_id**: `audit_20260916_A`
> **生成时间**: 2026-09-16（CST）
> **执行者**: 实现与验证 AI
> **对应指令**: `prompt.md`（`HRDecoder_下一轮执行Prompt_20260916.md`）+ `审查结论.md`

---

## 0. 当前状态

| 阶段 | 状态 | 说明 |
|---|---|---|
| **A 盘点与协议冻结** | ✅ **PASS**（可本地完成的部分全部完成） | 见 §2、`manifest.json`、`protocol.md`、`phaseA4_data_boundary.json` |
| **B 安全保护与 CPU 单元测试** | 🟡 **PARTIAL** | 指标 oracle 12/12 PASS（本地实跑）；依赖 mmcv 的部分 BLOCKED |
| **C 小样本真实模型验证** | ⛔ **BLOCKED** | 本地无 mmcv/mmseg，无法构建真实模型 |
| **D 固定 train/val 诊断** | ⛔ **BLOCKED** | 同上；且需服务器权重 |
| **E 两臂小规模对照** | ⛔ **NOT_RUN** | 前置关卡未通过，按 §2「不允许跳关」不得启动 |
| **F 独立复核与有限确认** | ⛔ **NOT_RUN** | test 未解封 |

**未启动任何付费实例。未修改任何历史 checkpoint / 日志 / 结果。未安装任何依赖。**

---

## 1. 收到指令后的首条回复（prompt §14 要求，逐项回答）

### 1.1 目前可访问的仓库、checkpoint、split 列表和日志有哪些，缺哪些

**可访问（本地，已验证 SHA-256，详见 `manifest.json`）**

- **代码 24/26 项**：`hrplus_core.py`、`hrplus_patch.py`、`hr100_refinement.py`、`ma_residual.py`、`hr100_ensemble.py`、`segmentor_base/lesion_encdec/HRDecoder/encoder_decoder`、`dataset_lesion`、`metric_lesion`、`loss_binary`、`pipeline_transforms/formating/test_time_aug`、`train_entry`、`test_entry`、`smoke_hrplus`、以及 `cfg_hrplus_dice/lovasz/hr100_a/hr100_b/e4/baseline`
- **checkpoint 4/7 项**（本地实体文件）：
  - `b_weights` 263,479,238 B（B@3000 weights-only）
  - `d_student_only` 263,419,124 B（D@2000 的 student 分支）
  - `e4_weights` 263,477,167 B（E4@36000）
  - `a_weights` 263,709,987 B（A 方案，**含 `ma_refiner.*` 17 个键**）
- **split 2/2 项**：`split_assignment.json`（757 条，含 `split`/`original_subset`/`tier`/`se_ratio`/`class_pixels`）、`data_manifest.json`（1514 条，含 path/bytes/sha256）
- **日志**：仅第一轮 L 的 `l_train.log` 副本、第二轮的 `e_stage.log`；**D/E 的原始训练日志缺失**

**缺失** → 全部列入 `missing_dependencies.md`。要点：

- **D@2000 完整 checkpoint（789 MB）、E/E2/E3/L 的 checkpoint 只在服务器上**
- **本地仓库副本不是 git 仓库**，无分支/提交/未提交改动可记录
- **本地无 mmcv/mmseg**，无法构建真实模型
- 数据集仅以 tar 形式存在（未解压）

### 1.2 是否具备当前有效的服务器授权、预算与截止时间

**❌ 不具备。** 上一轮凭据对应的实例已关机（TCP 探测返回 `ConnectionRefusedError`），
且 prompt §0.1 要求的是**当前**授权。为遵守「未获当前预算与开机授权，不启动付费实例」，
本轮**未开机、未训练、未全量重评测**。

需要的授权要素（§10.4 / §4.3）：实例标识、允许任务范围、总预算或最长用时、
**绝对截止时间（含时区）**、关机方式、以及云平台侧"停止计费"的独立确认手段。
（凭据一律不在任何文件中记录。）

### 1.3 最先核查的五项

| # | 核查项 | 本轮进展 |
|---|---|---|
| 1 | **加载契约** | 🟡 静态 + 部分实证：4 个本地 checkpoint 的键集合/前缀/是否有头已记录；`b_weights` vs `d_student_only` 已做逐键 shape/dtype 比对（**零差异**）。缺 `load_checkpoint` 真实返回类型的运行时核对（见 V6） |
| 2 | **post-step 全状态冻结** | ⛔ 未做。需真实模型与真实 optimizer → Phase C2 |
| 3 | **评测几何/聚合** | ✅ **已完成协议冻结并落地独立 oracle**：12/12 PASS，含 S2（累计 vs 逐图平均）、S8（严格 `>`）、S11（ignore 行为） |
| 4 | **train/inference 路径** | ⛔ 未做。需 hooks 捕获真实中间张量 → Phase C3/D。**已识别出两条不同推理函数**（`HRPlusRefinement.inference` vs `LesionEncoderDecoder.inference`），列为 V4 |
| 5 | **关机保护** | 🟡 静态审查结论：旧 guard 存在 `审查结论.md` 指出的 120 秒睡眠跨截止、关机失败后退出等问题，**不能原样信任**。本轮未开机，故未部署新 guard |

### 1.4 能在不开付费实例的前提下立即完成什么

**已经完成的：**

1. `manifest.json` —— 代码/checkpoint/split 的 SHA-256、字节数、环境版本、确定性开关
2. `protocol.md` —— 从**真实源码**反推的完整评价协议（公式、阈值、聚合、量纲、空并集）
3. `phaseA4_data_boundary.json` —— 数据边界审计，**PASS**
4. `tests/loss_ignore_metrics.json` —— **独立指标 oracle，12/12 PASS**（纯 NumPy，不 import 被测模块）
5. `phaseA_identity_split.json` —— checkpoint 身份 + split 结构
6. `missing_dependencies.md` —— 缺失项与授权缺口
7. **发现并实证了一个真实的实现问题**（见 §3.1）

**还可在不开机情况下继续做的（建议下一步）：**

- 写完全部 16 项**故障注入**测试的 CPU 版本（不依赖 mmcv 的部分）
- 用 `ddr_split613.tar` 的局部解压实测标签值域（消解 V1）
- 把几何 helper（`tiled_forward` / `blend_probs` / `resize_logit` / `foreground_crop_bbox` / `crop_arrays` / `pad_to_size`）的边界测试写成**可直接在服务器 CI 式运行**的脚本
- 为 guard 写 fake-clock + mock-shutdown 测试（§4.3-8）

### 1.5 第一阶段完成后会提交哪些证据文件

已按 prompt §12 的目录契约建立 `audit_runs/audit_20260916_A/`。**已产出**：

```
audit_runs/audit_20260916_A/
  README.md                      ← 本文件
  manifest.json                  ✅
  missing_dependencies.md        ✅
  protocol.md                    ✅
  tests/
    loss_ignore_metrics.json     ✅ 12/12 PASS
  phaseA_identity_split.json     ✅
  phaseA4_data_boundary.json     ✅
```

**待产出**（依赖开机或后续本地工作）：
`commands.log`、`code.patch`、`tests/{acceptance_matrix.csv, fault_injection.json,
unit_results.json, checkpoint_loading.json, frozen_state_steps.json, geometry_tta_ensemble.json,
save_resume.json, guard_mock_results.json}`、
`diagnostics/{train_inference_contract.md, branch_visit_counts.json, val_per_image_counts.csv,
paired_bootstrap.json, failure_case_manifest.csv, resource_profile.json}`、
`experiments/{experiment_decision.md, preregistered_comparison.json, resolved_configs/,
raw_logs/, checkpoint_manifest.json, all_val_results.csv}`、
`operations/{budget_and_deadline.json, guard.log, artifact_checksums.sha256, shutdown_status.json}`、
`final_report.md`

---

## 2. Phase A 已完成工作的具体结论

### 2.1 身份清单（`manifest.json`）

- 环境：Python 3.13.11 / numpy 2.4.1 / torch 2.6.0+cu124 / **mmcv=None / mmseg=None**
- 确定性开关：`cudnn.deterministic=False`、`cudnn.benchmark=False`、
  `matmul.allow_tf32=False`、**`cudnn.allow_tf32=True`** ← 注意最后一项
- 本地仓库：`NOT_A_GIT_REPO`
- 26 项代码、7 项 checkpoint、2 项 split 的 SHA-256 与字节数

### 2.2 评价协议冻结（`protocol.md`）

**从真实代码反推，不是按名字猜：**

```
对 c ∈ {EX, HE, SE, MA}（通道 0..3 ↔ 标签 1..4）:
  TP_c = #{gt == c  ∧  prob[c] > 0.5}
  FP_c = #{gt != c  ∧  prob[c] > 0.5}
  FN_c = #{gt == c  ∧  prob[c] <= 0.5}
  IoU_c = TP_c / (TP_c + FP_c + FN_c)     # 分母 0 -> NaN -> 置 0
  mIoU  = (IoU_EX + IoU_HE + IoU_SE + IoU_MA) / 4
```

**冻结项**：阈值 **0.5**、比较符 **严格 `>`**、聚合为**全数据集累计**、
背景**不计入平均**（`iou[-4:]`）、量纲内部 0–1 报告 ×100、空并集 NaN→0。

> 特别核对了一个容易搞错的地方：`np.linspace(0,1,11)` + `argmax(threshs==0.5)` 在
> `compute_aupr=True` 时命中的**仍是 0.5**，所以两条路径的主指标阈值一致。

### 2.3 数据边界（`phaseA4_data_boundary.json`）—— **PASS**

| 检查 | 结果 |
|---|---|
| split 计数 | train **454** / val **76** / test **227** —— 与约定**完全一致** |
| 合计 | 757 = 454+76+227 ✓ |
| 两两不相交 | train∩val = 0，train∩test = 0，val∩test = 0 **PASS** |
| 重复 basename | 0 |
| 图像/标签配对 | 757 / 757，**完全配对 PASS** |
| 跨文件重复图像哈希 | 0 |
| `original_subset`（官方划分）分布 | train 383 / val 149 / test 225 |
| `tier` 分布 | A 518 / B 119 / C 120 |
| 逐类像素（**仅 train/val**，test 按 §3.4 有意跳过） | train: EX 5,786,406 / HE 8,873,022 / SE 1,327,479 / MA 527,407<br>val: EX 292,046 / HE 1,207,733 / SE 201,420 / MA 69,284 |

> **披露**：自定义划分是对官方 train（383 张）重新切分的（`original_subset=train` 的图
> 出现在 454/76/227 三个 split 中），这与既有的"用自定义划分做内部对比"的做法一致，
> 但**必须继续在论文中披露其性质**。

**⚠️ 我撤回了一个自己先前的错误结论**：我最初用文件名前 3 位当患者号，得到
"train∩val 患者重叠 = 2"。经核查，644 个 `NNN-xxxx-NNN` 形式文件的**前缀全部是 `007`**
（批次编号而非患者号），另 113 个是纯数字时间戳无法归组。**该规则失效，任何基于文件名的
患者重叠结论都是假象。** 患者级审计现记为 **BLOCKED**，需要数据集自带的患者映射表。

---

## 3. 本轮已证实的问题（不是"看起来没问题"）

### 3.1 ⚠️ `ignore_index` 被传入但从未使用（源码 + 实证双重证据）

**源码证据**
- `lesion_metric.py:173` 定义了形参 `ignore_index=None`
- 函数体内（178–205 行）**一次都没有引用它**；下游 `sigmoid_metrics` /
  `sigmoid_confused_matrix` 的签名里也没有这个参数
- 调用侧 `lesion_dataset.py:71-72` 确实传了 `ignore_index=self.ignore_index`

**实证证据（S11，本地实跑）**

```
S11  PASS  oracle不变=True  legacy随ignore变化=True
```

即：在 `gt==255` 的位置把预测从 0.1 改成 0.9，
**独立 oracle 的 TP/FP/FN 完全不变**（因为它显式排除 ignore），
而**忠实复刻被测实现的 replica 的 IoU 发生了变化** —— 证明 ignore 未被排除。

**后果**：`gt==255` 的区域被隐式当作**负样本**；模型在该区域的阳性预测会计入 FP。

**影响范围**：取决于 val/test 标签是否含 255 → **未验证**（V1）。
按 prompt §3.3，**现在只报告不一致，保留 legacy 结果，不偷偷换指标，不重算历史成绩**。

### 3.2 其他需验证项（不得写成 PASS）

| ID | 内容 | 阶段 |
|---|---|---|
| V1 | val/test 标签是否含 255（决定 3.1 是否影响历史分数） | C4 |
| V2 | `auc(x,y)` 在召回率单调递减时是否按预期工作（**非主指标**） | C4 |
| V3 | `num_classes` 真实调用值是 5 还是 4 | C1 |
| V4 | `HRPlusRefinement.inference` 与 `LesionEncoderDecoder.inference` 的真实调用点 | C3 |
| V5 | `index=argmax(threshs==0.5)` 的运行时断言 | C1 |
| V6 | `load_checkpoint` 在**服务器安装版本**中的真实返回类型（旧评估器解包失败即置空，不是可信审计） | C1 |
| V7 | 本地代码副本与服务器当前版本是否逐文件一致 | A（需开机） |

---

## 4. 我**没有**做的事（遵守禁令）

- ❌ 未开机、未启动任何付费实例
- ❌ 未做任何 GPU 训练或全量重评测
- ❌ 未使用 test 做调参 / 阈值选择 / 错误案例挖掘 / 提前停止 / 方案方向选择
- ❌ 未解压数据做 test 标签统计（A4 中 test 逐类像素**有意跳过**）
- ❌ 未合并 train+val，未替换划分，未重划
- ❌ 未创建虚拟环境、未安装依赖、未升级训练栈
- ❌ 未修改任何历史 checkpoint / 日志 / 结果
- ❌ 未记录任何凭据
- ❌ 未重写父类实现来"复现"原方法

**也未被使用来支撑任何设计决策**：本轮的 A4 只做完整性审计。

---

## 5. 下一步

### 5.1 你可以二选一

**选项 A（推荐，零成本）**：我继续在本地推进不依赖 mmcv 的工作 —— 完整故障注入测试
（16 项中不依赖真实模型的部分）、几何/平铺/TTA/集成的纯 torch 单元测试、
guard 的 fake-clock 测试、以及局部解压数据消解 V1。**做完再一次性开机，付费时间最短。**

**选项 B**：现在开机，我立即执行 Phase C1（checkpoint 与模型装配的运行时契约核对），
取回 `load_checkpoint` 真实返回类型、解析后 config 哈希、以及 D/E 完整 checkpoint。

### 5.2 若要开机，请提供（§10.4 / §4.3）

1. 实例标识（哪个实例）
2. 允许任务范围（是否允许跑 Phase C 的短跑；是否允许后续两臂 pilot）
3. 总预算或最长用时
4. **绝对截止时间 + 时区**
5. 关机方式，以及**云平台侧"停止计费"的独立确认手段**（TCP 拒绝只能作辅助信号）
6. **test 是否解封**（默认封存）

### 5.3 关于 GO/NO-GO

按 §10.1，现在**不能**给出 GO：首选试验（"部署一致的 MA 监督"）成立的前提是
§9.1 证明确有训练/部署监督不一致，而这一项（V4）**尚未在真实路径上验证**。
本轮给出的状态是 **Phase A PASS / Phase C 起因缺依赖 BLOCKED**，不是 NO-GO。

---

## 6. 一条命令复算本轮关键结论

```bash
# 独立指标 oracle（纯 NumPy，不 import 被测模块）
python tools/oracle_metrics.py --out-dir audit_runs/audit_20260916_A/tests
# 期望：12/12 PASS，退出码 0；S2 必须给出 0.1（不是逐图平均的 0.5）
#       S11 必须给出 oracle不变=True 且 legacy随ignore变化=True
```
