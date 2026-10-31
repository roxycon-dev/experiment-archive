# TRAINING_IMPLEMENTATION_STATUS — 训练入口与公共训练核心实施状态

日期：2026-09-20 ｜ 本阶段：**未连服务器、未训练真实数据、未联网、未下载权重、未修改 `D:\HRDecoder`**。
所有"通过"均为 CPU 上的合成（fake model + 合成数据）实测结果。

---

## 0. 一句话结论

- 公共训练核心（数据 → masked loss → 整图指标 → 训练状态机 → checkpoint）**已实现并通过合成测试**；
- U 臂（ConvNeXt-S + U-Net decoder）在 CPU 上**真实构建 + 真实前向/反向通过**；
- H 臂（旧仓库 HRNet-W48 + FCNHead adapter）在本机 **DEPENDENCY_BLOCKED**（缺 `mmcv`/`mmengine`），
  **未用自造 HRNet 冒充**；
- **正式训练仍不可开**：四臂配置都还有 `REQUIRED` 占位符（见 §4）；
- 短时 GPU 校准：**H0/H1 readiness 已清空可开，U0/U1 不可开**（见 §5）。

## 1. 交付文件

| 文件 | 作用 |
|---|---|
| `train.py` | 唯一训练入口（`--stage external_pretrain\|ddr_finetune`，`--mode formal\|calibration`） |
| `trainer/engine.py` | 训练状态机：梯度累积、scheduler、AMP、BN 冻结、eval/ckpt 调度、断点恢复 |
| `trainer/config.py` | 配置加载 + 两层闸门 + split/test 政策 + 生效配置 |
| `trainer/eval.py` | **整图 / 滑窗**验证路径（唯一合法的 val 指标来源）+ fail-closed 断言 |
| `trainer/loss.py` | masked BCE + masked soft Dice（含全无效类零梯度语义） |
| `trainer/metrics.py` | 整图累计 TP/FP/FN + AUPR + JSON/CSV 落盘 |
| `trainer/optim.py` | 参数组、按 update 的 warmup+cosine、裁剪、BN 冻结/审计、参数组审计 |
| `trainer/checkpoint.py` | checkpoint 格式、原子写、`weights_only` 安全加载、断点一致性校验 |
| `trainer/gate.py` | 占位符两层分类（readiness / calibration）的单一权威定义 |
| `trainer/paths.py`、`trainer/seed.py` | 落盘路径校验、种子与 RNG 捕获/恢复 |
| `trainer/data/{common,augment,ddr_dataset,external_dataset}.py` | DDR 固定 split、外部 JSONL manifest、共享几何增强 |
| `trainer/models/convnext_unet.py`（U）、`hrdecoder_adapter.py`（H） | 两个结构族；输出严格 `N×4×H×W` |
| `trainer/models/weight_loading.py` | 预训练权重加载：禁止静默部分加载，missing/unexpected 全量报告 |
| `tools/preflight_training.py` | CPU 训练链路自检（合成数据；E0–E12） |
| `tools/preflight_contract.py` | 新增 `--layer readiness`（原有 formal 闸门**未放宽**） |
| `tests/run_training_smoke.py` | 训练链路合成冒烟（T0–T16） |
| `docs/TRAINING_OBJECTIVE.md` | 损失与指标的精确定义 |

## 2. 本轮审查发现并修复的缺陷（都会影响科学有效性或直接跑不起来）

| # | 位置 | 缺陷 | 影响 | 修复 |
|---|---|---|---|---|
| 1 | `models/convnext_unet.py:UNetDecoder` | `e4,e3,e2,e1 = dims` 与注释/用法错位，`c4` 被建成 `ConvBlock(96,256)` 却要吃 768 通道 | **U 臂第一次前向就 RuntimeError**，"已实现"不成立 | 改为按 stride 显式命名 `e_s4/e_s8/e_s16/e_s32` |
| 2 | `models/convnext_unet.py:ConvNeXtBlock` | `dwconv` 被包了一层 `DWConv` 容器 ⇒ 键变 `dwconv.dwconv.weight`，与官方 `dwconv.weight` 对不上 | 官方 ImageNet 权重加载时 dwconv 全部 missing，**等于没加载主干** | `self.dwconv` 直接为 `nn.Conv2d`；另设显式 key alias 兜底并记录 |
| 3 | `metrics.py:_aupr` | 在 **recall 递减**方向做阶梯积分 | 完美分类器 AUPR≈0.2（真值 1.0），**指标不可比** | 改为 recall 递增方向；加"完美分类器 AUPR==1.0"断言 |
| 4 | `checkpoint.py:save_checkpoint` | 临时文件用 `mkstemp(prefix=...)` 无扩展名 | `torch.save` 报 `invalid file name`，**checkpoint 保存 100% 失败** | 加 `suffix=".pth"` |
| 5 | `checkpoint.py:build_checkpoint_payload` | 写入 `torch.__version__`（torch≥2.5 是 `TorchVersion` 对象） | `weights_only=True` 拒绝加载 ⇒ **断点恢复完全不可用** | 归一化为 `str`；`_plain()` 递归归一 + 保存后立即回读自检 |
| 6 | `gate.py:placeholder_layer` | 分类表用无前缀点路径，而 `scan_placeholders` 产出的是 `$.` 前缀路径 | **永远匹配不到**，readiness/calibration 两层闸门形同失效 | 显式剥离 `$` 根前缀 |
| 7 | `optim.py:apply_lr` | 用 `param_groups.index(dict)` 反查组号 | 两个内容相同的参数组会写错 LR | 改 `enumerate`；并加 `base_lrs_of` 锁定基准 |
| 8 | `optim.py:audit_norm_policy` | 只调 `model.train()` 不施加冻结，却检查"BN 是否 eval" | 检查项与实现策略自相矛盾，**永远 FAIL 的假检查** | 审计时一并施加冻结策略（`apply_policy`） |
| 9 | `optim.py:classify_parameter` | 裸 `startswith(prefix)` | `encoder` 会吞掉 `encoder2` 等同前缀兄弟模块 | 改为点分段匹配 |
| 10 | `data/{ddr,external}_dataset.py` | 增强 RNG 只用 `(initial_seed + idx)` | 同一 worker 每 epoch 增强完全相同 ⇒ **等价于关掉增强随机性** | 引入 `sample_rng(worker_seed, idx, epoch)` + `set_epoch()` |
| 11 | `data/external_dataset.py` | manifest 里出现 forbidden split 记录就整体报错 | 只要 manifest 含官方 test 记录就**无法开工**（而 manifest 是统一台账，含 test 属正常） | 改为"排除并计数"（`forbidden_excluded`），并加 `strict_forbidden` 开关；`allowed∩forbidden≠∅` 仍无条件报错 |
| 12 | `data/external_dataset.py` | 未校验 `sample_id` | 缺字段时后段 `KeyError` | 加载期显式契约校验 |
| 13 | `paths.py:validate_output_path` | `if not os.path.isabs(path) or True:` 死条件；默认 base 用 `os.getcwd()` | 落盘位置依赖启动 cwd，不可复现 | 清理条件；默认 base 改为工作区根 |
| 14 | `loss.py` | 日志分量用 `bce_per_class.mean()`（含无效类的 0） | 读数被稀释，误导调参 | 只在有效类别上取均值，并输出 `valid_pixels_per_class` |
| 15 | `trainer/models/__init__.py` | `from .hr_adapter import ...` 指向不存在的模块 | H 臂构建直接 ImportError | 指向 `hrdecoder_adapter`；`build_model` 支持 `probe_family` |

## 3. 实际测试结果（本机 CPU，合成数据）

```
python -m py_compile train.py trainer/*.py trainer/data/*.py trainer/models/*.py tools/*.py tests/*.py
    → OK（29 个脚本）

python tests/run_contract_smoke.py
    → 12/12 PASS（P0 py_compile、P1 preflight 正式合成、N1/N2 配对、N3 test 泄漏、
       N4/N4b REQUIRED/TBD、P2 audit、N5/N6/N7 审计反例、R1 真实配置正式 preflight 非零退出）

python tests/run_training_smoke.py
    → 16/17 PASS，1 DEPENDENCY_BLOCKED（H 臂），rc=0
       T0  py_compile 29 脚本
       T1  tools/preflight_training 合成核心 FAIL=0
       T2  梯度累积按 optimizer update 计数（9 micro / accum=3 ⇒ 3 updates）
       T3  scheduler 按 update（LR 序列 == lr_multiplier 手算）
       T4  断点恢复 update 连续（save@2 → resume@2 → 4；LR 轨迹与一次性跑完对应段一致）
       T5  checkpoint 字段齐全（含 scheduler.update == update，weights_only 安全加载）
       T6  校准 checkpoint 拒绝作正式恢复源 / 正式 checkpoint 放行
       T7  formal 缺整图 evaluator ⇒ fail closed（0 个 update 被执行）
       T8  CLI 硬拒绝 test 训练 split（rc=1）
       T9  CLI formal 遇占位符 ⇒ rc=1
       T10 CLI formal 禁止校准覆盖 ⇒ rc=1
       T11 指标手算一致（TP/FP/FN、valid 屏蔽、严格 p>0.5、完美分类器 AUPR=1.0）
       T12 valid=0 整类零梯度；整批全无效零梯度且无 NaN
       T13 参数组互斥且完整（漏组/重复/冻结入组/前缀误并 四负例）
       T14 U 臂真实模型严格 N×4×H×W；非 32 倍数拒绝；detail_branch=true 拒绝
       T15 train.py 端到端 dry-run（合成 U0 配置 + 合成 DDR 树，calibration）rc=0，未写 ckpt
       T16 H 臂真实依赖状态 → DEPENDENCY_BLOCKED（不计 PASS）

python tools/preflight_training.py
    → PASS=20 FAIL=0 DEPENDENCY_BLOCKED=2 READINESS_BLOCKED=4 CALIBRATION_REQUIRED=1
       U0/U1: convnext_unet 52.99M 参数，前向/反向真实跑通
       H0/H1: DEPENDENCY_BLOCKED（reference_root 未回填）
```

**真实通过 = 20 项**；**DEPENDENCY_BLOCKED = 2 项（H 臂结构不可验）**，按约定不计 PASS。
没有把 DEPENDENCY_BLOCKED 算成 PASS。

## 4. 依赖阻塞与未清空的闸门

### 4.1 DEPENDENCY_BLOCKED（本机无法验证，但不算失败）

| 项 | 原因 | 解除条件 |
|---|---|---|
| H 臂模型结构 | 本机无 `mmcv` / `mmengine`；旧仓库 `HRNet` 依赖 `mmcv.cnn` + `mmengine.model` | 在服务器环境（torch 1.13 + mmcv-full 1.7.2 + mmengine）回填 `structure.reference_root` 后重跑 |

`trainer/models/hrdecoder_adapter.py` 会先探测 `mmcv/mmengine/mmseg` 可导入性与
`mmseg/models/backbones/hrnet.py` 存在性，任一不满足即抛 `DependencyBlocked`，
**本工作区不含任何自造 HRNet**。`train.py` 遇此情况以退出码 **3** 结束。

### 4.2 正式训练闸门（未清空，不得正式开训）

`python tools/preflight_contract.py` → **rc=1**，全部 `REQUIRED` 占位符即 blocker。
剩余 blocker 分类：

| 层 | 字段 |
|---|---|
| readiness（本机即可核验，**阻塞一切**） | U0/U1 的 `data.ddr_normalization.mean/std` |
| calibration（GPU/资产到位后回填，只阻塞**正式训练**） | `structure.reference_root`（H）、`structure.encoder_weights_*`、`structure.library_version`（U）、`initialization.imagenet_weights.*`、`initialization.external_pretrain_checkpoint`（H1/U1）、`pretrain_stage.{external_manifest,manifest_base_dir,init_checkpoint_output,overfit_monitor,micro_batch,grad_accum}`、`ddr_finetune_stage.{micro_batch,grad_accum,gradient_checkpointing}` |

`train.py --mode formal` 要求**两层都清零**；`--mode calibration` 只要求 readiness 清零。
`--layer readiness` 的 preflight 只是回答"能否先开一次短时校准"，
**不构**成正式训练放行依据。

## 5. 能否进入短时 GPU 校准

| 臂 | readiness | 结论 |
|---|---|---|
| H0 | `LOCAL_READY`（占位符全在 calibration 层） | ✅ 可开机做 dry-run / 显存校准 |
| H1 | `LOCAL_READY` | ✅ 可开机（pretrain 阶段需外部 manifest 到位） |
| U0 | `READINESS_BLOCKED`：`data.ddr_normalization.mean/std` | ❌ 先锁定 ConvNeXt 归一化统计 |
| U1 | 同上 | ❌ 同上 |

校准命令（服务器上、正式数据未接入前即可跑）：

```bash
# 1) 契约层：本机 readiness（H0/H1 应为 PASS）
python tools/preflight_contract.py --layer readiness

# 2) 训练链路：CPU 自检（含真实 U 臂前向反向 + checkpoint 往返）
python tools/preflight_training.py --output-json reports/preflight_training.json

# 3) 显存/micro_batch 校准：只做一次前向反向，不写 checkpoint
python train.py --config configs/H0_hrdecoder_ddr_only.yaml \
    --stage ddr_finetune --mode calibration \
    --micro-batch 1 --grad-accum 8 --gradient-checkpointing false \
    --out-dir work_dirs/calib_H0 --dry-run

# 4) 校准结果回填 YAML 后，重跑正式闸门（必须 rc=0 且无跳过项）
python tools/preflight_contract.py
```

## 6. 记录的冲突（以代码与真实目录为准，未擅自统一）

1. **`structure.library`（U0/U1）**：配置原为 `segmentation_models_pytorch`，
   但 v1 实现是本工作区纯 torch 的 ConvNeXt-S。保留一个不可构建的库声明
   属于"配置声称有依赖而代码没有"，故改 `library: builtin`（U0/U1 同步，
   `equivalence_lock` 仍逐字段一致）。`library_version` 仍为 REQUIRED，
   回填目标 GPU 栈的 torch/CUDA 版本。
2. **`structure.detail_branch`（U0/U1）**：配置原为 `true`，但 v1 无该支路、
   也无 shape test。按"配置不得声称 true 而代码无实现"统一改 `false`；
   `build_convnext_unet` 收到 `true` 时构建期即抛 `GateError`。
3. **旧仓库类序**：`D:\HRDecoder\tools\convert_dataset\ddr.py` 的
   `CLASSES = {EX:1, HE:2, SE:3, MA:4}` 与 `lesion_metric.sigmoid_confused_matrix`
   的 `pred = pred_logit[i-1] > thresh` 互相印证，通道序 `EX/HE/SE/MA`
   **无需置换**；与 `tools/contract_common.py` 一致，未发现冲突。
   实体 DDR 标签实测：`mode=L`，值域 `{0,1,2,3,4}`，无 255。
   目录实测：`labels/<split>/` 下除融合 PNG 外还有 `EX/HE/SE/MA` 子目录
   （逐类 TIF 存档），训练接口**不读取**这些子目录。
4. **`loss.per_class_valid_masking`**：原为 REQUIRED，现已在 `trainer/loss.py` 实现，
   四臂配置改为 `true`。
5. **external manifest 的 `official_split=test`**：见 §2 #11，由"硬报错"改为
   "排除并计数"，因为统一台账含 test 属正常；**test 记录绝不会进入训练池**
   （有断言覆盖）。

## 7. 五个最高风险

1. **H 臂结构从未在本机真实跑过**。`hrdecoder_adapter` 只有在服务器
   （mmcv/mmengine 可用）才第一次真正构建 HRNet-W48 + FCNHead，
   届时 `fcfeat` 的 4 尺度索引、`FCNHead` 的 `in_index=0`、BN `norm_eval`
   与 ImageNet 权重键名都可能暴露新问题。**这是全项目最大的未验证面。**
2. **旋转增强 + `valid` 屏蔽的联合语义**。`rotate_degrees: [-45,45]` 会把
   crop 角落填零并令 `valid=0`，DDR（本应四类全标注）的有效监督面积因此显著缩小；
   旋转角度越大、被屏蔽区域越多，等效 batch 的信息量越低。需要在校准阶段
   实测"每 update 的有效像素数"，否则 12000 update 的日程可能严重欠训。
3. **U0/U1 归一化统计未定**，且它同时是 readiness blocker 与输入分布定义。
   在锁定前，任何 U 臂数值（loss 量级、梯度范数、学习率是否合适）都不可比。
4. **DDR val 只有 76 张，且是固定单 split**。用 val 选 checkpoint 会产生
   选择偏差；mIoU 的方差会明显大于四臂之间的真实差异。当前设计**只用 val 一次**
   做最佳选择（`checkpoint_selection.allowed_splits: [val]`），
   但没有做重复种子/交叉验证——"H1 比 H0 好"这类结论需要显式的不确定性度量，
   否则很可能把噪声当效应。
5. **数据泄漏的残余面**：外部 manifest 的 `official_split` 只反映数据集官方划分，
   不反映**患者级**泄漏；IDRiD/MAPLES-DR 与 DDR 之间可能存在同一患者/同一相机
   的重复或近重复图像。`tools/audit_manifest.py` 有感知哈希近重复检测，
   但尚未与 DDR 图做跨数据集比对。若存在近重复，预训练臂会被不公平地抬高。

补充（次级风险）：`mIoU` 会剔除分母为 0 的类。DDR val 四类都有正样本，
但若某类在某次 eval 中 TP=FP=FN=0（模型与 GT 都为 0），该类的剔除会抬高均值，
需要在报告中持续检查 `excluded_classes` 是否非空。

## 8. 下一步（按优先级）

1. 服务器环境跑 H 臂：回填 `structure.reference_root` → `preflight_training` 的
   E3/E4/E5/E6 应从 DEPENDENCY_BLOCKED 变为 PASS。
2. 锁定 U 臂 `ddr_normalization`（按 ConvNeXt 官方预处理，实测编码器接口后回填）。
3. 用 `--mode calibration --dry-run` 实测 micro_batch / grad_accum /
   gradient_checkpointing，回填四份 YAML 的 calibration 层字段。
4. 产出并通过审计的 train-only 外部 manifest，H1/U1 指向**同一份文件**。
5. `python tools/preflight_contract.py` 必须 **rc=0 且无跳过项**，才允许
   `train.py --mode formal` 正式开训。
