# HRDecoder 专家互补性轮 — 执行报告（2026-09-20）

执行代理：WorkBuddy CLI（本轮实际执行，非计划稿）
唯一访问服务器：`root@connect.nmb2.seetacloud.com:48301`（RTX3090，未关机，未访问 44795/4090）
服务器根目录：`/root/autodl-tmp/HRDecoder_FullCompetition_20260920`
本地根目录：`D:\dataset\experiment\HRDecoder_FullCompetition_20260920`
设置文件：`analysis/expert_eval_settings.json`（sha256 `c44992e8c0d9628689f6ee48a17c6e45e22f347608a8f3bfd1f3916d23872d6d`）

**全程约束遵守情况**：未使用 test 数据（契约层 `FORBIDDEN_PATH_COMPONENTS=("test",)` 硬闸门 + `test_val_only_guard` 单测通过）；未启动任何训练；未并发 GPU 作业（GPU 在全部本代理作业前后均为 `1 MiB / 0 %`）；未下载任何 checkpoint/权重。

---

## 0. 续跑起点核查（接手时服务器与本地现状）

前一个 CLI 回合被外部中断，接手时实测：

| 项 | 接手时状态 | 判定 |
|---|---|---|
| `expert_outputs/hrplus_exit_code.txt` | `2`（时间戳 21:40，2 字节） | **陈旧**：属 21:40 那次失败的导出 |
| HRPlus 六目录 npz | 各 76 个，时间戳 21:28–21:54 | 已完成，未重跑 |
| C2000 / MIX_C2000 / S4000 contract | 已离线补齐（21:54:50） | 已完成 |
| U0 / H0 smoke | `logs_expert_comp/smoke_{U0,H0}.log` 存在（22:03 / 22:05） | 已完成，未重跑 |
| U0 / H0 完整 76 张导出 | `full_U0H0.log`（22:10–22:14），contract 已写 | 已完成，未重跑 |
| 互补性分析 | `analysis/` 五个产物（22:16:36） | 已完成 |
| 本地回收 | 已有 contracts / logs / manifests / analysis_outputs / scripts | 部分完成，本轮补齐 |
| GPU | 1 MiB / 0 %，无训练进程 | 空闲 |

**结论：未重复任何已完成的 GPU 推理。** 本轮实际新增工作为：契约修复的独立复核与回归测试实测、npz 结构与目录覆盖的独立取证、TP/FP/FN 的独立重算、scalar leaderboard 交叉核对、全量本地回收、本报告。

---

## 1. 失败根因与修复（任务 1）

### 1.1 根因

`export_hrplus_expert_masks.pipeline_definition()` 把一个**纯文字说明**的元数据字段命名为 `"probability"`，撞上 `expert_mask_contract.FORBIDDEN_CONTRACT_KEYS` 硬黑名单，`assert_no_probability_payload()` 在 `contract[C2000]` 处 fail closed → `hrplus_exit_code.txt = 2`。

关键判断：**失败发生在 champion / BASE_0 / D_blend 已成功写 contract 之后、C2000 处**，所以是纯元数据命名冲突，不是真的把概率数组写进了契约。

### 1.2 修复（只收紧、不放松）

`analysis/expert_mask_contract.py:228` `assert_no_probability_payload()` 现为两条互相独立、都不可绕过的规则：

- **规则 (1) 键名硬黑名单（未放松）**：`FORBIDDEN_CONTRACT_KEYS = ("prob","probs","probability","prob_fp32","prob_cache","logits","probabilities")` 一旦出现即拒绝，**无论值是什么**（连字符串也拒绝）。没有为字符串开任何后门。
- **规则 (2) 类型禁令（新增，加强项）**：与键名无关，契约中**任何位置**出现浮点数组（`np.issubdtype(float)`、torch `float16/half/bfloat16`）或**纯浮点 list/tuple** 一律拒绝。把概率改名成 `"data"` 也照样拦下。孤立浮点标量（iou/threshold 等）**允许**，避免误伤指标字段。

修复动作是把元数据字段改名为 `probability_source`（`export_hrplus_expert_masks.py:575`、`:582`），而不是给保留键开字符串后门。

### 1.3 回归测试（已实测）

`tests/test_packed_mask_contract.py`，服务器 CPU 实跑（**12/12 passed, ALL PASS, exit 0**）：

```
[PASS] test_probability_key_blacklist_and_float_payload_ban     <- 本次缺陷的直接回归
[PASS] test_export_scripts_do_not_use_reserved_keys             <- 静态扫描导出脚本不得使用保留键
[PASS] test_pack_unpack_roundtrip_varying_hw
[PASS] test_unpack_rejects_padding_and_length
[PASS] test_strict_threshold_and_precision
[PASS] test_counts_hand_computed_with_ignore
[PASS] test_or_and_and_decomposition_identity
[PASS] test_iou_zero_union_fails_closed
[PASS] test_val_only_guard
[PASS] test_contract_integrity_and_atomic_write
[PASS] test_gt_idempotent_and_tamper_detection
[PASS] test_end_to_end_complementarity_synthetic
```

日志：`logs/test_contract.log`

### 1.4 离线补写 contract（未重跑 GPU）

`rebuild_hrplus_contracts.py` 从**现有 76 张输出**离线补写 C2000 / MIX_C2000 / S4000 的 contract。掩膜本身仍来自原 run，`masks_produced_by.script_sha256=12a90992…`。见 `manifests/REBUILD_REPORT.json`（`gpu_used:false, test_used:false, verified_against_disk:true`）。

**核对结果 — 与预期完全一致（Δ = 0）：**

| 管线 | 预期 mIoU | 实测 mIoU（独立重算） | Δ |
|---|---|---|---|
| C2000 | ≈ .397974 | 0.3979739003932351 | 0 |
| MIX_C2000 | ≈ .428657 | 0.4286569005579753 | 0 |
| S4000 | ≈ .398320 | 0.39831982892663487 | 0 |

### 1.5 npz 结构 / 每目录 76 个 / 无覆盖

- 8 个模型目录 **各 76 个 npz**，`_shared/gt` 76 个。
- 全部目录 **唯一一套 key/dtype 签名**（`n_key_signatures=1`, `n_dtype_signatures=1`）：
  - 模型：`packed` uint8 `[4, ceil(C·H·W/8)]`、`n_pixels` int64 标量、`orig_hw` int64 `[2]`
  - GT：额外 `ignore_packed` uint8 `[ceil(H·W/8)]`
  - 无任何浮点数组 → 与"只允许二值 packbits"一致。
- **无跨目录覆盖**（28 个两两组合全查）：唯一 `nonMA_diff=0` 的是 `champion|BASE_0`，即冠军的 EX/HE/SE 与 BASE_0 **逐位相同**（契约 `self_checks.champion_shared_bitwise_equal=true, max_abs_diff=0.0`）；其 MA 在 76 张中有 72 张不同，符合"= BASE_0 + H4 MA 头"的定义。其余 27 个组合 `nonMA_diff_stems ≥ 74`，确认六模型输出目录互相隔离。

---

## 2. U0 / H0 单图 smoke（任务 2）

`analysis/export_fullmodel_masks.py` 的守卫（代码复核 + 日志实证）：

| 要求 | 实现 | 证据 |
|---|---|---|
| 严格 checkpoint 加载 | `load_checkpoint(weights_only=True)` + `load_state_dict(strict=True)` + **逐张量 bitwise 回验**；`sha256` 与 settings 声明不符即拒 | `export_fullmodel_masks.py:211–220` |
| 四类二值 packed mask | `binarize_strict`：先 `check_float32_like` 拒 fp16，再严格 `p > 0.5`，输出 `np.packbits(bitorder="big")` | `:359`，与 HRPlus 同一契约模块 |
| 独立目录 | `logs_expert_comp/smoke_masks/{U0,H0}/`，与 HRPlus / 正式输出隔离 | 服务器实测存在 |
| 无预训练联网 | `disable_pretrained_init()` 强制把 `structure.encoder_weights_file` / `initialization.imagenet_weights.file` 覆盖为 `<disabled>`；若发现已应用外部初始化则直接拒绝出图 | `:137–161`、`:209`；**U0/H0 contract 的 `identity.config_overrides_applied` 已实测记录这两条禁用项，且 `imagenet_init_applied_before_checkpoint = false`** |
| 验证口径一致 | val-only 闸门 + `expected_miou` 仅在 76 张整轮口径比对 | `:413–429`；U0/H0 contract `eval_contract.split = val`、`expected_images = 76`、`prob_payload_saved = false`、`fp16_probability_cache = false` |

smoke 实测：U0 1/1 用时 2.5 s、H0 1/1 用时 2.7 s，均正常出图并**正确地拒绝**在 1 张口径下比对期望 mIoU、拒绝写 contract（fail closed 设计正确）。
日志：`logs/smoke_U0.log`、`logs/smoke_H0.log`。

---

## 3. U0 / H0 完整 76 张导出与历史 mIoU 核对（任务 3）

顺序执行，全部 76 张，自动比对，**Δ = 0（精确相等，非舍入内）**：

| 模型 | 历史期望 val mIoU | 本轮实测 | Δ | contract |
|---|---|---|---|---|
| U0 | 0.3844641257662971 | 0.3844641257662971 | 0.000e+00 | 已写 |
| H0 | 0.36922642429539565 | 0.36922642429539565 | 0.000e+00 | 已写 |

U0 per-class EX .436760 / HE .384266 / SE .509362 / MA .207468
H0 per-class EX .418771 / HE .435195 / SE .414545 / MA .208394
日志：`logs/full_U0H0.log`；清单 `manifests/RUN_MANIFEST_fullmodel.json`（`complete:true`）。
**无超差项，未触发停查。**

---

## 4. 互补性分析（任务 4）

`analyze_expert_complementarity.py` 对 8 个模型（champion、BASE_0、D_blend、C2000、MIX_C2000、S4000、U0、H0）完成逐像素分析，val n=76，全程未读 test。产物：`analysis_outputs/`（JSON + MD + per_class/per_image/area_buckets 三个 CSV）。

### 4.1 各模型 val 指标（位图重算）

| 模型 | mIoU | EX | HE | SE | MA |
|---|---|---|---|---|---|
| champion | 42.6694 | 45.7715 | 43.7583 | 55.1807 | **25.9672** |
| MIX_C2000 | **42.8657** | **46.1442** | **44.1510** | **55.4204** | 25.7471 |
| BASE_0 | 41.7973 | 45.7715 | 43.7583 | 55.1807 | 22.4788 |
| D_blend | 41.6149 | 44.0397 | **44.7262** | 55.0758 | 22.6179 |
| C2000 | 39.7974 | 44.6415 | 42.9644 | 49.6322 | 21.9515 |
| S4000 | 39.8320 | 43.0961 | 40.7203 | 52.5842 | 22.9273 |
| U0 | 38.4464 | 43.6760 | 38.4266 | 50.9362 | 20.7468 |
| H0 | 36.9226 | 41.8771 | 43.5195 | 41.4545 | 20.8394 |

### 4.2 相对冠军的逐像素互补（候选独有 TP / 候选独有 FP）

| 候选 | EX (TP/FP) | HE (TP/FP) | SE (TP/FP) | MA (TP/FP) | OR mIoU | ΔOR |
|---|---|---|---|---|---|---|
| BASE_0 | 0/0 | 0/0 | 0/0 | 1718/11559 | 42.3931 | −0.2764 |
| D_blend | 5775/28930 | **28629/52006** | 5066/11452 | 1507/11163 | 41.8995 | −0.7699 |
| C2000 | 14197/40629 | 54159/127037 | 12655/38271 | 2267/13782 | 41.3907 | −1.2788 |
| **MIX_C2000** | **4070/9558** | **15842/25084** | **3974/7132** | 400/1378 | **42.7387** | **+0.0693** |
| S4000 | 15715/59307 | 34516/125701 | 11363/32497 | 3477/22837 | 40.6885 | −1.9809 |
| U0 | 17007/62318 | 79689/287515 | 10293/35495 | 6121/45368 | 39.6819 | −2.9875 |
| H0 | 6853/17570 | 85883/196496 | 4683/33165 | 3934/19149 | 41.1915 | −1.4779 |

### 4.3 单类替换 mIoU（标量，可实现的"只换一类"口径）

把冠军的第 c 类换成候选的第 c 类、其余三类保持冠军，重算整体 mIoU（champion 基线 0.4266942845）：

| 候选 | EX | HE | SE | MA |
|---|---|---|---|---|
| **MIX_C2000** | **+0.000932** | **+0.000982** | **+0.000599** | −0.000550 |
| D_blend | −0.004330 | **+0.002420** | −0.000262 | −0.008373 |
| BASE_0 | 0.000000 | 0.000000 | 0.000000 | −0.008721 |
| C2000 | −0.002825 | −0.001985 | −0.013871 | −0.010039 |
| S4000 | −0.006689 | −0.007595 | −0.006491 | −0.007600 |
| U0 | −0.005239 | −0.013329 | −0.010611 | −0.013051 |
| H0 | −0.009736 | −0.000597 | −0.034315 | −0.012820 |

---

## 5. 独立重算与交叉核对（任务 5）

`analysis/independent_recompute.py`（**不 import 分析器/契约模块**，自实现 `np.unpackbits` 解包 + 自实现 TP/FP/FN/IoU）。

### 5.1 三方一致（全部精确相等）

| 模型 | contract mIoU | analysis JSON mIoU | 独立重算 mIoU | Δ |
|---|---|---|---|---|
| champion | 0.42669428454584546 | 0.42669428454584546 | 0.42669428454584546 | 0 |
| BASE_0 | 0.4179731494431753 | 0.4179731494431753 | 0.4179731494431753 | 0 |
| D_blend | 0.4161489586241723 | 0.4161489586241723 | 0.4161489586241723 | 0 |
| C2000 | 0.3979739003932351 | 0.3979739003932351 | 0.3979739003932351 | 0 |
| MIX_C2000 | 0.4286569005579753 | 0.4286569005579753 | 0.4286569005579753 | 0 |
| S4000 | 0.39831982892663487 | 0.39831982892663487 | 0.39831982892663487 | 0 |
| U0 | 0.3844641257662971 | 0.3844641257662971 | 0.3844641257662971 | 0 |
| H0 | 0.36922642429539565 | 0.36922642429539565 | 0.36922642429539565 | 0 |

- **608 个逐文件 sha 核对（8 目录 × 76）：0 处不符。**
- **逐类 `global_counts`：8 个模型 × 4 类 = 32 组，0 处差异。**
- 逐像素 OR / AND / candidate_only_tp / candidate_only_fp / champion_only_tp：**28 组单元格（7 候选 × 4 类），每格 5 个字段，全部与 analysis JSON 6 位小数一致。**

### 5.2 scalar leaderboard 与 mask 结果核对

对 `analysis/historical_class_leaderboard.csv` 中可比的 val 标量逐条比对：

| 条目 | leaderboard | mask 重算 | Δ | 判定 |
|---|---|---|---|---|
| CHAMPION_REMOTE | 0.42669428454584546 | 0.42669428454584546 | 0 | 一致 |
| MIX_C2000 | 0.4286569005579753 | 0.4286569005579753 | 0 | 一致 |
| C_step_2000 | 0.3979739003932351 | 0.3979739003932351 | 0 | 一致 |
| S_step_4000 | 0.39831982892663487 | 0.39831982892663487 | 0 | 一致 |
| U0_12000 | 0.3844641257662971 | 0.3844641257662971 | 0 | 一致 |
| H0_6000 | 0.36922642429539565 | 0.36922642429539565 | 0 | 一致 |
| BASE0 | 0.41796267336729404 | 0.4179731494431753 | **+1.05e−5** | 见下 |
| D_BLEND | 0.41614841513965567 | 0.4161489586241723 | **+5.4e−7** | 见下 |

两处小差异**不是新错误**，而是 leaderboard 里已登记的 **champion "variant A / variant B" 证据冲突（C1）**：`BASE0` 行与 `D_BLEND` 行的 EX/HE/SE 取自 variant A（EX 0.4577047841571581…），而本轮同一主机复现的 variant B 为 EX 0.45771525507464833…。本轮重算与 **CHAMPION_REMOTE（variant B，同主机复现）精确一致**，属自洽口径。**建议后续统一到 variant B，并在 leaderboard 中把 BASE0/D_BLEND 的 variant A 数值显式标注。**

### 5.3 真正有净互补价值的类别 / 模型

判据：单类替换 mIoU > 0（可实现口径）且 OR 上限为正。

- **MIX_C2000 = 0.75×champion + 0.25×C@2000 — 唯一在多类上呈净正贡献的模型**：EX +0.000932、HE +0.000982、SE +0.000599；MA −0.000550。它是唯一 OR mIoU 超过冠军的候选（42.7387 vs 42.6694，+0.0693 点）。其候选独有像素精度 EX 0.299 / HE 0.387 / SE 0.358，优于同族 C2000（0.259/0.299/0.248）。
- **D_blend 仅在 HE 单类有净正贡献**（+0.002420，候选独有 HE TP 28629 / FP 52006）；其余三类全负。
- **C2000、S4000、U0、H0 无任何净互补价值**：单类替换四类全负，OR 上限亦全负。H0 的 SE 尤其差（替换后 −0.034315）。
- **BASE_0 对 MA 是负贡献**（−0.008721），反向确认 H4 头对 MA 的价值（+0.034885 点 MA IoU）是整个冠军增量的来源。

**oracle 只能称上限**：上表 OR 列是"逐像素取并集"的oracle，需要每像素真值才可构造，**不是可实现融合**。analysis JSON 的 `caveats` 已显式写明；`direct_four_class_combination`（0.43064503817412）是**在 val 上逐类 argmax 选出来的**，属选择偏差样本，`note` 字段亦已声明。

**元结论**：所有候选的互补增益（最大 +0.0020 点）都远低于历史配对 bootstrap 95% 区间的宽度（`[-0.5454,+0.8119]`，ROUND_CS；`[+0.5718,+1.1815]`，H4−BASE_0）。历史上 val 锁定的 0.75×champion+0.25×C@2000 在 val 上 +0.1963、在 test 上回退 −0.4052。**本轮分析不产出任何可上线的融合方案。**

---

## 6. 本地回收（任务 6）

回收目标：`D:\dataset\experiment\HRDecoder_FullCompetition_20260920\analysis_results\expert_complementarity_20260920\`
**共 1416 个文件 / 9.77 MB**，服务器原件全部保留，未下载任何 checkpoint。

```
analysis_results/expert_complementarity_20260920/
├── COLLECTION_MANIFEST.json          1416 条逐文件清单（远端路径/本地路径/字节数/sha 状态）
├── scripts/
│   ├── analysis/   analyze_expert_complementarity.py, expert_mask_contract.py,
│   │               export_hrplus_expert_masks.py, export_fullmodel_masks.py,
│   │               rebuild_hrplus_contracts.py, diag_fullmodel.py,
│   │               independent_recompute.py, expert_eval_settings.json(+example),
│   │               README_EXPERT_COMPLEMENTARITY.md
│   └── tests/      test_packed_mask_contract.py
├── contracts/
│   ├── hrplus/{champion,BASE_0,D_blend,C2000,MIX_C2000,S4000}/contract.json   (6)
│   ├── fullmodel/{U0,H0}/contract.json                                        (2)
│   └── _shared/                                                                (77)
├── masks/                                                              （附带证据，共 684 npz，约 4 MB）
│   ├── hrplus/ 6 模型 × 76 = 456 ；fullmodel/ 2 模型 × 76 = 152 ；_shared/gt/ 76
├── logs/  hrplus_export.log, hrplus_exit_code.txt, smoke_U0.log, smoke_H0.log,
│          full_U0H0.log, analyze.log, test_contract.log, independent_recompute.log, smoke_masks/
├── manifests/  REBUILD_REPORT.json, RUN_MANIFEST_hrplus.json, RUN_MANIFEST_fullmodel.json
├── analysis_outputs/  expert_complementarity.{json,md}, _per_class.csv, _per_image.csv, _area_buckets.csv
├── independent_verification/  independent_recompute.json（含 32 组逐类计数 + 56 组 OR/AND 分解）
└── EXECUTION_REPORT_20260920.md   ← 本文件
```

npz 掩膜超出清单要求，但仅约 4 MB 且是全部结论的原始证据，故一并纳入以便本地复算。

---

## 7. 失败项 / 风险 / 未决

| # | 项 | 状态 | 说明 |
|---|---|---|---|
| F1 | `hrplus_exit_code.txt = 2` | **保留为历史证据，未篡改** | 该文件是 21:40 那次失败 run 的真实退出码。契约已在 21:54 离线补齐且三方核对通过。如需消除歧义，建议**另加**说明文件而非改写该文件。 |
| F2 | BASE0 / D_BLEND leaderboard 标量与 mask 重算差 1.05e−5 / 5.4e−7 | **已定位，非新错误** | champion variant A / variant B 冲突（leaderboard 已登记为 C1）。本轮重算与 variant B 精确一致。 |
| F3 | 所有互补增益 < 噪声 | **不构成上线依据** | 最大净增益 +0.0020 点，远小于 bootstrap 95% 区间宽度；且有 val→test 回退 −0.4052 的先例。 |
| F4 | test 已被查看 ≥5 次 | **不可逆** | val 独立性无法恢复；本轮严格未触 test，但泛化估计已不可信。 |
| F5 | H0 本地无 `.pth` | 已解除 | 服务器权重存在且 sha256 匹配，本轮已成功导出并精确复现历史 mIoU。 |

## 8. 可执行的下一步

1. **不要**基于本轮结果做融合上线。若仍要探索，唯一有信号的方向是 MIX_C2000 的 EX/HE/SE 三类，且必须先在**独立验证集**上验证，不能在 val 上调参。
2. 若要落地 C 类增益，应把 0.75/0.25 的**权重搜索限制在 val 之外**的协议内固定，而不是在 val 上取 argmax。
3. 统一 champion 口径到 variant B，并把 leaderboard 中 BASE0 / D_BLEND 的 variant A 数值显式标注，消除 C1 冲突。
4. MA 是冠军唯一的净增量来源（BASE_0→champion MA +0.034885），也是全池最弱类（冠军 MA IoU 仅 25.97）。后续收益应优先投在 MA 的独立验证协议上，而非在 EX/HE/SE 上加融合复杂度。
5. 保留服务器原件；本轮未关机、未删改任何服务器文件。

---

**本报告绝对路径**
`D:\dataset\experiment\HRDecoder_FullCompetition_20260920\analysis_results\expert_complementarity_20260920\EXECUTION_REPORT_20260920.md`
