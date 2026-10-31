# HRDecoder 历史实验 — 按类别最佳候选池（只读盘点）

生成时间：2026-09-20 · 范围：`D:\dataset\experiment`、`C:\Users\GuanXingRen\WorkBuddy`、`C:\Users\GuanXingRen\AppData\experment`
性质：**纯资产盘点**。不联网、不连服务器、不训练、不改动任何既有文件；不提出新网络、不运行推理。

---

## 0. 口径定义（判定"可比较"的唯一基准）

| 维度 | 基准值 |
|---|---|
| split | 454 train / 76 val / 227 test（`split_assignment.json`、`phaseA4_data_boundary.json` 均通过） |
| 类别通道 | 4 通道 EX / HE / SE / MA（ch0=EX,ch1=HE,ch2=SE,ch3=MA） |
| 二值化 | 严格 `p > 0.5` |
| 聚合 | **全 val（n=76）累计 TP/FP/FN** 后逐类 IoU，mIoU=四类算术平均，背景不计入 |
| TTA | 允许（但需登记；champion 含 hflip TTA） |
| test | **只登记，严禁用于候选排序** |

只接受同时满足以上口径的 **val** 结果进入排序。凡 split/口径/来源任一不明者 → `not_comparable`，不猜。

---

## 1. 冠军组成（显式列出，依 `full_eval_settings.json` + `RUN_PLAN.json`）

**CHAMPION_0 = 概率加权集成 { D×3 + E×2 + B×1 + E4×1 } + 水平翻转 TTA + H4 后融合 MA 校正头**

| 角色 | 名称 | 配置 | 服务器权重 | 本地权重（已验证存在） | 本地大小 |
|---|---|---|---|---|---|
| 成员 | **D** | `configs/lesion/exp/hrplus_patch_dice.py` | `D_student_2000.pth` (load=student_only) | `hrdecoder_plus3\results\D_0913_retry\backup_student_2000.pth` | 263,419,124 B |
| 成员 | **E** | `configs/lesion/exp/hrplus_mares.py` | `E_1500.pth` (load=full) | `HRDecoder_FusionAware_20260917\assets\E_1500.pth` | 527,412,743 B |
| 成员 | **B** | `configs/lesion/exp/hr100_b.py` | `B_weights.pth` (load=full) | `hrdecoder_plus3\repo\weights\b_weights.pth` | 263,479,238 B |
| 成员 | **E4** | `configs/lesion/exp/e4_detail_split613.py` | `E4_weights.pth` (load=full) | `autodl100\results\e4_reference_weights.pth` | 263,477,167 B |
| **H4 头** | **H4** | 后融合 MA 校正器 7ch→16→16→1，`Δ = 6·tanh(d)`，仅 MA 通道 | `H4_best.pth` | `HRDecoder_审查轮_20260916\audit_runs\audit_20260916_A\h4_best.pth` | 15,447 B |

- 融合方式：概率级（`sigmoid(logit)·w` 加权再归一），每图 4 成员 × 2(flip) = 8 次前向。
- H4 输入 7 通道：`[R,G,B, logit(P_fused_MA), logit(P_student_MA), logit(P_reference_MA), P_student−P_reference]`。
- 冠军 val（p0 复现）＝ `0.4266857717990218`；冠军 val（远端同机复现）＝ `0.42669428454584546`（两值冲突，见 §6-C1）。
- 冠军 test（仅登记）＝ `0.4421152424994267` / `0.442115035614162`。

---

## 2. 按类别 Top 10（仅可比 val；保留原始精度）

> 标记：`[H]`=high 置信口径；`[M]`=medium；`~`=合成值（非 MA 类别取自 BASE_0 契约）；`(F)`=融合体；`(2dp)`=源文件仅存 2 位小数。

### 2.1 EX（最高：MIX_C2000 0.46144…）
| # | 实验 | val IoU | 口径 | 权重可用 |
|---|---|---|---|---|
| 1 | MIX_C2000 (0.75 champ + 0.25 C@2000) | **0.46144246242771647** | [H] (F) | 由 champion+C 复现 |
| 2 | CHAMPION_0 (远端) | 0.45771525507464833 | [H] | 成员齐 |
| 3 | CHAMPION_0 (p0) | 0.4577047841571581 | [H] | 成员齐 |
| 4 | BASE_0（＝冠军非 MA 三类的来源） | 0.4577047841571581 | [H] | 成员齐 |
| 5 | T0_s1 / T1_s1 / U0_s1 / E1_* / C1_LOCKED | 0.4577047841571581 ~ | [H] 合成 | 远端头 |
| 6 | L (Dice+0.1 Lovasz) @6000 | 0.4605 (46.05) | [M] EvalHook | 无本地 |
| 7 | E @1500 / B_base（冻结基座） | 0.4498 (44.98) | [H] (2dp) | E:有 / B:有 |
| 8 | C_step_2000 | 0.44641456257082257 | [H] | **有** |
| 9 | B reference-only | 0.4424785186978095 | [H] | **有** |
| 10 | D_blend (student+ref 0.5) | 0.4403739050793795 | [H] | **有** |

### 2.2 HE（最高：D_BLEND 0.44727…）
| # | 实验 | val IoU | 口径 | 权重可用 |
|---|---|---|---|---|
| 1 | **D_blend (D student + B reference 0.5/0.5)** | **0.4472733283763723** | [H] | **有** |
| 2 | E2 @1500（D 基座同源） | 0.4473 (44.73) | [M] EvalHook | 有 |
| 3 | MIX_C2000 | 0.4415102912853148 | [H] (F) | 复现 |
| 4 | H9 (BCE-dominant HE head) | 0.4382 | [M] | 无本地 |
| 5 | CHAMPION_0 (远端) | 0.43758256131196327 | [H] | 成员齐 |
| 6 | CHAMPION_0 (p0) | 0.4375516601740191 | [H] | 成员齐 |
| 7 | H6 (HE 校正头，**已否决**) | 0.43602374795415394 | [H] | `head_HE.pth` 有 |
| 8 | C_step_2000 | 0.42964404397868905 | [H] | **有** |
| 9 | B reference-only | 0.41548180621072456 | [H] | **有** |
| 10 | U0 @12000 | 0.3842662751606068 | [H] | **有** |

> 关键发现：**HE 的最优不是冠军集成，而是 D_blend（0.44727 > 冠军 0.43758）**。这正是需要做像素级互补分析的理由之一。

### 2.3 SE（最高：MIX_C2000 0.55420…）
| # | 实验 | val IoU | 口径 | 权重可用 |
|---|---|---|---|---|
| 1 | MIX_C2000 | **0.5542037406936626** | [H] (F) | 复现 |
| 2 | CHAMPION_0 (远端) | 0.5518070087629399 | [H] | 成员齐 |
| 3 | CHAMPION_0 (p0) | 0.5518055058991777 | [H] | 成员齐 |
| 4 | D_blend | 0.5507727092517065 | [H] | **有** |
| 5 | L @6000 | 0.5327 (53.27) | [M] EvalHook | 无本地 |
| 6 | S_step_4000 | 0.5258420516141853 | [H] | **有** |
| 7 | E@1500 / B_base | 0.5184 (51.84) | [H] (2dp) | 有 |
| 8 | B reference-only | 0.5162428177387987 | [H] | **有** |
| 9 | U0 @12000 | 0.5093623706425702 | [H] | **有** |
| 10 | C_step_4000 | 0.5068666895383048 | [H] | 未回收 |

### 2.4 MA（最高：H4 0.25968…）
| # | 实验 | val IoU | 口径 | 权重可用 |
|---|---|---|---|---|
| 1 | **H4（冠军 MA 头）** | **0.25968113696573225** / 远端 0.25967231303383037 | [H] | **有** |
| 2 | MIX_C2000 | 0.2574711078252072 | [H] (F) | 复现 |
| 3 | H8_lr2e3 @6000 | 0.2569 | [M] | 无本地 |
| 4 | H12（whole-image 监督） | 0.2565 | [M] | 无本地 |
| 5 | H11（BCE-dominant） | 0.2510 | [M] | 无本地 |
| 6 | C1_LOCKED (U0·0.5+T1·0.5) | 0.248429 | [M] | 无本地 |
| 7 | E1_T0 / T1_s1 | 0.247878 / 0.24786845919336298 | [H]/~ | 远端头 |
| 8 | U0_s1 | 0.247509 | [M] | 无本地 |
| 9 | T0_s1 | 0.24703834348802053 | [H] ~ | 远端头 |
| 10 | MA expert R6 (α=0.3) | 0.24375282009411056 | [H] | `ma_expert_r6.pth` 有 |

（其后：R6b 0.24305823784386937 / S_step_4000 0.22927309605366145 / D_student 0.22783988125608795 / U0 0.20746772322571896。）

---

## 3. 去重规则与执行结果

1. **同一实验相邻 checkpoint**：`C@2000` vs `C@4000`、`S@2000` vs `S@4000` → 各保留 val 更优且有本地权重者（C@2000、S@4000）。
2. **明显同构重复（小头家族）**：`T0_s1 / T1_s1 / U0_s1 / E1_T0 / E1_T1 / C1_LOCKED / H8 / H11 / H12` 共用同一契约——EX/HE/SE 恒等于 BASE_0，仅 MA 不同（0.247~0.257）。同构 → **折叠为 1 个代表**（T1_s1，MA 最高者之一），且该家族**无本地权重**，本就不进 GPU 短名单。
3. **同 ckpt 多口径**：`B_reference / B_base`、`D_student / D_blend` 为同一权重的不同评估上下文 → 各保留 1 条（D 取 blend 形态以覆盖 HE）。
4. **冠军两变体**（p0 / 远端）为同一配方两次评估 → 视为 1 个模型，两值均登记为证据。
5. **MA 头家族**：H4 为最优，H3/H6/H9/H11/H12/ma_expert_r6/r6b 均为被否决或更弱分支，仅作机理证据。

---

## 4. 最终候选集（≤ 8 个模型）

| 槽位 | 模型 | 类别强项 | 入选理由（单类最优 / 机理差异） | 本地权重 |
|---|---|---|---|---|
| 1 | **D**（hrplus_patch_dice，student_only） | HE 0.44727、EX 0.4404、SE 0.5508 | 冠军成员；**HE 全池最高**（blend 形态） | ✅ `backup_student_2000.pth` |
| 2 | **E**（hrplus_mares，MA 残差头） | MA 0.2281、EX/HE/SE=0.4498/0.4139/0.5184 | 冠军成员；唯一带残差头的成员，MA 机理代表 | ✅ `E_1500.pth` |
| 3 | **B**（hr100_b，reference） | EX 0.4425、SE 0.5162 | 冠军成员；E 的冻结基座、D 的 reference 配对 | ✅ `b_weights.pth` |
| 4 | **E4**（e4_detail_split613，细节分支） | SE/EX 结构性补充 | 冠军成员；**唯一细节分支架构**，形态多样性 | ✅ `e4_reference_weights.pth` |
| 5 | **H4**（后融合 MA 校正头） | MA 0.25968（全池最高） | 冠军 MA 头；独占 MA 单类最优 | ✅ `h4_best.pth`（head-only） |
| 6 | **C_step_2000**（class-balanced 实例采样） | EX 0.4464 / HE 0.4296 | 采样机理差异；MIX_C2000 另一成员 | ✅ `C_step_2000.pth` |
| 7 | **S_step_4000**（class-balanced 条件采样） | SE 0.5258（单模型最高） | 采样机理差异；SE 单类次优 | ✅ `S_step_4000.pth` |
| 8 | **U0**（ConvNeXt-S+U-Net，whole-image） | HE 0.3843 / SE 0.5094 | **架构族差异**（全新骨干）；口径**完全 canonical** | ✅ `runs\ckpt_best.pth` |

- **未因"最近"而优先**：U0（2026-09-20）val 0.38446、H0 val 0.36923，均**低于**冠军 0.42669；U0 仅因架构差异入选；**H0 因无本地权重被排除**（见 §5）。
- **组合覆盖**：BASE_0（0.41796）与 MIX_C2000（0.42866）无需独立权重——前者 = D/E/B/E4 集成，后者 = champion + C_step_2000，均可由本候选集复现。
- **被排除**：所有 test-only 候选、所有无 checkpoint 候选（H0、T 家族头、H8/H11/H12、R5 BNEW/V1/V2、C@4000、S@2000、AppData 八月全部）。

---

## 5. 资产存在性核验（实测 stat）

**存在（可用于 3090 重算）**：`backup_student_2000.pth`(263MB)、`b_weights.pth`(263MB, 另 `autodl100\results\trial_b\selected_weights_only.pth` 同哈希)、`e4_reference_weights.pth`(263MB)、`E_1500.pth`(527MB, 含 meta+optimizer)、`h4_best.pth`(15KB)、`head_HE.pth`(15KB)、`ma_expert_r6.pth`/`ma_expert_r6b.pth`(7.7MB)、`C_step_2000.pth`(718MB)、`S_step_4000.pth`(718MB)、`runs\ckpt_best.pth`(636MB, U0, sha256 `98d98a5d…`)、`trial_a\selected_weights_only.pth`。

**缺失 / 不可用**：
- `H0`：仅 `H0_stopped6000_metadata.tar.gz` 元数据，**无 .pth** → 排除出 GPU 短名单。
- `C_step_4000.pth`、`S_step_2000.pth`：finalize manifest 引用但**未回收**。
- `D_full_2000.pth.part`：**分片**，非完整权重，勿用。
- T 家族头、H8/H11/H12、R5 `BNEW/V1/V2`：仅服务器路径。
- AppData 八月全部 checkpoint（baseline/E4/E1v2/combo/E2E3/E4v2）：**本地不存在**；`D:\dataset\lesion_segmentation\server_checkpoints\` 路径不存在。

---

## 6. 证据冲突、口径疑点与缺失权重

**冲突**
- **C1（未解）**：冠军 val 两值 `0.4266857717990218` vs `0.42669428454584546`（Δ≈8.5e-6）；test 两值 `0.442115035614162` vs `0.4421152424994267`。同配方两次评估，逐类亦有 ~1e-4 差。文件层面无法判定谁对。
- **C2**：`ROUND_CS_FINAL_REPORT_20260920.md` 写冠军 val 42.6694%，却与其同目录 `evalcand_summary.json` 的 42.6694 一致、与 `p0_report.json` 的 42.6686 不一致——树内自相矛盾。
- **C3**：H4 头步数 文档 2000 vs 实测 `best.step=1600`（`h4_post_fusion_corrector.json`）→ 采用 **1600**。
- **C4**：同一 B 权重两个 HE 值：reference-only `0.41548180621072456` vs B_base `41.39`(0.4139)。**未解**。
- **C5**：`champion_asset_inventory_workbuddy.md` §4.4 断言 H0@6000 本地无出处；实际 `H0_stopped6000_metadata.tar.gz` 内 `eval_u6000.json` 逐字命中（该盘点文档早于新跑）。
- **C6**：盘点文档把 `D_full_2000.pth.part` 路径写成 `hrdecoder_plus3/results/D_0913_retry/`，实际在 `HRDecoder_FusionAware_20260917\assets\`。
- **C7**：`eval_VAL_D_renorm_all.json`(0.25286) 是 **已知伪影**（reference BN 访问计数=0），非模型质量结果；仅 `renorm_student`(0.40363) 有效。
- **C8**：`h4_test_confirmation.json` 的 fused 基线是 **D 单模型**（test 0.42576），与 champion 集成 fused（test 43.2289）不可混用。
- **C9**：`全部结果.csv` 的 "VAL D 单模型" 41.6148 实为 **0.5 blend** 而非 student-only（同表另有 student 单测 42.5764），命名有歧义。

**口径疑点**
- E/B 逐类 P/R 在 eval_json 中仅存字符串 2 位小数（`"44.98"` 等），**全精度不可恢复**。
- `L / E2 / E3` 的 val 来自 train-log EvalHook（whole-image），与 eval_ens 分块口径不同 → [M]。
- 多数 eval JSON **未记录 seed**；seed=42 仅见于叙述文档。
- 旧评估器 `load_report` 曾把 checkpoint 误拆包，故其 "missing:0/unexpected:0" **不能作为加载成功证据**。
- **AppData 八月项目**（week1/week2）split 相符，但指标为**逐图平均**（非全 val 累计 TP/FP/FN）→ 全部 `not_comparable`，只登记。
- `最佳点核对.csv` 的 `n_val` = 10/9/6/5/1/0 ≠ 76，其中 `combo_20260817_204954` n_val=1 → 明确排除。
- test 已被多次查看（报告自述 ≥5 次），**非独立泛化估计**；`MIX_C2000` val +0.1963 却在 test 回退 −0.4052，是典型 val 选择偏差样本。

**缺失权重**：H0；T0_s1/T1_s1/U0_s1/E1_*/C1 头；H3/H8/H11/H12；R5 BNEW/V1/V2；C@4000、S@2000；AppData 全部；`D_full_2000.pth.part`（分片）。

---

## 7. 推荐重算顺序（仅资产盘点结论，3090 / val-only）

> 全部只用 val；test 保持锁定，不得据其结果调参。每步先复现"预期 val"以校验环境。

| 步 | 模型 | 预期 val（复现目标） | 目的 |
|---|---|---|---|
| 1 | B (`b_weights.pth`) | EX/HE/SE/MA ≈ 0.4425/0.4155/0.5162/0.1940（ref-only） | 最小风险、校验评估管线 |
| 2 | E4 (`e4_reference_weights.pth`) | 仅登记（无 canonical val 单列） | 补齐冠军成员 |
| 3 | D student (+B) → D_blend | 0.41614841513965567（EX 0.4404/HE **0.4473**/SE 0.5508/MA 0.2262） | 锁定 **HE 最优** |
| 4 | E (`E_1500.pth`) | 0.4025462524944335（MA 22.81） | MA 残差头基线 |
| 5 | 组合 BASE_0 = D×3+E×2+B+E4 + flip | 0.41796267336729404（0.4577047841571581/0.4375516601740191/0.5518055058991777/0.2247887432388212） | 复现集成基线 |
| 6 | + H4 头 (`h4_best.pth`) | 0.4266857717990218（MA 0.25968113696573225） | 复现冠军 |
| 7 | C_step_2000 → MIX_C2000 | 0.3979739003932351 → 0.4286569005579753 | 采样互补 + 融合验证 |
| 8 | S_step_4000 | 0.39831982892663487（SE 0.5258） | SE 互补 |
| 9 | U0 (`ckpt_best.pth`) | 0.3844641257662971 | 架构多样性 |

**像素级互补分析建议重点**（仅观察，不预设结论）：HE 上的 **D_blend vs 冠军集成**；SE 上的 **S_step_4000 vs C/D**；MA 上的 **H4 vs R6/R6b**；EX 上的 **C vs 集成**。全部以 §2 的逐类 IoU 与逐图 TP/FP/FN 为依据（凡有逐图计数的记录已在 CSV 登记）。
