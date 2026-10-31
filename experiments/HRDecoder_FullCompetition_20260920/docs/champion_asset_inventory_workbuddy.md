# HRDecoder 冠军资产盘点（WorkBuddy 本地只读核查）

盘点日期：2026-09-20 ｜ 执行者：WorkBuddy DeepSeek v4.1 Flash
工作目录：`D:\dataset\experiment\HRDecoder_FullCompetition_20260920`

读取范围（只读，未修改、未删除、未训练、未联网）：
`D:\dataset\experiment`、`C:\Users\GuanXingRen\WorkBuddy`、`C:\Users\GuanXingRen\AppData\experment`。

核查口径：**以本地实际文件的原文数值与哈希为准**；服务器上已知的 5 条权重路径仅作为对照锚点（D/E/B/E4/H4）。文档转述与本地原始证据冲突处显式标注。

---

## 1. 结论摘要

1. **冠军 = BASE_0（4 模型概率加权集成 + 翻转 TTA）+ H4 后融合 MA 校正头**，加权 3:2:1:1（D:E:B:E4），与服务器只读核查给出的 5 条权重路径完全对应。
2. **5 路权重在本地全部有副本**，且 **5 个 SHA256 全部与服务器记录的期望哈希逐位一致**（见 §3.2）。冠军在本机是**可重建**的，不依赖服务器在线。
3. **冠军 val 逐类数字（EX 45.7715 / HE 43.7583 / SE 55.1807 / MA 25.9672）在本地找到逐字命中**：
   `execution_20260919/champion_val_counts_remote.json`（mIoU `0.42669428454584546` = 42.6694）。**用户给定值 = 该文件口径**，非 test 口径。
4. **冠军 test mIoU 44.2115 的逐类不是用户给的这组**：test 逐类为 EX 53.8611 / HE 44.5131 / SE 49.8164 / MA 28.6555（`experiments/h5_test.json`、`h7_test.json`）。**用户把 val 逐类与 test 总分写在了一起**，两者不是同一次评估。
5. **本地不存在任何概率缓存**：全域 `*.npy/*.npz/*.pt` 命中 0 个。概率缓存只是**代码机制**（`ChampionProbCache` 等），本体在服务器 `/dev/shm` 与 `/root/.../cache/`，未回收、非持久。
6. **仅凭逐类 IoU，不足以支撑"值得做像素级融合缓存"**：U0 在 val 四类全部被冠军严格支配，H0 仅 HE 接近；而逐类 IoU 是标量，**无法反映像素级错误互补性**，且 val n=76 的选择噪声（±0.5~0.8 点）大于任何可见的融合增益。详见 §6。

---

## 2. 冠军完整配置（本地原始证据）

### 2.1 集成成员与融合方式

权威记录：`HRDecoder_FusionAware_20260917/execution_20260919/full_eval_settings.json`（champion 段）、`HRDecoder_审查轮_20260916/audit_runs/audit_20260916_A/experiments/h5_test.json`、`h7_test.json`。

| 成员 | 配置 | 服务器权重路径（锚点） | 本地副本 | 权重 |
|---|---|---|---|---|
| D | `configs/lesion/exp/hrplus_patch_dice.py` | `/root/autodl-tmp/hrplus/D_0913_retry/best_mIoU_iter_2000.pth` | `hrdecoder_plus3/results/D_0913_retry/backup_student_2000.pth` | **3.0** |
| E | `configs/lesion/exp/hrplus_mares.py` | `/root/autodl-tmp/hrplus/results/E_0916/best_mIoU_iter_1500.pth` | `HRDecoder_FusionAware_20260917/assets/E_1500.pth` | **2.0** |
| B | `configs/lesion/exp/hr100_b.py` | `/root/autodl-tmp/hr100/trial_b/best_mIoU_iter_3000.pth` | `hrdecoder_plus3/repo/weights/b_weights.pth` | **1.0** |
| E4 | `configs/lesion/exp/e4_detail_split613.py` | `/root/HRDecoder/work_dirs/exp_e4_detail_split613/best_mIoU_iter_36000.pth` | `autodl100/results/e4_reference_weights.pth` | **1.0** |
| H4 头 | 后融合 MA 校正头 | `/root/autodl-tmp/audit/audit_20260916_A/h4_best.pth` | `HRDecoder_审查轮_20260916/audit_runs/audit_20260916_A/h4_best.pth` | （后处理） |

融合定义（`HRDecoder_MA残差头与集成_20260915-16/03_代码/ops_eval_ens.py`，并被 `execution_20260919/eval_candidates.py` 复用）：

- **概率级加权平均**（非 logit 平均）：`acc += sigmoid(logit_i) * w_i`，再 `avg = acc / Σw_i`；开启 **horizontal-flip TTA**（`flip: true`），每图 4 成员 × 2（原图+翻转）共 8 次前向。
- **类序固定为 通道0=EX, 1=HE, 2=SE, 3=MA**（`tools/contract_common.py` 与旧仓库 `lesion_dataset.py` 一致，无置换）。
- **H4 头**：`P_new_MA = sigmoid( logit(P_fused_MA) + 6·tanh(head_7ch) )`，输入 7 通道 = `[R, G, B, logit(fused MA), logit(student MA), logit(reference MA), (P_student − P_reference)]`。
  结构（`h4_best.pth` 实测 keys）：`net.0 Conv3×3(7→16)` → `net.2 Conv3×3(16→16)` → `net.4 Conv1×1(16→1)`，**共 3,361 参数**（零初始化起步）。
- **训练/评测几何**：输入 2048² 整图；HR 分支 tile=1024 / stride=768（`RUN_PLAN.json: geometry_fixed`）；H4 头在 1024² crop 特征上训练。
- **指标口径（冻结协议）**：4 个 sigmoid 通道、**严格 `p > 0.5`**、全数据集**累计** TP/FP/FN 后逐类算 IoU，mIoU = 四类算术平均，单位百分数；`union==0` 直接 fail-closed。
- **seed**：H4 头训练用 seed=1（`RUN_PLAN.json: seed_initial`）；集成评测本身无随机性。
- **一次性 test**：`eval_candidates.py` 中 `test` 需同时 `--allow-test` 且 `--locked-candidate`，并核对 `locked_candidate.json`（`test_seen_before_lock: false`）。

### 2.2 与文档转述的冲突（以本地文件为准）

| 项 | 文档转述 | 本地实测 | 处置 |
|---|---|---|---|
| H4 头训练步数 | "2000 步" | `h4_best.pth` 内 `step = 1600`；`h4_post_fusion_corrector.json` 的 `best.step = 1600`；`h7_test.json` 的 `heads.MA = 1600` | **实际采用 step 1600**（2000 步跑完但 best@1600） |
| 冠军 val mIoU | 42.6686 / 42.6694 并存 | 两次评估：`p0_report.json` 42.6686、`champion_val_counts_remote.json` 42.6694（相差 0.0008） | 两者均为 val；差异来源本地未定论（见 §4.3） |

---

## 3. 本地权重：实际大小与 SHA256

### 3.1 实测清单（`stat` + `sha256sum`，逐字节）

| 代号 | 本地路径（相对 `D:\dataset\experiment\`） | 实际大小 (bytes) | SHA256 |
|---|---|---|---|
| D | `hrdecoder_plus3/results/D_0913_retry/backup_student_2000.pth` | 263,419,124 | `5fe18717b62d8fc3ab4846a50c046588e2af05abbbfdb3a958f4794018b319ea` |
| B | `hrdecoder_plus3/repo/weights/b_weights.pth` | 263,479,238 | `0b3cd72e740fc5c69242994c12cfd71a49ac892aff37d09c50bb9fb6931a8626` |
| B（副本） | `autodl100/results/trial_b/selected_weights_only.pth` | 263,479,238 | 同上（逐位相同） |
| E | `HRDecoder_FusionAware_20260917/assets/E_1500.pth` | 527,412,743 | `a18ff92b4da24d870554f17856983d4be042386cacbcaba0d4e13f3f248ed5e5` |
| E4 | `autodl100/results/e4_reference_weights.pth` | 263,477,167 | `c5a3d0f0b990d2430e545146745bc809538bbe764979baf54f62b450785268e0` |
| H4 头 | `HRDecoder_审查轮_20260916/audit_runs/audit_20260916_A/h4_best.pth` | 15,447 | `14757e02de49f6f065d4ef82cb8b1c4ee8a6a7f3f66cb1867276993950f80c50` |
| A（非冠军） | `autodl100/results/trial_a/selected_weights_only.pth` | 263,709,987 | `82689516488846d3129f8576fa0533bad954cfd546f8c2187f64ca9c8fb903b0` |
| C（回收） | `HRDecoder_FusionAware_20260917/execution_20260919/remote_recovery_20260920/runs/cs4000_v2/C_step_2000.pth` | 718,620,759 | `119793766067d872ecc8d420cda4f2ed36d359480ee4581f85d3cd37b74f928e` |
| S（回收） | 同上 `S_step_4000.pth` | 718,620,823 | `abb2f078e57fed2444b74f2d81d3aa22049fbc17ea5d7fc4e8fd5f1a94c52179` |

### 3.2 与服务器记录哈希的交叉验证（关键）

`full_eval_settings.json` 的 `expected_sha256` 段记录了服务器侧 5 路基线的期望哈希。逐条比对结果：

| 服务器记录键 | 记录值 | 本地实测 | 一致 |
|---|---|---|---|
| `assets/D_student_2000.pth` | `5fe18717…19ea` | D | ✅ |
| `assets/B_weights.pth` | `0b3cd72e…8626` | B | ✅ |
| `assets/E_1500.pth` | `a18ff92b…ed5e5` | E | ✅ |
| `assets/E4_weights.pth` | `c5a3d0f0…268e0` | E4 | ✅ |
| `assets/H4_best.pth` | `14757e02…0c50` | H4 头 | ✅ |

**5/5 逐位一致。冠军全部输入权重在本机完整可用，可离线重建，无需重新从服务器拉取。**

### 3.3 权重结构（`torch.load` 实测）

| 文件 | 顶层 keys | 参数量 (numel) | 结构判读 |
|---|---|---|---|
| D | `state_dict, meta` | 65,695,585（1877 tensors） | 含 `detail_scale` + `detail_head.*` ⇒ HRPlusRefinement（detail 支路开启），student 纯权重 |
| B | `state_dict, meta` | 65,695,585（1877） | 与 D 同构 |
| E4 | `state_dict, meta` | 65,695,585（1877） | 与 D 同构 |
| E | `meta, state_dict, optimizer` | 131,449,611（3774） | 含 `ma_refiner.*` ⇒ HRPlusMAResidual（student+reference+MA refiner），且**带 optimizer**（故 527 MB） |
| H4 头 | `head, step` | 头 3,361（step=1600） | `net.0/2/4` 三层，与 §2.1 描述一致 |

说明：D/B/E4 约 263 MB ≈ 65.7 M × 4 B（fp32 纯权重）；E 约 527 MB 因额外含 reference 分支与 optimizer 状态。

---

## 4. val / test 原始结果（逐类，原文数值）

### 4.1 冠军 val（**用户给定值的出处**）

`HRDecoder_FusionAware_20260917/execution_20260919/champion_val_counts_remote.json`：

```
split = val,  n = 76
mIoU  = 0.42669428454584546   (42.6694)
EX = 0.45771525507464833      (45.7715)
HE = 0.43758256131196327      (43.7583)
SE = 0.5518070087629399       (55.1807)
MA = 0.25967231303383037      (25.9672)
```

✅ **四项全部与用户给定值逐字一致**。同值另见 `locked_candidate.json: champion_miou` 与 `runs/evalcand_val/*/evalcand_champion_counts.json`（4 个候选目录内冠军行完全相同）。

### 4.2 冠军 test

`HRDecoder_审查轮_20260916/audit_runs/audit_20260916_A/experiments/h7_test.json`（同 `h5_test.json`）：

```
split = test, n = 227
final.mIoU = 0.442115035614162   (44.2115)
EX = 0.5386111922725695          (53.8611)
HE = 0.4451296131641197          (44.5131)
SE = 0.4981614515880462          (49.8164)
MA = 0.2865578854319127          (28.6555)
fused(无 H4 头).mIoU = 0.43228924245323747 ; fused.MA = 0.2472547127882145
非 MA 三类在加头前后 bit-identical = true
```

**对照：用户给出的 EX 45.7715 / HE 43.7583 / SE 55.1807 / MA 25.9672 是 val 口径，不是 test 口径。** 44.2115（test 总分）与 val 逐类混写会造成口径错配，后续引用须分开。

### 4.3 其他相关原始结果（用于 §6 判断）

| 方案 | split | mIoU | EX | HE | SE | MA | 出处 |
|---|---|---|---|---|---|---|---|
| BASE_0（集成，无头） | val | 41.7963 | 45.7705 | 43.7552 | 55.1806 | **22.4789** | `diagnostics/miou4_report.json` |
| 冠军（BASE_0 + H4） | val | 42.6686 | 45.7705 | 43.7552 | 55.1806 | **25.9681** | `p0_report.json` |
| 冠军（同机复现） | val | **42.6694** | 45.7715 | 43.7583 | 55.1807 | 25.9672 | `champion_val_counts_remote.json` |
| 冠军 | test | **44.2115** | 53.8611 | 44.5131 | 49.8164 | 28.6555 | `h7_test.json` |
| student-only（D） | val | 39.3351 | 40.2364 | 43.0491 | 51.2710 | 22.7840 | `diagnostics/val_blend_decomposition.json` |
| reference-only（B） | val | 39.2050 | 44.2479 | 41.5482 | 51.6243 | 19.3999 | 同上 |
| **新 U0 @12000（本机）** | val | **38.4464** | **43.6760** | **38.4266** | **50.9362** | **20.7468** | 本工程 `runs/eval_u12000.json` + `completion_summary.json` |
| 新 U0 @6000 | val | 35.0006 | 40.9355 | 36.3871 | 42.8032 | 19.8764 | `runs/eval_u6000.json` |
| **H0@6000（用户给定）** | val? | (36.9225) | **41.88** | **43.52** | **41.45** | **20.84** | **本地全域未找到原始出处** |

关于 val 两个数（42.6686 vs 42.6694）的差异 0.0008：
两次评估的**非 MA 三类计数几乎相同但非逐位相同**（如 `p0_report` 的 EX counts `[181593,105944,109210]` vs `champion_val_counts_remote` 的 `[181588,105924,109215]`）。
`ROUND_CS_FINAL_REPORT_20260920.md` 自述"主审查修复了**冠军缓存 fp16 改变阈值附近像素**"，且 `eval_candidates.py` 头注明确要求"概率保持 fp32，使缓存不改变严格 0.5 阈值附近的判定"。**因此 42.6694 是 fp32 修正后的数、42.6686 疑为更早（缓存精度受影响）的数**——本地无法百分百确证，但方向与时间线（p0=S09-18、remote=S09-19）一致。**引用冠军 val 时以 42.6694 / 逐类 45.7715-43.7583-55.1807-25.9672 为准。**

### 4.4 H0@6000 的证据状态

用户给定的 `41.88 / 43.52 / 41.45 / 20.84`（均值 36.9225）在 **`D:\dataset` 全域、`C:\Users\GuanXingRen\WorkBuddy`、`C:\Users\GuanXingRen\AppData\experment`、本工程** 均 **未找到任何逐字或近似出处**（含 `0.4188/0.4352/0.4145/0.2084` 等变形检索）。
本工程 `configs/H0_hrdecoder_ddr_only.yaml` 的 H0 臂状态为 `locked-intent`，`docs/READINESS_RECHECK.json` 中 H0 仍有 6 项 `GPU_REQUIRED` 占位未回填 ⇒ **H0 尚未在本机训练**。结论：**该组数字属于外部/口头提供，本地无原始证据，仅登记、不做进一步推断。**

---

## 5. 概率缓存：机制存在，本地不存在

### 5.1 本地实际产物

- **全域 `*.npy / *.npz / *.pt / *.ckpt` 检索：0 个**（三个授权目录 + 本工程）。
- 本地**唯一**的"类概率"产物是**逐图计数 JSON**，不是概率图：
  `champion_val_counts_remote.json`、`runs/evalcand_*/*_counts.json`（每图每类 `[TP,FP,FN]` + `global_counts`）。
- 回收清单 `remote_recovery_20260920/recovery_manifest.json` 共 40 项，**全部为脚本/日志/计数/权重**，**无任何概率缓存文件**；`backups/finalize_20260920_005521.tar.gz`（56 KB，40 条目）同样不含概率数据。

### 5.2 缓存机制（代码存在，本体在服务器）

| 位置 | 内容 | 缓存对象 | 持久性 |
|---|---|---|---|
| `execution_20260919/eval_candidates.py:165` `class ChampionProbCache` | 冠军概率磁盘缓存：**per-image float32 npy**、身份哈希 = 冠军规格 + 全部权重/配置哈希 + 评测配置文本、每次访问 sha256 校验、不完整 fail-closed | 冠军 4 通道概率 | 服务器 |
| `full_eval_settings.json: cache_dir` | `/root/autodl-tmp/hr_exec_20260919/cache/reference` | **reference 特征缓存** | 服务器 |
| `HRDecoder_审查轮_20260916/tools/build_cache.py` | 输出 `/dev/shm/hrdec_cache{,_val}/base0\|ps\|pr/%04d.npy`，**仅 MA 单通道 float32 @ 原始分辨率**（train 454 / val 76） | BASE_0/student/reference 的 MA 概率 | **内存盘，非持久** |
| 文档自述 | "`/dev/shm` 常驻概率缓存 31.5 G"；test 缓存在 `hrdec_cache_test` | — | 非持久 |

**结论：冠军的"概率缓存"是一项服务器机制，本机 0 个缓存文件；缓存内容未随本轮回收落盘。** 若后续要做像素级融合，需从零重建缓存（或重算概率）。

---

## 6. 逐类 IoU 对比与"像素级融合缓存"价值判断

### 6.1 逐类对比（**全部为 val，n=76；严禁用 test 选参**）

| 方案 | mIoU(val) | EX | HE | SE | MA |
|---|---:|---:|---:|---:|---:|
| 冠军（BASE_0+H4） | 42.6694 | 45.7715 | 43.7583 | 55.1807 | 25.9672 |
| 新 U0 @12000 | 38.4464 | 43.6760 | 38.4266 | 50.9362 | 20.7468 |
| Δ(U0 − 冠军) | **−4.2230** | −2.0955 | **−5.3317** | −4.2445 | **−5.2204** |
| H0@6000（无本地证据） | (36.9225) | 41.88 | 43.52 | 41.45 | 20.84 |
| Δ(H0 − 冠军) | (−5.7469) | −3.8915 | **−0.2383** | **−13.7307** | −5.1272 |

### 6.2 判据与推理

**(a) 逐类 IoU 只能给出"水平"，给不出"互补性"。**
融合增益来自"**一方错、另一方对**"的像素级重叠，其充分性证据是**逐像素误差互补度 / oracle 上界**，不是逐类 IoU。两个逐类 IoU 完全相同的模型，一个可能近乎冗余、另一个可能高度互补。因此**仅凭逐类 IoU 在原理上就不足以决定是否值得付出像素级融合缓存的代价**。

**(b) 即便退一步只看"水平"，也没有正向信号。**
- **U0 在 val 四类全部劣于冠军**（−2.10 / −5.33 / −4.24 / −5.22），且异构（ConvNeXt-S + U-Net）。一个在每一类都被支配的成员，其可贡献的"对的像素"上界有限；**没有出现任何"U0 独占优势"的类别**。
- **H0 只在 HE 接近冠军**（43.52 vs 43.76，差 0.24 点），EX/SE/MA 全面落后，其中 SE 落后 **13.73 点**。且 H0 与冠军同属 HRDecoder 族（与 D/E/B/E4 同源），**误差相关性高**，互补潜力先天偏低。
- 冠军现有的增益来源已很清楚：**全部来自 H4 MA 头**（BASE_0 val 41.7963 → 42.6686，MA 22.4789 → 25.9672，非 MA 三类 bit-identical）。这说明该体系当前的瓶颈是 **MA**，而不是"再多一路弱成员做像素级平均"。

**(c) 代价明确且不小（4 通道、2048² 整图、fp32）**

| 对象 | 单图 | val(76) | train(454) | test(227) |
|---|---:|---:|---:|---:|
| 4ch fp32 | 64 MiB | 4.75 GiB | 28.4 GiB | 14.2 GiB |
| 4ch fp16（**不推荐**，会动阈值附近像素） | 32 MiB | 2.4 GiB | 14.2 GiB | 7.1 GiB |
| 仅 MA 单通道 fp32（旧 `build_cache.py` 口径） | 16 MiB | 1.19 GiB | 7.1 GiB | 3.6 GiB |

叠加磁盘/内存压力与"每次权重或配置变动即整体失效、需重算"的维护成本。

**(d) val 选择噪声大于任何可见增益。**
n=76 的经验选择带宽：`ROUND_CS` 报告的配对 bootstrap 95% 区间为 **[−0.5454, +0.8119]**（跨零）；`审查轮` 的 H4−BASE_0 CI 为 [+0.5718, +1.1815]。任何 < ~0.8 点的融合增益都落在噪声内。**先例更直接**：`ROUND_CS` 中在 val 锁定的 `0.75×冠军 + 0.25×C@2000`（val +0.1963）在 test 上 **−0.4052**。即在 val 上"看起来改善"的融合，很可能只是选择偏差。

### 6.3 结论

> **仅凭逐类 IoU，判定为：现在不值得为像素级融合缓存付出代价。**
>
> 理由：① 逐类 IoU 在原理上无法证明像素级互补性；② 退一步看水平，U0 被全类支配、H0 仅 HE 接近且同族高相关，均无正向信号；③ 代价为 GiB 级（4ch fp32 全库约 47 GiB）且随任一权重/配置变动整体失效；④ val n=76 的选择噪声 ≥ 任何可见增益，且已有"val 选点、test 回退"的直接先例。

### 6.4 若仍要推进：条件式路径（不满足则不做）

1. **先做零持久化、val-only 的逐像素 oracle 上界**：流式跑一遍候选与冠军的逐像素概率，统计"逐像素取优"能达到的 val mIoU 上界（**不落盘、不用 test**）。仅当 oracle 上界增量 **显著大于 val 噪声带宽（≥ ~0.8–1.0 点）** 才进入下一步。
2. 建缓存前必须**预注册**：候选集、融合形式、权重网格、选择规则、判据，**全部在 val 上锁定**（`locked_candidate.json` 模式：`test_seen_before_lock: false`）。
3. 缓存精度必须 **fp32**（`eval_candidates.py` 已明确 fp16 会改变严格 0.5 阈值附近像素）；缓存身份须绑定**全部权重/配置哈希**并在每次读取时 sha256 校验、不完整 fail-closed。
4. 最优先的对照不是"再多一路弱 student"，而是**换信息来源或监督质量**（与 `ROUND_CS_FINAL_REPORT` 的研究判断一致）：当前瓶颈在 MA，冠军的增量已全部来自 MA 校正头。
5. test 只允许在锁死之后**跑一次**；任何"看 test 再调权重/阈值"都禁止。

---

## 7. 未决事项与风险

| # | 事项 | 状态 |
|---|---|---|
| 1 | H0@6000（41.88/43.52/41.45/20.84）本地零证据 | **需外部来源回填**；不得据此做任何推断 |
| 2 | 冠军 val 42.6686 与 42.6694 的 0.0008 差异 | 疑与 fp16 缓存修正有关，本地未定论；引用以 42.6694 为准 |
| 3 | H4 头步数：文档"2000"vs 实测 `step=1600` | 以 **1600**（best）为准 |
| 4 | E 权重含 optimizer，527 MB；D/B/E4 为纯权重 | 复现时注意加载方式（`load: full` vs `student_only`） |
| 5 | 概率缓存本体是否仍在服务器 | 未回收；`/dev/shm` 非持久，重启即失 |
| 6 | D 的 `D_full_2000.pth.part`（617 MB 分片） | 非完整文件，登记备查，勿当权重用 |

---

## 8. 证据索引（关键文件绝对路径）

- 冠军配置与哈希期望：`D:\dataset\experiment\HRDecoder_FusionAware_20260917\execution_20260919\full_eval_settings.json`
- 冠军 val 逐类（**用户给定值出处**）：`…\execution_20260919\champion_val_counts_remote.json`
- 冠军 test 逐类：`…\HRDecoder_审查轮_20260916\audit_runs\audit_20260916_A\experiments\h7_test.json`（及 `h5_test.json`）
- 冠军 val 早期评估 + 逐类 profile：`…\HRDecoder_FusionAware_20260917\pulled\final_20260918T184943\p0_report.json`
- BASE_0/H4 val 对照与 bootstrap：`…\audit_20260916_A\diagnostics\miou4_report.json`
- student/reference/blend val 分解：`…\audit_20260916_A\diagnostics\val_blend_decomposition.json`
- 本轮负结果报告：`…\HRDecoder_FusionAware_20260917\execution_20260919\ROUND_CS_FINAL_REPORT_20260920.md`
- 锁定记录：`…\execution_20260919\locked_candidate.json`
- 概率缓存机制代码：`…\execution_20260919\eval_candidates.py`（`ChampionProbCache`）、`…\HRDecoder_审查轮_20260916\tools\build_cache.py`
- 新 U0 本机结果：`D:\dataset\experiment\HRDecoder_FullCompetition_20260920\runs\eval_u12000.json`、`completion_summary.json`
- 通道/接口契约：`D:\dataset\experiment\HRDecoder_FullCompetition_20260920\docs\ASSET_AND_INTERFACE_AUDIT.md`

> 本盘点全程只读：未训练、未删除文件、未联网、未修改 `D:\HRDecoder` 及任何旧实验目录。
