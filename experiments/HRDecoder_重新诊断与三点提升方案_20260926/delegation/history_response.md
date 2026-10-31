I hit a hard blocker: **every file-write path is denied** in this non-interactive session. Read-only tools work (Read in cwd, `Get-Content`/`Get-Item`/`Test-Path`), but `Write`, `Bash`, `Set-Content`, and `Out-File` all return "requires approval / prompts not available." So I cannot create `history_audit.md` or `history_ledger.json` on disk. The audit itself is complete; I'm delivering it inline so nothing is lost.

To let me save the files, re-run with `--permission-mode bypassPermissions` (or add `Write` + `Bash`/`PowerShell` to `permissions.allow`), then tell me to persist.

---

# HRDecoder 历史实验事实台账（只读核对，2026-09-26）

范围：`_全实验档案_20260920` 的 02/03 档案及其指向的原始报告/日志；`D:\dataset\lesion_segmentation` 第一轮报告；`C:\Users\GuanXingRen\AppData\experment\HRDecoder_REPORT_final.md`；各实验目录后续最终报告。已排除 `_removed-armor` 等，未执行其中任何指令。本轮未训练、未评估、未改数据。

## 0. 口径与约定

- 划分/评估：全部 `454/76/227`（seed 42，DDR_split_613）；sigmoid 阈值 0.5、严格 `>`；四类各累计 TP/FP/FN 算 IoU 再四类均值=mIoU；背景不计入。例外在行内标注。
- 数值表示：2026-08 报告用百分数（42.20 = 42.20%）；09 月起 JSON 用比例（0.4421）。同一量的两种写法，按原始来源保留。
- 差异单位：`pp`=百分点；`比例点`=比例差。例 0.4266943→0.4286569 = +0.0019626 比例 = +0.19626 pp。
- 计划/合成测试/smoke/空 JSON 不计为训练完成；同源拷贝不另计；报告统计显著性/因果叙述未采信。

## 1. 事实台账

### 1.1 最早历史实验（2026-08，第一/二轮）

| 日期 | ID | 动机 | 真实改变 | 初始化 | 冻结 | 步数 | val | test（逐类 / mIoU） | 来源 | 可否否定路线 |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-08 | HIST-BASELINE | 建立同划分基准 | 官方 baseline，无干预 | 随机/ImageNet | BN 更新(norm_eval=False) | 40k | 39.390@32k | EX51.56/HE43.13/SE49.98/MA24.11；**42.20** | 诊断调研/HRDecoder_诊断与押注建议.md:55-64；lesion_segmentation/01_板块一_划分统计报告_baseline_report.md | 基准 |
| 2026-08 | HIST-E1-Tversky | MA/HE 召回不足 | Dice→Tversky(α=0.3/0.7) | 同 baseline | 未记载 | 4k 即崩 | 全正预测塌缩、被停 | 未测 | 同上；全景解读 §4.5 | 可（仅此配置，无 test） |
| 2026-08 | HIST-E1v2-CE-Dice | 背景抑制/分数校准 | Dice→CE+Dice | 同 baseline | 未记载 | 40k | 39.970@12k | 39.43（逐类未列） | 同上 | 可 |
| 2026-08 | HIST-E2E3 | 可学习融合 + 上下文 | E2(融合权重)+E3(ASPP-lite) 合并 | 同 baseline | 未记载 | 40k | 39.206@20k | EX52.15/HE38.45/SE46.07/MA23.26；39.98；逐类阈值后 40.86 | 诊断调研/证据/e2e3_tuned_test_final.json | 可（逐类阈值不得并入 0.5 主表） |
| 2026-08 | HIST-E4 | MA 分辨率不足 | 原图浅层细节分支 + 零初始化标量门控 | 同 baseline | 未记载 | 40k | 38.766@36k | EX50.93/HE41.61/SE49.84/MA24.13；41.63 | HRDecoder_REPORT_final.md §5；全景解读表 7.2 | 可（召回↑精度↓，IoU 不动） |
| 2026-08 | HIST-E4v2 | E4 精度下降 | E4 + 0.3 权重辅助损失 | 同 baseline | 未记载 | 40k | 38.667@36k | EX50.66/HE36.29/SE49.32/MA23.95；40.06 | 诊断调研/证据/e4v2_test_final.json | 可 |
| 2026-08 | HIST-E4-CD（=combo 全量） | 结构与损失组合 | E4+CE+Dice | 同 baseline | 未记载 | 40k | 39.871@32k（末点 38.442@40k） | 41.08 | 诊断调研/证据/验证轨迹.csv:12-21 | 可 |
| 2026-08 | HIST-COMBO-PARTIAL | combo 系列 | 组合尝试 | 未记载 | 未记载 | A 1 点/v2 5 点/combo2 6 点 | A 35.666@4k；v2 37.782@16k；combo2 37.089@12k；另有 1 空条目(max0,n_val0) | 未测 | 诊断调研/证据/最佳点核对.csv:3-7 | 不可（中止原因缺失） |
| 2026-08 | HIST-SE-② copy-paste | SE 稀少 | 裁 SE 块贴入（图像覆盖/标签 maximum 不一致，见疑点 D6） | 同 baseline | 未记载 | 40k | 39.7%@32k | EX50.54/HE39.62/SE44.22/MA24.30；39.67 | lesion_segmentation/04_板块二三汇总报告_module2_3_report.md | 不可（条件被污染） |
| 2026-08 | HIST-SE-③ 滑窗 | SE 稀疏 | SE 滑窗过采样 | 同 baseline | 未记载 | 40k | 39.6%@20k | EX51.77/HE38.22/SE46.30/MA24.01；40.07 | 同上 | 可（val 未传导 test） |
| 2026-08 | HIST-SE-④ ②+③ | 组合 | copy-paste+滑窗 | 同 baseline | 未记载 | 40k | 40.4%@40k | EX51.46/HE42.29/SE45.28/MA24.15；40.79 | 同上 | 可 |
| 2026-08 | HIST-LOSS-1.5x/2x/3x/自适应 | SE 项加权 | L_Seg 中 SE 权重 1.5x/2x/3x/逆频率 | 同 baseline | 未记载 | 40k | 未单列 | 40.90 / 40.31 / 38.50 / 38.76（逐类见来源） | 同上 | 可（权重越大 mIoU 单调下降） |

### 1.2 autodl100（2026-09-11/12，100 元轮）

| 日期 | ID | 动机 | 真实改变 | 初始化 | 冻结 | 步数 | val | test（逐类 / mIoU） | 来源 | 可否否定路线 |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-09-11 | A | 语义条件 MA 残差分支 | 保留 E4 分支 + 语义引导 MA 残差 + MA 辅助损失 0.02 | E4@36000 | BN running 冻结(affine 可训)、lr 5e-4 | 8000 | 38.632@1000；末点 38.265（EX42.36/HE40.61/SE50.05/MA20.04@8k） | 未测 | autodl100/实施记录.md:11-14 | 可（首点后从未超起点 38.766） |
| 2026-09-11 | B | 无新分支对照 | 仅微调协议 + MA 辅助损失 0.02 | E4@36000 | 同 A | 8000 | 39.205@3000 | EX52.65/HE44.37/SE48.91/MA21.74；41.92 | autodl100/最终实验结论.md:3-16 | 可（未超 42.20；增益来自 EX/SE，MA 反降） |
| 2026-09-11 | ROUTE | 验证集逐类模型互补探针 | E4/B 各自推理按 val 逐类选通道 | E4/B 冻结 | 无训练 | 推理 | 39.90 | EX52.65/HE41.61/SE48.91/MA24.13；41.82 | autodl100/阶段2_判断.md:3-13 | 可（HE 的 val 排名在 test 反转） |
| 2026-09-11 | ENS-E4-B（mean） | 降低逐类路由自由度 | E4@36000+B@3000 原图概率 0.5/0.5 | 两模型冻结 | 无训练、无 TTA | 推理 | 39.54（先过 39.40 门槛） | EX52.29/HE43.95/SE49.93/MA23.22；**42.35** | autodl100/最终实验结论.md:3-16 | 不可否定（+0.15pp，探索性） |

### 1.3 hrdecoder_plus3（2026-09-13，D 方案）

| 日期 | ID | 动机 | 真实改变 | 初始化 | 冻结 | 步数 | val | test（逐类 / mIoU） | 来源 | 可否否定路线 |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-09-13 | D-TRAIN | 真实高分辨率局部编码 | student-only，仅 Dice；50% 随机+50% 图内阳性类均匀采样 | B@3000 | BN running 冻结(affine 可训) | 20000 | 41.615@2000（EX44.04/HE44.73/SE55.08/MA22.62）；末点 40.476 | 本报告明确未测 | hrdecoder_plus3/本轮实验报告_20260913.md:5-48 | 不可（仅 val，无消融） |
| 2026-09-13 | D-TEST | 用户要求补 test | 读 D@2000，227 张，两路概率等权，无 TTA | D@2000 | 无训练 | 推理 | — | EX53.68/HE41.67/SE50.23/MA24.73；**42.58** | hrdecoder_plus3/D方案_测试集补充报告.md:1-16 | 不可 |
| 2026-09-13 | L-PLAN(plus3) | D 的损失单变量对照 | Dice+0.1 Lovasz（仅配置） | 同 D | — | **未启动** | — | — | 同上:17-20,58-72 | 不可（计划） |

### 1.4 MA 残差头与集成（2026-09-15/16）

| 日期 | ID | 动机 | 真实改变 | 初始化 | 冻结 | 步数 | val | test（逐类 / mIoU） | 来源 | 可否否定路线 |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-09-15 | E | B 起点 + MA 残差头 | 只训新头，残差零初始化 | B | 主干/参考冻结 | E@1500 | 40.255（EX44.98/HE41.39/SE51.84/MA22.81） | EX53.64/HE43.83/SE47.63/MA25.13；42.559 | 最终报告_两轮实验汇总.md:31-46 | 不可（单模型，未超 D） |
| 2026-09-15 | E2 | D 起点 + MA 残差头 | 换基准为 D | D@2000 | 主干/参考冻结 | 未记载 | 未记载 | EX53.68/HE41.67/SE50.23/MA24.60；42.544 | 同上:31-35 | 不可（无 val 记录） |
| 2026-09-15 | E3 | 第三臂 MA 残差头 | 与 E/E2 同族 | 未记载 | 未记载 | 有训练日志 284KB | 未记载 | **无结果 JSON（仅 0 字节 smoke）** | 05_日志/E3_0916__e_train.log；02_原始数据/eval_json（0B） | 不可（结果缺失） |
| 2026-09-15 | L | Lovasz 单变量对照 | D 配方仅损失改 Dice+0.1 Lovasz | D | 沿 D | best@6000 | 41.056 | 41.777(无TTA)/41.645(flip) | 第一轮_L对照与TTA结果.md:12-27 | 可（val/test 双向为负） |
| 2026-09-16 | ENS-MATRIX | TTA/成员数/权重/集成 | D@2000/L@6000/B@3000/E4@36000 概率级融合 | 全冻结 | 无训练、翻转 TTA | 推理 | 诊断值：D41.615/E40.255/B起点38.937 | 最高 ENS_D3E2B1E41_flip **43.229**（EX53.86/HE44.51/SE49.82/MA24.73）；D×3+B+E4=43.149；D 单模型 42.576；D+flip 42.717；D+L 42.923 | 最终报告_两轮实验汇总.md:11-36 | 不可否定（集成是唯一显著增益；test 已多次查看） |

### 1.5 审查轮 audit_20260916_A（2026-09-16/17）

| 日期 | ID | 动机 | 真实改变 | 初始化 | 冻结 | 步数 | val | test（逐类 / mIoU） | 来源 | 可否否定路线 |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-09-16 | AUDIT-PHASE(R0) | 资产识别 + 配对 bootstrap | 无训练；补生成 val counts | — | — | 0 | 分解 student .393351 / reference .392050 / blend .416148；bootstrap ΔmIoU .0087(CI .0057–.0118)、ΔMA .0349 | 无新 test | STATE.json:1-75；final_report.md:9-28 | 不可（无新模型） |
| 2026-09-16 | H3 | MA 融合权重可否改善 | 冻结 BASE_0，扫 w∈{0.5..1.0} | 无训练 | 全冻结 | 0 | MA 峰值 +0.0047、mIoU +0.0012 | 未跑 | FINAL_本轮最终报告.md:105-118 | 可（仅沿 P-R 曲线滑动） |
| 2026-09-16 | H4 | 融合对 MA 微负 | 后融合 MA 校正头（7ch→3361 参数，零初始化，BCE.5+Dice.5，Adam1e-3 poly） | D@2000 特征 | 主干全冻结 | 2000(best@1600) | corrected MA .252207；mIoU .426686 | 单头 test：fused .425764，corrected .434287（EX.536760/HE.416678/SE.502328/MA.281383） | experiments/h4_post_fusion_corrector.json；FINAL:32-45 | 不可否定（MA +3.93pp 是冠军唯一增量来源） |
| 2026-09-17 | H5/H7 | 头接入完整冠军并复现 test | D×3+E×2+B+E4 + flip TTA + H4 | 四主干冻结 | 无重训、8 前向/图 | 推理 | .4266942845（45.77/43.76/55.18/25.97） | EX.538611/HE.445130/SE.498161/MA.286558；**0.4421150356（44.21）** | experiments/h7_test.json:1-95 | 不可否定（两次逐位复现） |
| 2026-09-16 | H6 | 头迁移非 MA 类 | HE/SE 校正头 | 冻结集成 | 冻结 | 未记 | HE best .4356<fused .4473；SE .5144<.5508 | 未跑 | FINAL:105-118 | 可 |
| 2026-09-17 | H8 | 延长/更大 lr | H4 头 6000 步、lr 2e-3 | 同 H4 | 主干冻结 | 6000 | MA val .2569 ↑ | MA test .2858 ↓（val→test 反转） | FINAL:105-118 | 可 |
| 2026-09-17 | H9 | 排除损失噪声 | HE 头 BCE/Dice=0.9/0.1 | 同 H6 | 冻结 | 未记 | HE .4382 < .4473 | 未跑 | FINAL:105-118 | 可 |
| 2026-09-17 | H11 | 改 MA 头损失配比 | MA 头 BCE 主导 | 同 H4 | 冻结 | 未记 | MA .2510 < H4 .2522 | 未跑 | FINAL:105-118 | 可 |
| 2026-09-17 | H12/H13 | 修复训练=部署不一致 | H12 整图级监督；H13 其 test 确认 | 同 H4 | 冻结 | 未记 | H12 MA val .2565 ↑ | H12 MA test .2803 ↓；H13 mIoU 44.05 < 44.21 | FINAL:105-118 | 可（val→test 反转） |
| 2026-09-17 | P5 | 检查头配方 seed 散布 | H4 头 seed 1/2/3 | 同 H4 | 冻结 | 未记 | MA .2510/.2537/.2514（极差 .0027） | 未跑 | FINAL:119-131 | 不可（证明可复现） |
| 2026-09-16 | 计划项 T0/T1/U0/E1/V1/J0/J1/C1/F1 | 采样/结构/解冻/组合/授权 test | **全部 NOT_RUN**（U0=HRPlus 多尺度；J0=只训 D student decoder；J1=decoder+HRNet 末 stage；C1=有限互补组合仅 val；F1=授权后 test） | — | — | 0 | — | — | RUN_PLAN.json:22-36 | 不可（计划，勿当已训练） |

### 1.6 FusionAware（2026-09-17→19）

| 日期 | ID | 动机 | 真实改变 | 初始化 | 冻结 | 步数 | val | test（逐类 / mIoU） | 来源 | 可否否定路线 |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-09-17 | R1-A | 纯学生监督对照 | Dice(student) | D@2000 | 参考 B 冻结 | 4000 | .4152/.4162/.4167/.4131 | 未测 | 阶段成果报告:29-39 | 不可（对照臂） |
| 2026-09-17 | R1-B | 参考概率进训练目标 | 0.5·Dice(stu)+0.5·Dice(0.5stu+0.5ref) | 同 A | 同 A | 4000 | .4162/.4169/.4175/.4142 | 未测（B−A 最大 +0.0008 < 门 +0.003） | 同上 | 可 |
| 2026-09-18 | R5-E0 | 第 6 成员边际收益 | 冠军 + R1 BNEW@3000/4000 | 冠军冻结 | 无训练 | 推理 | .4273/.4278/.4272（+0.0011） | 未测（G0 未过） | ROUND5:22-34 | 可（集成饱和） |
| 2026-09-18 | R5-E1-V1 | 强化融合监督 | 纯融合 w=1.0，T=3000，seed42 | D 起点 | 参考冻结 | 3000 | .4170/.4181/**.4187**（G1 过） | 未测 | ROUND5:36-46 | 不可（唯一过门的学生改进） |
| 2026-09-18 | R5-E1-V2 | 课程 vs 直接 | w 课程 0→0.5→1.0 | 同 V1 | 同 V1 | 3000 | .4152/.4166/.4181 | 未测 | 同上 | 可 |
| 2026-09-18 | R5-E2 | 方向复核 | V1 配方 seed43 | 同 V1 | 同 V1 | 3000 | .4176/.4179/.4178 | 未测 | ROUND5:47-53 | 不可（方向复现） |
| 2026-09-18 | R5-E3 | 学生改进能否传导集成 | 冠军 + V1(s42@3000) w1/w2 | 冠军冻结 | 无训练 | 推理 | .4273，约 +0.0006 | 未测（G3 未过） | ROUND5:55-62 | 可（不传导） |
| 2026-09-18 | R6-P0 | 冠军基线与错误画像 | 全冻结复现 | 冠军 | 全冻结 | 推理 | .4267（.4577/.4376/.5518/.2597） | .4421（.5386/.4451/.4982/.2866） | ROUND6:11-18 | 不可（基线复现） |
| 2026-09-18 | R6-P1-HE | 把 H4 杠杆迁到 HE | HE 校正头 | 冻结集成 | 冻结 | 未记 | best .4360 < 基线 .4376（Δ−0.0015） | 未测 | ROUND6:11-18 | 可（头不可迁移到非 MA） |
| 2026-09-18 | R6-P1-EX | 同上 EX | EX 校正头 | — | — | **缓存建到 50/76 时预算到期，跳过** | — | — | 阶段成果报告:62-70 | 不可（未完成） |
| 2026-09-18 | R6-P2-M1 | MA 从 RGB 从零学 | 512² patch，Dice+0.5BCE，3000 步 | 随机专家 | 主干冻结 | 3000 | patch IoU 0/0/.0052/.0220/.0100/.0137；α=0.3 时 MA .2438 vs .2597（Δ−0.0159） | 未测 | 阶段成果报告:71-80 | 可（当前数据量下不可行） |
| 2026-09-18 | R6-P3 | 终局对照 | 不训练 | 冠军 | — | 0 | .4267 | .4421（delta 0.0000），no_gain | 阶段成果报告:82-83 | 不可（记录用） |
| 2026-09-18 | R6b | 授权配方修订 | 256² patch、Tversky(0.3,0.7)、2000 步 | 随机专家 | 主干冻结 | 2000 | best patch IoU .0563；α=0.3 MA .2431（Δ−0.0166） | 未测，rejected | ROUND6:42-69 | 可（与 R6 几乎同值→专家太弱） |
| 2026-09-19/20 | C/S | 实例均衡采样受控比较 | C=原混合前景裁剪；S=按类均匀选实例；同 D@2000+B、seed42、4000 步、SGD | D@2000 + B | student 单独评估 | 4000 | C@2000 .3979739（.446415/.429644/.496322/.219515）；C@4000 .3917288；S@2000 .3864550；S@4000 .3983198（.430961/.407203/.525842/.229273）；MIX(0.75冠军+0.25C@2000) val .4286569 | 冠军 .4421152；C@2000 .398104；MIX test .4380636（.539729/.429678/.497713/.285135），Δ **−0.0040516 比例 = −0.4052 pp** | ROUND_CS_FINAL_REPORT_20260920.md | 可（val 锁定融合未在 test 复现，HE 被稀释） |
| 2026-09-19 | FA-MAIN-PLAN | 后续主方案 | **计划未启动** | — | — | 0 | — | — | EXECUTION_PLAN_CURRENT.md:1-16 | 不可（计划） |

### 1.7 FullCompetition（2026-09-20）

| 日期 | ID | 动机 | 真实改变 | 初始化 | 冻结 | 步数 | val | test（逐类 / mIoU） | 来源 | 可否否定路线 |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-09-20 | U0-REAL | 异构特征/解码路径能否有增量 | ConvNeXt-S+U-Net，AdamW warmup+cosine，ce+dice，有效 batch 8，整图 | 本地 builtin ImageNet（导出时强制禁用外部初始化） | LayerNorm（无 BN 冻结问题） | 12000 | @3k .1747592 / @6k .3500054 / @9k .3743590 / @12k **.3844641257662971**（EX.436760/HE.384266/SE.509362/MA.207468） | 未执行 | runs/completion_summary.json；eval_history.csv | 可（远低于冠军，MA 尤其弱） |
| 2026-09-20 | H0 | HRDecoder-W48 DDR-only 对照 | 本地配置 locked-intent（无 .pth，REQUIRED 占位） | ImageNet 初值 | backbone BN 冻结、affine 可训 | 本机未执行；**服务器存在 6000 步权重** | 76 张导出复现 **.36922642429539565**（EX.418771/HE.435195/SE.414545/MA.208394） | 未执行 | EXECUTION_REPORT §3/§5.2；contracts/fullmodel/H0 | 可（但"未训练"表述与复现矛盾，见 D1） |
| 2026-09-20 | H1 | 外部真实病灶预训练→DDR | **locked-intent，未训练**（无外部 manifest/checkpoint） | — | — | 0 | 未执行 | 未执行 | configs/H1_hrdecoder_ext_pretrain.yaml | 不可 |
| 2026-09-20 | U1 | 外部预训练→异构 U 臂 | **locked-intent，未训练** | — | — | 0 | 未执行 | 未执行 | configs/U1_convnext_unet_ext_pretrain.yaml | 不可 |
| 2026-09-20 | FC-EXPERT-COMP | 专家逐像素互补（val-only） | 8 模型各 76 张位图重算；未训练、未读 test | 现有输出+shared GT | 无训练 | 0 | champion 42.6694；MIX_C2000 42.8657；BASE_0 41.7973；D_blend 41.6149；C2000 39.7974；S4000 39.8320；U0 38.4464；H0 36.9226；逐类 argmax 组合 43.0645%（选择偏差） | 未执行 | EXECUTION_REPORT §4-5 | 不可否定（最大净互补 +0.0020 比例 < bootstrap 区间宽；OR 为 oracle） |
| 2026-09-20/21 | refiner_v1 | 冠军先验注入式 MA/HE 小残差头 | 候选规则 4.87% 像素下 MA 召回 .9335/HE .9302；损失冻结 L=30·focal_bce(.75,2)+1·soft_dice；stage1 8 图 200 epoch，stage2 454 图 30 epoch | 零初始化残差（step0 恒等）；复用冠军概率需重推理 | 冠军冻结、候选阈值 t=.10/d=0/margin24 冻结 | stage1 200ep；stage2 于 step 12597 触发 identity_broken(max_abs_diff .37159) | stage1：refined mIoU **.4191121** vs champ **.4266943**（Δ−.0075822），MA −.00332、HE −.02701 → 不晋级 | 未执行（test 从未读，失败守卫 fail-closed） | refiner_v1/GO_NO_GO.md、RESUME_REPORT.md、logs/.../stage2__TRAIN_FAILED.json | 可（stage2 恒等性破裂而失败；stage1 也未超冠军） |

## 2. 缺失材料

1. E3（MA 残差头家族）只有 284KB 训练日志与 0 字节 smoke，**无 eval/test JSON**。
2. H4 完整服务器 checkpoint 与原始训练日志缺失：本地仅 `h4_best.pth`(15,447 B) 与结果 JSON（`missing_dependencies.md:10-17` 仍列 BLOCKED）。
3. D@2000、E/E2/E3、L@6000 的完整服务器 checkpoint 与原始训练日志未回收。
4. autodl100 A/B 仅 weights-only，`backup_verified.json` 自标 "model metadata verification pending"。
5. FusionAware 概率缓存机制在代码中存在，但本地无任何 `.npy/.npz/.pt` 概率缓存。
6. U0/H0 无 test；H1/U1 无训练；FullCompetition 正式 preflight 占位符（`REQUIRED/TBD`）未清空。
7. `audit_20260916_A/experiments/experiment_decision.md` 为空文件。
8. 历史 combo/部分 combo 的原始服务器日志与中止原因缺失。
9. FullCompetition H0 的服务器权重与训练日志未本地回收（仅有 76 张导出 mask/contract 与复现值）。

## 3. 旧档案明显过时 / 矛盾之处

- **D1（H0 训练状态）**：02 档案与 CSV 记 H0 为 `locked_intent_not_trained`、H0@6000 为"无本地出处的口头数字"；但同日 `EXECUTION_REPORT_20260920.md §3/§5.2` 在 76 张 val 上**精确复现** H0 `.36922642429539565`（Δ=0）。→ H0 实为服务器已训练（≥6000 步）、本地仅缺权重/日志。
- **D2（E3 证据）**：03 档案称"`E3_0916__eval_*` 是逐配置原始结果"，但实测 `02_原始数据/eval_json/` 只有 `E3_0916__smoke.json`、`E3_0916__smoke_maond.json`（均 0 字节），无 E3 eval。与磁盘不符。
- **D3（冠军数值口径）**：早期审查写 test 44.21，FullCompetition/FusionAware 用精确 44.2115；champion val 有 `.426686`/`.4266942845`/`.4267` 三种精度（H4 单头 fused 口径 vs 完整冠军口径不同）。
- **D4（BASE0/D_BLEND 小数冲突）**：leaderboard 与位图重算差 1.05e-5 / 5.4e-7，属 champion variant A/B（EX .4577047841… vs .4577152550…）差异；未改写原文件。
- **D5（同名 U0）**：审查轮 RUN_PLAN 的 U0（HRPlus 多尺度计划，NOT_RUN）与 FullCompetition `runs` 的 U0（ConvNeXt-UNet，已完成 12k）不是同一实现；比较须带 family/目录。
- **D6（第一轮 copy-paste 条件被污染）**：`se_aug.py` 图像整块覆盖、标签逐像素 `maximum`、病灶库只存 SE 通道掩膜 → 图像与标签不一致。"SE copy-paste 泛化差"应撤销（≠反向成立）。
- **D7（"阈值校准是唯一正收益"撤回）**：val +0.7pp、E2E3 test +0.88pp 来自弱模型在 test 上调出的阈值；baseline 自身可调出多少从未测。
- **D8（test 独立性）**：多份报告自述 test 已被查看 ≥5 次；互补性报告另标 val 选择偏差样本。所有 test 值只能当探索性内部基准。
- **D9（42.20 的地位）**：官方 DDR mIoU ~32%，本项目 42.20 高约 10pp，来自自定义划分/训练量差异；baseline 从未重跑。

## 4. 所需证据

1. E3 的 val/test 原始 JSON，或明确宣告"E3 未评估"。
2. H4/H0/H1/U1 的服务器 checkpoint + 原始训练日志（含 config hash），完成身份核验。
3. autodl100 A/B 完整 checkpoint 元数据（现仅 weights-only）。
4. C/S 的 `C_summary/S_summary` 高精度字段与报告四舍五入值的逐字段映射。
5. refiner stage2 step 12597 恒等性破裂的定位证据（数值/实现问题 vs 协议后果）。
6. 患者/眼别级分组信息（文件名前缀全为批次号 `007`，患者级审计 BLOCKED）。
7. 历史 combo 空/部分 run 的原始日志与中止原因。

## 5. 可否否定技术路线的证据强度（汇总）

- **较硬可否定（同协议、有 test、反向一致）**：E1v2、E2E3、E4、E4v2、E4-CD、L(Lovasz)、R6-P1-HE、R6-P2-M1、R6b、C/S(MIX)、R1-B、R5-E0、R5-E3、refiner_v1。
- **仅可部分否定/未过门（无 test 或需谨慎）**：E1(Tversky)、E、E2、SE②、SE 板块三、D-TRAIN、H3/H6/H8/H9/H11/H12、R5-E1-V1/V2/E2、R6-P1-EX。
- **不可用于否定（计划/缺结果/单点/选择偏差/oracle）**：T0/T1/U0(审查计划)/E1/V1/J0/J1/C1/F1、FA-MAIN-PLAN、H1/U1、E3、combo 部分 run、FC 的 OR/argmax 组合、SE④ 的因果结论。
- **不可否定反而支持**：ENS-E4-B(mean)、D-TEST、ENS-MATRIX 集成、H4/H5/H7 冠军链路（均处探索性 test 口径）。

---

## history_ledger.json（同内容机器可读版）

```json
{"archive_date":"2026-09-26","mode":"read_only_audit","conventions":{"split":"454/76/227","seed":42,"threshold":0.5,"cmp":"strict_>","aggregation":"dataset_cumulative_TP_FP_FN","mean":"4-class equal","bg_excluded":true,"percent_vs_ratio":"2026-08 reports=percent; 09+ JSON=ratio; same quantity","delta_unit":"pp=percentage-point; ratio_diff=proportional"},"records":[
{"date":"2026-08","id":"HIST-BASELINE","family":"hist_0910","motivation":"internal baseline","change":"none","init":"random/ImageNet","frozen":"BN updating","steps":40000,"val":"39.390@32k","test":"EX51.56/HE43.13/SE49.98/MA24.11; 42.20","source":"诊断调研/HRDecoder_诊断与押注建议.md:55-64","can_negate":"baseline","status":"historical"},
{"date":"2026-08","id":"HIST-E1-Tversky","family":"hist_0910","change":"Dice->Tversky(a=0.3/0.7)","steps":"stop@4k","val":"collapse all-positive","test":"not_run","source":"同上","can_negate":"yes(no test)","status":"failed"},
{"date":"2026-08","id":"HIST-E1v2-CE-Dice","family":"hist_0910","change":"Dice->CE+Dice","steps":40000,"val":"39.970@12k","test":"39.43","source":"诊断调研/证据/验证轨迹.csv","can_negate":"yes","status":"complete"},
{"date":"2026-08","id":"HIST-E2E3","family":"hist_0910","change":"learnable fusion + ASPP-lite","steps":40000,"val":"39.206@20k","test":"EX52.15/HE38.45/SE46.07/MA23.26; 39.98; thr-tuned 40.86","source":"诊断调研/证据/e2e3_tuned_test_final.json","can_negate":"yes","status":"complete"},
{"date":"2026-08","id":"HIST-E4","family":"hist_0910","change":"detail branch + zero-init gate","steps":40000,"val":"38.766@36k","test":"EX50.93/HE41.61/SE49.84/MA24.13; 41.63","source":"HRDecoder_REPORT_final.md","can_negate":"yes","status":"complete"},
{"date":"2026-08","id":"HIST-E4v2","family":"hist_0910","change":"E4 + aux loss 0.3","steps":40000,"val":"38.667@36k","test":"EX50.66/HE36.29/SE49.32/MA23.95; 40.06","source":"诊断调研/证据/e4v2_test_final.json","can_negate":"yes","status":"complete"},
{"date":"2026-08","id":"HIST-E4-CD","family":"hist_0910","alias":"combo full","change":"E4+CE+Dice","steps":40000,"val":"39.871@32k; last 38.442@40k","test":"41.08","source":"诊断调研/证据/验证轨迹.csv:12-21","can_negate":"yes","status":"complete"},
{"date":"2026-08","id":"HIST-COMBO-PARTIAL","family":"hist_0910","val":"A 35.666@4k(n1)/v2 37.782@16k(n5)/combo2 37.089@12k(n6)/one empty","test":"not_run","source":"诊断调研/证据/最佳点核对.csv:3-7","can_negate":"no","status":"partial"},
{"date":"2026-08","id":"HIST-SE-2-copypaste","family":"lesion_seg","change":"SE copy-paste (img/label inconsistent)","steps":40000,"val":"39.7%@32k","test":"EX50.54/HE39.62/SE44.22/MA24.30; 39.67","source":"lesion_segmentation/04_板块二三汇总报告_module2_3_report.md","can_negate":"no(condition contaminated)","status":"complete"},
{"date":"2026-08","id":"HIST-SE-3-sliding","family":"lesion_seg","change":"SE sliding oversample","steps":40000,"val":"39.6%@20k","test":"EX51.77/HE38.22/SE46.30/MA24.01; 40.07","source":"同上","can_negate":"yes","status":"complete"},
{"date":"2026-08","id":"HIST-SE-4-combo","family":"lesion_seg","change":"copy-paste+sliding","steps":40000,"val":"40.4%@40k","test":"EX51.46/HE42.29/SE45.28/MA24.15; 40.79","source":"同上","can_negate":"yes","status":"complete"},
{"date":"2026-08","id":"HIST-LOSS-1.5x/2x/3x/adaptive","family":"lesion_seg","change":"SE loss weight","steps":40000,"test":"40.90 / 40.31 / 38.50 / 38.76","source":"同上","can_negate":"yes","status":"complete"},
{"date":"2026-09-11","id":"A","family":"autodl100","change":"semantic-guided MA residual + aux 0.02","init":"E4@36000","frozen":"BN running; lr 5e-4","steps":8000,"val":"38.632@1000; final 38.265","test":"not_run","source":"autodl100/实施记录.md:11-14","can_negate":"yes","status":"stopped"},
{"date":"2026-09-11","id":"B","family":"autodl100","change":"protocol-only + MA aux 0.02","init":"E4@36000","frozen":"same as A","steps":8000,"val":"39.205@3000","test":"EX52.65/HE44.37/SE48.91/MA21.74; 41.92","source":"autodl100/最终实验结论.md:3-16","can_negate":"yes","status":"complete"},
{"date":"2026-09-11","id":"ROUTE","family":"autodl100","change":"per-class val routing E4/B","steps":0,"val":"39.90","test":"EX52.65/HE41.61/SE48.91/MA24.13; 41.82","source":"autodl100/阶段2_判断.md:3-13","can_negate":"yes","status":"failed"},
{"date":"2026-09-11","id":"ENS-E4-B-mean","family":"autodl100","change":"0.5/0.5 prob mean","steps":0,"val":"39.54","test":"EX52.29/HE43.95/SE49.93/MA23.22; 42.35","source":"autodl100/最终实验结论.md:3-16","can_negate":"no","status":"complete"},
{"date":"2026-09-13","id":"D-TRAIN","family":"hrdecoder_plus3","change":"student-only, Dice, 50/50 sampling","init":"B@3000","frozen":"BN running","steps":20000,"val":"41.615@2000; final 40.476","test":"not_run","source":"hrdecoder_plus3/本轮实验报告_20260913.md:5-48","can_negate":"no","status":"complete"},
{"date":"2026-09-13","id":"D-TEST","family":"hrdecoder_plus3","change":"D@2000 test, no TTA","steps":0,"test":"EX53.68/HE41.67/SE50.23/MA24.73; 42.58","source":"hrdecoder_plus3/D方案_测试集补充报告.md:1-16","can_negate":"no","status":"complete"},
{"date":"2026-09-13","id":"L-PLAN(plus3)","family":"hrdecoder_plus3","change":"Dice+0.1 Lovasz config only","steps":0,"status":"planned_not_run","source":"本轮实验报告_20260913.md:17-20"},
{"date":"2026-09-15","id":"E","family":"MA_head","change":"MA residual head on B","init":"B","frozen":"backbone/ref frozen; head only","steps":"E@1500","val":"40.255 (EX44.98/HE41.39/SE51.84/MA22.81)","test":"EX53.64/HE43.83/SE47.63/MA25.13; 42.559","source":"最终报告_两轮实验汇总.md:31-46","can_negate":"no","status":"complete"},
{"date":"2026-09-15","id":"E2","family":"MA_head","change":"MA residual head on D","init":"D@2000","frozen":"backbone/ref frozen","steps":"not_recorded","val":"not_recorded","test":"EX53.68/HE41.67/SE50.23/MA24.60; 42.544","source":"同上:31-35","can_negate":"no","status":"complete"},
{"date":"2026-09-15","id":"E3","family":"MA_head","change":"third MA-residual arm","steps":"has 284KB train log","val":"not_recorded","test":"MISSING (only 0-byte smoke)","source":"05_日志/E3_0916__e_train.log","can_negate":"no(missing)","status":"incomplete"},
{"date":"2026-09-15","id":"L","family":"MA_head","change":"Dice+0.1 Lovasz","init":"D","steps":"best@6000","val":41.056,"test":"41.777(noTTA)/41.645(flip)","source":"第一轮_L对照与TTA结果.md:12-27","can_negate":"yes","status":"complete"},
{"date":"2026-09-16","id":"ENS-MATRIX","family":"MA_head","change":"prob-level ensemble + flip TTA","frozen":"all frozen","steps":0,"val":"D41.615/E40.255/B-start38.937","test":"best ENS_D3E2B1E41_flip 43.229; D*3+B+E4 43.149; D 42.576","source":"最终报告_两轮实验汇总.md:11-36","can_negate":"no","status":"complete"},
{"date":"2026-09-16","id":"AUDIT-PHASE-R0","family":"audit_0916","change":"no training; blend decomposition + bootstrap","steps":0,"val":"student .393351/reference .392050/blend .416148; dMIoU .0087 [.0057,.0118]; dMA .0349","test":"no_new","source":"STATE.json:1-75; final_report.md:9-28","can_negate":"no","status":"complete"},
{"date":"2026-09-16","id":"H3","family":"audit_0916","change":"MA fusion weight scan w in {0.5..1.0}","steps":0,"val":"MA +0.0047; mIoU +0.0012","test":"not_run","source":"FINAL_本轮最终报告.md:105-118","can_negate":"yes","status":"complete"},
{"date":"2026-09-16","id":"H4","family":"audit_0916","change":"post-fusion MA head 7ch 3361 params zero-init BCE.5+Dice.5 Adam1e-3 poly","init":"D@2000 features","frozen":"backbone frozen","steps":"2000(best@1600)","val":"corrected MA .252207; mIoU .426686","test":"fused .425764; corrected .434287 (EX.536760/HE.416678/SE.502328/MA.281383)","source":"experiments/h4_post_fusion_corrector.json","can_negate":"no","status":"complete"},
{"date":"2026-09-17","id":"H5/H7","family":"audit_0916","change":"head into champion D*3+E*2+B+E4 + flip TTA","frozen":"4 backbones frozen","steps":"inference","val":".4266942845 (45.77/43.76/55.18/25.97)","test":"EX.538611/HE.445130/SE.498161/MA.286558; 0.4421150356 (44.21)","source":"experiments/h7_test.json:1-95","can_negate":"no","status":"complete"},
{"date":"2026-09-16","id":"H6","family":"audit_0916","change":"HE/SE correction head","val":"HE best .4356<.4473; SE .5144<.5508","test":"not_run","source":"FINAL:105-118","can_negate":"yes","status":"complete"},
{"date":"2026-09-17","id":"H8","family":"audit_0916","change":"6000 steps lr2e-3","steps":6000,"val":"MA .2569","test":"MA .2858(down)","source":"FINAL:105-118","can_negate":"yes","status":"complete"},
{"date":"2026-09-17","id":"H9","family":"audit_0916","change":"HE head BCE/Dice=0.9/0.1","val":"HE .4382<.4473","test":"not_run","source":"FINAL:105-118","can_negate":"yes","status":"complete"},
{"date":"2026-09-17","id":"H11","family":"audit_0916","change":"MA head BCE-dominant","val":"MA .2510<H4 .2522","test":"not_run","source":"FINAL:105-118","can_negate":"yes","status":"complete"},
{"date":"2026-09-17","id":"H12/H13","family":"audit_0916","change":"whole-image supervision","val":"H12 MA .2565","test":"H12 MA .2803(down); H13 44.05<44.21","source":"FINAL:105-118","can_negate":"yes","status":"complete"},
{"date":"2026-09-17","id":"P5","family":"audit_0916","change":"H4 head seeds 1/2/3","val":"MA .2510/.2537/.2514","test":"not_run","source":"FINAL:119-131","can_negate":"no","status":"complete"},
{"date":"2026-09-16","id":"AUDIT-PLANNED-T0/T1/U0/E1/V1/J0/J1/C1/F1","family":"audit_0916","status":"NOT_RUN","note":"U0=HRPlus multi-scale; J0=train D student decoder; J1=decoder+HRNet last stage; C1=limited combo val-only; F1=authorized test","source":"RUN_PLAN.json:22-36"},
{"date":"2026-09-17","id":"R1-A","family":"fusionaware","change":"Dice(student)","init":"D@2000","frozen":"ref B frozen","steps":4000,"val":".4152/.4162/.4167/.4131","test":"not_run","source":"阶段成果报告:29-39","can_negate":"no","status":"complete"},
{"date":"2026-09-17","id":"R1-B","family":"fusionaware","change":"0.5*Dice(stu)+0.5*Dice(0.5stu+0.5ref)","init":"D@2000","frozen":"ref frozen","steps":4000,"val":".4162/.4169/.4175/.4142","test":"not_run (B-A max +0.0008 < gate +0.003)","source":"同上","can_negate":"yes","status":"complete"},
{"date":"2026-09-18","id":"R5-E0","family":"fusionaware","change":"champion + 6th member BNEW","steps":"inference","val":".4273/.4278/.4272 (+0.0011)","test":"not_run","source":"ROUND5:22-34","can_negate":"yes","status":"complete"},
{"date":"2026-09-18","id":"R5-E1-V1","family":"fusionaware","change":"pure fusion w=1.0 seed42 T=3000","init":"D","steps":3000,"val":".4170/.4181/.4187 (G1 pass)","test":"not_run","source":"ROUND5:36-46","can_negate":"no","status":"complete"},
{"date":"2026-09-18","id":"R5-E1-V2","family":"fusionaware","change":"curriculum 0->0.5->1.0","steps":3000,"val":".4152/.4166/.4181","test":"not_run","source":"ROUND5:36-46","can_negate":"yes","status":"complete"},
{"date":"2026-09-18","id":"R5-E2","family":"fusionaware","change":"V1 seed43","steps":3000,"val":".4176/.4179/.4178","test":"not_run","source":"ROUND5:47-53","can_negate":"no","status":"complete"},
{"date":"2026-09-18","id":"R5-E3","family":"fusionaware","change":"champion + V1 student","steps":"inference","val":".4273 (+0.0006)","test":"not_run (G3 fail)","source":"ROUND5:55-62","can_negate":"yes","status":"complete"},
{"date":"2026-09-18","id":"R6-P0","family":"fusionaware","change":"champion reproduce","frozen":"all frozen","steps":"inference","val":".4267 (.4577/.4376/.5518/.2597)","test":".4421 (.5386/.4451/.4982/.2866)","source":"ROUND6:11-18","can_negate":"no","status":"complete"},
{"date":"2026-09-18","id":"R6-P1-HE","family":"fusionaware","change":"HE correction head","val":"best .4360 < baseline .4376 (-0.0015)","test":"not_run","source":"ROUND6:11-18","can_negate":"yes","status":"rejected"},
{"date":"2026-09-18","id":"R6-P1-EX","family":"fusionaware","change":"EX correction head","steps":"cache 50/76 budget expired","status":"skipped","source":"阶段成果报告:62-70","can_negate":"no"},
{"date":"2026-09-18","id":"R6-P2-M1","family":"fusionaware","change":"MA expert 512^2 Dice+0.5BCE","steps":3000,"val":"patch IoU 0/0/.0052/.0220/.0100/.0137; a=.3 MA .2438 vs .2597 (-0.0159)","test":"not_run","source":"阶段成果报告:71-80","can_negate":"yes","status":"rejected"},
{"date":"2026-09-18","id":"R6-P3","family":"fusionaware","change":"final champion test","steps":0,"val":".4267","test":".4421 (delta 0.0000)","source":"阶段成果报告:82-83","can_negate":"no","status":"no_gain"},
{"date":"2026-09-18","id":"R6b","family":"fusionaware","change":"256^2 + Tversky(0.3,0.7) 2000 steps","steps":2000,"val":"best patch IoU .0563; a=.3 MA .2431 (-0.0166)","test":"not_run","source":"ROUND6:42-69","can_negate":"yes","status":"rejected"},
{"date":"2026-09-19","id":"C_S","family":"fusionaware","change":"C=orig mixed crop; S=class-balanced instance crop; init D@2000+B; seed42; 4000 steps SGD","steps":4000,"val":"C2000 .3979739; C4000 .3917288; S2000 .3864550; S4000 .3983198; MIX val .4286569","test":"champion .4421152; C2000 .398104; MIX .4380636; delta -0.0040516 = -0.4052pp","source":"ROUND_CS_FINAL_REPORT_20260920.md","can_negate":"yes","status":"complete"},
{"date":"2026-09-19","id":"FA-MAIN-PLAN","family":"fusionaware","status":"planned_not_started","source":"EXECUTION_PLAN_CURRENT.md:1-16"},
{"date":"2026-09-20","id":"U0-REAL","family":"fullcomp","change":"ConvNeXt-S+U-Net AdamW ce+dice eff batch8 whole-image","init":"builtin ImageNet (external init disabled at export)","steps":12000,"val":"@3k .1747592 / @6k .3500054 / @9k .3743590 / @12k .3844641257662971 (EX.436760/HE.384266/SE.509362/MA.207468)","test":"not_run","source":"runs/completion_summary.json","can_negate":"yes","status":"complete"},
{"date":"2026-09-20","id":"H0","family":"fullcomp","change":"HRDecoder-W48 DDR-only; local config locked-intent","init":"ImageNet","frozen":"backbone BN frozen, affine trainable","steps":"local none; server >=6000","val":"76-img export reproduces .36922642429539565 (EX.418771/HE.435195/SE.414545/MA.208394)","test":"not_run","source":"EXECUTION_REPORT §3/§5.2","can_negate":"yes","status":"server_trained_local_assets_missing"},
{"date":"2026-09-20","id":"H1","family":"fullcomp","status":"locked_intent_not_trained","source":"configs/H1_hrdecoder_ext_pretrain.yaml"},
{"date":"2026-09-20","id":"U1","family":"fullcomp","status":"locked_intent_not_trained","source":"configs/U1_convnext_unet_ext_pretrain.yaml"},
{"date":"2026-09-20","id":"FC-EXPERT-COMP","family":"fullcomp","change":"val-only per-pixel complementarity, 8 models x 76","steps":0,"val":"champion 42.6694; MIX_C2000 42.8657; BASE_0 41.7973; D_blend 41.6149; C2000 39.7974; S4000 39.8320; U0 38.4464; H0 36.9226","test":"not_run","source":"EXECUTION_REPORT §4-5","can_negate":"no","status":"analysis_complete"},
{"date":"2026-09-20","id":"refiner_v1","family":"fullcomp","change":"champion-prior MA/HE residual head; L=30*focal_bce(.75,2)+1*soft_dice frozen","init":"zero-init residual (step0 identity)","frozen":"champion frozen; candidate rule t=.10/d=0/margin24 frozen","steps":"stage1 200ep (8 img); stage2 30ep (454 img)","val":"stage1 refined .4191121 vs champ .4266943 (d -0.0075822); MA -0.00332; HE -0.02701 -> no promote","test":"not_run","source":"refiner_v1/GO_NO_GO.md; RESUME_REPORT.md; stage2__TRAIN_FAILED.json (identity_broken @step12597, max_abs_diff .37159)","can_negate":"yes","status":"failed"}],
"missing":["E3 eval/test JSON","H4 full server ckpt+logs","D/E/E2/E3/L full ckpts","autodl100 A/B full ckpt metadata","FusionAware prob caches","U0/H0 test; H1/U1 training","experiment_decision.md empty","historical combo server logs","H0 server weights/logs"],
"conflicts":["D1 H0 trained-on-server vs archive locked_intent","D2 03 archive claims E3_0916__eval_* exist but disk has only 0-byte smoke","D3 champion 44.21 vs 44.2115; val .426686/.426694/.4267","D4 BASE0/D_BLEND variant A/B diff 1.05e-5/5.4e-7","D5 duplicate name U0 (audit plan vs fullcomp run)","D6 SE copy-paste img/label inconsistency invalidates causal claim","D7 threshold-calibration 'only gain' retracted","D8 test viewed >=5 times (not independent)","D9 42.20 is internal baseline vs official ~32%"],"needed_evidence":["E3 results","H4/H0/H1/U1 ckpt+logs+hash","A/B ckpt metadata","C/S summary vs report mapping","refiner step12597 root cause","patient/eye grouping","historical combo logs"]}
```

Both files are ready to write the moment write permission is granted. Want me to retry once you've enabled it?