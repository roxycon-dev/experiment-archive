# HRDecoder 单模型架构设计历史证据审计

本审计整理既有训练、验证、测试和离线互补性结果，供单个 HRDecoder 新架构设计参考；不据此选定核心方案。指标均为 DDR 四类病灶 mIoU，排除背景，历史主要划分为 train/val/test = 454/76/227。表内 pp 指百分点；以 0–1 小数保存的 mIoU 差值会同时换算成 pp。所有引文路径相对于 `D:\dataset\experiment`，行号对应本地文件。

## 先分清比较对象

45.8604% 是最终固定概率系统的测试成绩，不是单个 HRDecoder 的成绩。最终输出为 `0.75·P0 + 0.25·S20 EMA`；P0 已包括旧 HRDecoder 多成员融合及 H4 MA 修正，并已混入 ConvNeXt-S U0，S20 是 DINOv3 ViT-B/16 分支。S20 的独立验证 mIoU 为 42.2703%，相对匹配继续训练对照 C20 高 1.2459 pp；其测试只在固定概率组合中评估，未测 S20 单分支。对上一系统 R 的最终测试增量仅 0.1388 pp，图像级配对 bootstrap 95% 区间为 −0.2626 至 +0.6519 pp。故 45.8604%可作为项目系统参照，不能当作单个 W48 网络已达到的分数，也不能把累计差额归因给一次架构改动。[最终系统及分支口径：`HRDecoder_FinalDecision_20261005/结题报告/HRDecoder项目结题报告.md:7-15,472-486,498-500`；S20 独立验证与匹配对照：`HRDecoder_FinalDecision_20261005/最佳模型技术报告/报告正文.md:181-187,219-223`；bootstrap：`HRDecoder_FinalDecision_20261005/结题报告/HRDecoder项目结题报告.md:393`]

历史上最接近“ImageNet HRNet-W48、完整 40k 预算”的参照是 B0：原 HRDecoder、ImageNet HRNet-W48 初始化、训练预算 40,000 步，最佳验证点 @32k，val 39.3900%、test 42.20%。E4 同样跑 40k，但它是另一个加细节分支的实验：ImageNet 初始化、BN 统计继续更新、细节尺度零初始化，best @36k，val 38.766%、test 41.63%。这两项说明 40k 和 ImageNet 起点确有历史参照，但它们没有测试本次拟议的新结构或强教师蒸馏。[`HRDecoder_FinalDecision_20261005/最佳模型技术报告/报告正文.md:33`；`_全实验档案_20260920/02_HRDecoder逐实验事实档案.md:30`]

FullCompetition H0 是另一条需单列的记录：它实际用 ImageNet HRNet-W48 启动，配置目标为 12k DDR updates，有 train 日志和 3k/6k 验证 JSON；记录止于 6,000 步，未找到完整 H0 checkpoint。76 图重算得到 val 36.9226424%（EX 41.8771、HE 43.5195、SE 41.4545、MA 20.8394），test 未测。由于运行只到 6k，不能把它解读成 40k 结果或模型上限。[`HRDecoder_重新诊断与三点提升方案_20260926/history_ledger_reviewed.json:623-634`；`HRDecoder_FullCompetition_20260920/analysis_results/expert_complementarity_20260920/EXECUTION_REPORT_20260920.md:110-122`]

档案状态有一处已被后续证据纠正的冲突：9/20 快照把 H0 标为未训练/本地无证据；9/26 审核台账和 FullCompetition 执行报告显示了真实 6k 日志及 76 图结果。应记作“实际部分训练，未完成计划预算”，而非“未运行”。[旧快照：`_全实验档案_20260920/02_HRDecoder逐实验事实档案.md:102`、`HRDecoder_FullCompetition_20260920/docs/champion_asset_inventory_workbuddy.md:155-158`；后续核对：`HRDecoder_重新诊断与三点提升方案_20260926/history_ledger_reviewed.json:623-634`]

## 局部学生、参考分支与融合感知

D 是最直接的局部高分辨率学生记录，但其“student-only”容易造成误读。D 从 DDR 训练过的 B@3000 权重启动，不是从 ImageNet 直接启动；只训练学生 20k 步，BN running stats 冻结、affine 可训练，Dice 损失，50% 随机视野与 50% 类平衡阳性中心裁块。best val 在 @2k，为 41.615%；末点为 40.476%。后续 D@2k test 为 42.576%，较历史 B0 的 42.20 高 0.376 pp；然而这个推理输出是高分辨率学生滑窗预测与冻结 B 整图参考概率各 0.5 融合、无 TTA，因而 41.615/42.576 都不是学生单模型指标。D 的局部训练、采样、初始化和参考融合同时变化，不能将变化单独归因于局部窗口模块。[`_全实验档案_20260920/02_HRDecoder逐实验事实档案.md:48`；`HRDecoder_FinalDecision_20261005/结题报告/HRDecoder项目结题报告.md:194-198`]

实现也不是共享参数的全局/局部分支：HRPlus 代码分别构造学生和 `reference`，从同一 B checkpoint 分别加载，reference 参数冻结；推理把整图 reference 概率与学生 tiled 输出做等权概率融合。这与“单个共享 HRNet 主干上同时走全局和局部路径”是不同的算力、参数和预测结构。[`hrdecoder_plus3/repo/hrplus_patch.py:53-88,116-151`；`hrdecoder_plus3/repo/hrplus_core.py:67-76`]

R1 对照直接问过把冻结参考概率放进训练目标是否帮助 D 学生：两臂均从 D@2000 起步，reference B 冻结，各训 4k。A 用 Dice(student)，val 四节点为 41.52/41.62/41.67/41.31%；B 用 `0.5·Dice(student)+0.5·Dice(0.5·student+0.5·reference)`，为 41.62/41.69/41.75/41.42%。最佳差是 0.0008 mIoU，即 0.08 pp；预注册门槛是至少 0.003 mIoU（0.30 pp），因此未过门、未测 test。该结果限制在已 DDR 训练的 D 起点、4k 预算和这项混合损失；它没有检验从相同 ImageNet W48 起步、40k 训练的蒸馏，也不是对所有 teacher loss 的否定。[`HRDecoder_FusionAware_20260917/HRDecoder-DDR-阶段成果报告-20260917至0918.md:29-39`；`HRDecoder_重新诊断与三点提升方案_20260926/history_ledger_reviewed.json:432-456`]

R5 的纯融合监督 V1（D 起点、seed42、3k）达到 41.87% val 并首次过 G1 的 41.85% 门槛；课程式 V2 @3k 为 41.81%。V1 seed43 的节点为 41.76/41.79/41.78%，没有复制到 G1 线。更关键的是，加入冠军集成后，BNEW 最好仅增加 0.11 pp，V1 仅约 0.06 pp，均未过预注册的 +0.20 pp 集成门槛，也都没有 test 确认。可以确认的是“该训练目标在一个 seed 的 student val 首次过线，但加入既有系统后没有达到采纳门槛”；“集成饱和”是报告对这些观测的解释，尚非已证明的普遍机制。[`HRDecoder_FusionAware_20260917/ROUND5_RESULT_20260918.md:13-17,24-62,68-79`；seed43：`HRDecoder_重新诊断与三点提升方案_20260926/history_ledger_reviewed.json:496-517`]

## MA 残差分支的实际作用范围

H4 是历史上证据最清楚的类选择性残差模块，但它依赖一个已训练的多路预测底座。H4 有 3,361 个参数，输入包括 RGB、融合后的 MA logit、学生 MA logit、reference MA logit及学生/reference 概率差；零初始化，仅训练校正头。单接 D+reference 时，best@1600 的 val MA IoU 为 25.2207%；test 上 fused mIoU 42.5764%、MA 24.7290%，加 H4 后分别为 43.4287%、28.1383%，EX/HE/SE 逐位不变。接到完整 D/E/B/E4 翻转集成时，历史 H5/H7 的 test mIoU 从 43.2289%到 44.2115%，MA 从 24.7255%到 28.6558%。两组比较的底座不同，不能把其中一个数字拼到另一个基线上；更不能把 H4 的提升视作从单一 ImageNet W48 特征直接得到的增量，因为它读取的输入正是各分支和融合预测。[结构与参数：`_全实验档案_20260920/02_HRDecoder逐实验事实档案.md:69`；D+reference 原始结果：`HRDecoder_审查轮_20260916/audit_runs/audit_20260916_A/experiments/h4_test_confirmation.json:1-55`；H5/H7：`HRDecoder_重新诊断与三点提升方案_20260926/history_ledger_reviewed.json:328-352`；H4 直接效应摘要：`HRDecoder_PaperInnovation_20261006/论文创新定位与方法主线.md:23-30`]

更简单的 E/E2 MA 头分别报告 test 42.559%（B 起点）和 42.544%（D 起点）；这些也是各自实际推理管线的数字，不能替代未融合新单模型的结果。[`HRDecoder_MA残差头与集成_20260915-16/01_结果报告/最终报告_两轮实验汇总.md:11-36`；复核台账：`HRDecoder_重新诊断与三点提升方案_20260926/history_ledger_reviewed.json:238-263`]

MA 从 RGB patch 直接训练的两次专家实验均未形成有效融合增益。R6 的 512² Dice+0.5 BCE 方案曾出现全负退化，patch IoU 长段为 0，α 扫描最优 val MA 24.38%，比冠军 25.97%低 1.59 pp。报告记录 patch 正像素率约 1e-4、BCE 受大量负像素主导，这与退化现象相符。R6b 改为 256² 和 Tversky(0.3,0.7)，best patch IoU 只有 0.0563，最佳融合 MA val 24.31%（低 1.66 pp）；训练末期 loss 仍下降，且 2k 预算短，所以“本配方、本预算失败”有实证，“所有 RGB MA 专家都不可行”没有实证。HE 修正头 best val 43.60%，低于恒等基线 43.76%；这只拒绝这次小头及条件，不能泛化到所有非 MA 头。[`HRDecoder_FusionAware_20260917/ROUND6_RESULT_20260918.md:11-18,31-40,42-69`；较审慎的审计表述：`HRDecoder_重新诊断与三点提升方案_20260926/history_ledger_reviewed.json:533-563`]

## FullCompetition：结果与互补性口径纠正

U0 是实际完成的 ConvNeXt-S+U-Net、官方 ImageNet ConvNeXt-S 初始化、12k DDR updates。best@12k val mIoU 38.4464%（EX 43.6760、HE 38.4266、SE 50.9362、MA 20.7468），test 未测。它单模型 val 明显低于旧 HRDecoder 系统，但之后仍成为有效融合伙伴。[`HRDecoder_重新诊断与三点提升方案_20260926/history_ledger_reviewed.json:609-620`；`HRDecoder_FullCompetition_20260920/runs/completion_summary.json:1-40`]

必须纠正 FullCompetition 专家分析报告 §5.3 对 OR 的文字。该报告说 OR “需要每像素真值才可构造”且把 OR 列称作“上限”；这不符合代码。分析脚本直接从 candidate 预测 `p`、champion 预测 `q`计算双方共同/独有 TP、FP，再按并集计数；GT 参与的是把预测像素标记为 TP、FP、FN 并计算 IoU，不用于选择 `p|q` 中像素。OR 是可执行的二值并集规则，通常以增加召回换取 FP；它既不是 GT 选择的 oracle，也不是对任意概率融合或模型的通用上界。OR 负值只说明这一个硬并集规则在该 val 集上净 IoU 下降，不能推出所有概率融合都无互补价值。[原报告错误表述：`HRDecoder_FullCompetition_20260920/analysis_results/expert_complementarity_20260920/EXECUTION_REPORT_20260920.md:209-220`；代码：`HRDecoder_FullCompetition_20260920/analysis_results/expert_complementarity_20260920/scripts/analysis/analyze_expert_complementarity.py:304-343`；后续审核已指出“OR是可执行并集，不是GT oracle”：`HRDecoder_重新诊断与三点提升方案_20260926/history_ledger_reviewed.json:653-662`]

数值上，U0 的 76 图 val mIoU 为 38.4464%；U0 与冠军二值 OR 的 mIoU 为 39.6819%，相对冠军 42.6694%低 2.9875 pp。U0 独有 TP/FP 计数（EX/HE/SE/MA）分别是 17,007/62,318、79,689/287,515、10,293/35,495、6,121/45,368。这里可得出的结论是 U0 独有正确像素伴有大量独有误报，使硬 OR 不划算；不能据此说 U0 的软概率对所有决策都无价值。[`HRDecoder_FullCompetition_20260920/analysis_results/expert_complementarity_20260920/analysis_outputs/EXPERT_COMPLEMENTARITY.md:18,23-33`]

后续固定概率融合提供了反例：0.75 旧冠军 + 0.25 U0 在唯一锁定的 227 图评估中从同次冠军复算 44.2110%升至 44.7809%，+0.5699 pp，EX/HE/SE/MA 四类分别 +0.7367/+0.6899/+0.2704/+0.5825 pp；本轮没有在完整 val 上重新搜索融合权重。但融合不是稳定保证：更早 0.75 champion + 0.25 C@2000 的 val 增益 +0.1963 pp，test 反而 −0.4052 pp。两者共同说明单模型 mIoU、二值 OR 与概率融合不是同一量，某一个融合规则的正负结果不能移植到另一个规则或新学生。[U0 融合：`_全实验档案_20260920/04_本轮实验补充_20261002.md:5-18,43-45`；C 融合失败：`HRDecoder_重新诊断与三点提升方案_20260926/history_ledger_reviewed.json:590-599`]

## 对单模型证据的边界

- 历史确有 ImageNet HRNet-W48、40k B0 基线和 40k E4 结构变体；没有找到同一起始 ImageNet W48、相同 40k 更新、单模型评估，并对 teacher 蒸馏作开/关匹配的已完成对照。R1/R5 的 teacher/fusion-aware 证据起点是 D@2000，只有 3k–4k 更新，不能回答此次同起点完整训练下蒸馏的净效应。
- D 是局部学生+冻结参考的双模型概率输出，不是共享主干单模型的证据。H4 是作用于已融合输出的 MA 后校正器，不是单 W48 内部 MA 分支的等价测试。U0 的 test 融合增益来自异构 ConvNeXt 互补，不能当作 HRNet 单模型提升。
- FullCompetition 的 C@2000 概率混合出现 val 正、test 负；U0 混合则 test 正。项目台账指出历史 test 多次被查看，因此这些数字可用于理解本划分内的方向和失败模式，不是独立外部泛化估计。[`HRDecoder_FullCompetition_20260920/analysis_results/expert_complementarity_20260920/analysis_outputs/EXPERT_COMPLEMENTARITY.md:69-76`；`_全实验档案_20260920/04_本轮实验补充_20261002.md:43-45`]
- 因此，历史资料对“哪些模块有过条件性收益、哪些组合方式曾失败”给出了证据；它没有给出一个已验证的单个 HRDecoder 架构能够达到 45.8604% test 的历史实例。该缺口不证明新架构做不到，只限定了不能把哪些历史复合分数当作单模型证据。

本审计只读取既有档案、原始汇总、关键代码和指标；没有运行训练或评估，也没有修改任何历史文件。
