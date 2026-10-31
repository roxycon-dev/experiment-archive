# v4 根因诊断：历史修复约束

本页只归纳本地已完成实验和源码，可支持的是“哪些处理曾与效果同现、哪些做法出现反例”；不把相关性写成当前 v3 崩塌的原因。此次另只读检查了 20k 与 best-val@8k 的数值审计，没有训练、远程访问或改写旧文件。

## 可核验的正向经验及其适用边界

- **融合监督的正向证据有限且是 val 证据。** R1 从 D@2000 学生起步，冻结 B 参考概率；对照 A 只监督 student Dice，B 则为 0.5·Dice(student)+0.5·Dice(0.5·student+0.5·reference)。R1 三千步时 A/B 分别为 0.4167/0.4175（差 0.0008），四千步分别 0.4131/0.4142；两臂都未进入 test。R5 的纯融合目标 V1（w=1）seed42 在 3k 为 0.4187，seed43 3k 为 0.4178，相对 seed42 的 student-only A@3k 0.4167 方向一致；但新学生加入冠军仅增约 0.0006，未过集成门、无 test。源码明确 reference detach，融合在概率域做 50/50；R5 训练共用 runner，batch=2、SGD lr=5e-5/momentum=0.9/weight_decay=0.0005，冻结 BN running stats，只保留 decode_head 与 backbone.stage4 等许可参数训练。此记录支持“把下游固定融合写进训练目标可在特定 D 起点短训中小幅改善 val”，不支持它是 MA 专项蒸馏或可迁移到共同更新的多支路单模型的保证。没有逐类结果，不能声称它提升了 MA/小病灶。
  - 证据：HRDecoder_FusionAware_20260917/HRDecoder-DDR-阶段成果报告-20260917至0918.md:29-39；HRDecoder_FusionAware_20260917/ROUND5_RESULT_20260918.md:36-62,68-75；HRDecoder_FusionAware_20260917/run_round5.py:47-61,263-291；HRDecoder_FusionAware_20260917/run_protocol.py:430-468；HRDecoder_FusionAware_20260917/fusion_core.py:91-110,228-264。

- **MA 专项成功路径有已学概率先验。** H4 不是从 RGB 从零学习 MA：D@2000 student 与独立 B reference 冻结，先做 0.5/0.5 概率融合；只有一个 3,361 参数零初始化头拟合 MA 残差。输入含 RGB、fused/student/reference MA logits 和 student-reference 概率差，输出只修正 MA。训练用 BCE 0.5 + soft Dice 0.5、Adam 1e-3、线性 poly 衰减、梯度裁剪 5；基模型处于 eval 且不回传梯度。H4 val MA IoU 从 fused 0.226174 到最佳 corrected 0.252207（step 1600）；独立 test confirmation 中 D/B fused MA 0.24729→corrected 0.281383，整体 mIoU 0.425764→0.434287。接到完整冠军集成后，test MA 0.247255→0.286558，整体 mIoU 0.432289→0.442115；这属于七成员概率融合加 TTA 的系统收益，不是单网络收益。H4 式头迁到 HE 的 val 0.4360 低于融合基线 0.4376，整图 MA 监督 H12 的 test mIoU 44.05 也低于旧冠军 44.21，故不能把 MA 修正收益泛化到其它类别或任意训练/推理网格。
  - 证据：HRDecoder_ArchitectureDesign_20261008/evidence/h4_code_audit.md:9,14-20,22-26；HRDecoder_审查轮_20260916/tools/h4_post_fusion_corrector.py:56-78,112-120,175-225；_全实验档案_20260920/02_HRDecoder逐实验事实档案.md:69-76,124；HRDecoder_FusionAware_20260917/ROUND6_RESULT_20260918.md:31-38。

- **稀疏病灶的采样要保留“阳性可见机会”，但 D 不是浅层头对照。** D 训练 1024² patch，50% 随机裁、50% 从图中实际存在的病灶类均匀选类并锚定一个该类正像素；没有病灶的图退回随机裁。D 用 HRNet 对每个 tile 重新编码，冻结 B 全图 reference，推理固定两路各 0.5；BN running stats 冻结、affine 可训练，SGD lr=5e-4、poly 到 5e-5、warmup 200。val 最佳 41.615@2k、20k 末点 40.476；D@2000 test 42.58。其输出仍是 student+reference，不能把 41.615/42.58 记成独立单网分数，也不能把 D 的重复局部 HRNet 编码等价成一条 RGB 浅层残差头。
  - 证据：_全实验档案_20260920/02_HRDecoder逐实验事实档案.md:48-49；hrdecoder_plus3/repo/hrplus_patch.py:154-201；HRDecoder_FusionAware_20260917/fusion_core.py:113-139；HRDecoder_ArchitectureDesign_20261008/evidence/review_v1_objectives.md:17。

## 空类别、忽略像素与失败边界

- DDR 的四个输出是 EX/HE/SE/MA 独立前景通道；标签 0 是背景、1–4 是四类、255 是 ignore。**一个 crop 某类没有正像素，不等于该类未标注：**原 BinaryLoss 对每个前景通道仍计算 BCE/Dice；无正例时平滑项使 Dice 有限，非零预测仍受惩罚。融合版 dice_prob 也把 255 排除，但对已标注而无阳性的类保留负例惩罚。FullCompetition 后续 valid-mask 契约把两种情形分开：valid=0 表示未标注整类，零梯度；DDR full 标注则 valid=1，空目标仍是负监督。不得把“当前 crop 没有 MA”编码成 valid=0。
  - 证据：hrdecoder_plus3/repo/mmseg/models/losses/binary_loss.py:11-37,46-62,94-115；HRDecoder_FusionAware_20260917/fusion_core.py:7-9,34-79；HRDecoder_FullCompetition_20260920/trainer/loss.py:58-86；HRDecoder_FullCompetition_20260920/docs/TRAINING_OBJECTIVE.md:36-44,75-86。

- **ignore 语义不要照搬 H4 的实现细节。** H4 训练代码把 gt==255 替成 0 后做 BCE 与 Dice，因此 ignore 像素会被当作 MA 负例；评估时则排除了 ignore。该已获益头仍带此实现瑕疵，不能把正向结果当作此处理正确的证据。若新目标保留 ignore，监督分子、归一化分母和 Dice 项都应由 valid 一致屏蔽。
  - 证据：HRDecoder_审查轮_20260916/tools/h4_post_fusion_corrector.py:139-140,198-202；HRDecoder_ArchitectureDesign_20261008/evidence/h4_code_audit.md:18,53。

- **无先验的 MA 专家是强反例。** R6 512² RGB 专家正例率约 1e-4，Dice+BCE 早期落入全负解，loss 约 1.0、patch IoU 为 0；3000步后与冠军融合仍把 MA val IoU 从 0.2597 降到 0.2438（−0.0159）。256²+Tversky 将最佳 patch IoU 提到 0.0563，但融合 MA val 仍降至 0.2431（−0.0166），拒绝。结论只适用于已测的无先验专家配方，不能说加大分支或改 Tversky 本身必然无效；它支持优先保留成熟概率先验和检查正例暴露量。
  - 证据：HRDecoder_FusionAware_20260917/ROUND6_RESULT_20260918.md:11-18,31-38,42-69。

## BN 与优化器：历史上观察到什么

- 原 HRDecoder baseline 使用 batch=2、SGD lr=0.005/momentum=0.9/weight_decay=0.0005，HRNet norm_eval=False，BN running stats 在 train 模式更新；它本身是有效的历史基线。D 与 R1/R5 后续路线冻结 running stats、允许限定参数的 BN affine 训练，采用较低 SGD lr；H4 则冻结模型并只用 Adam 更新小头。它们是不同初始化/目标/训练量下的配方，不能从“成功运行过”推出 BN 冻结导致提升。
  - 证据：hrdecoder_plus3/repo/configs/lesion/exp/baseline_split613.py:26-56；hrdecoder_plus3/repo/configs/_base_/models/hrdecoder_fcn_hr18.py:2-10,45-60；_全实验档案_20260920/02_HRDecoder逐实验事实档案.md:48,69；HRDecoder_FusionAware_20260917/run_protocol.py:430-468；HRDecoder_审查轮_20260916/audit_runs/audit_20260916_A/FINAL_本轮最终报告.md:40-45。

- BN 重估的已执行对照对 D@2000 有害：val mIoU 41.615→40.363（只重估 student）；把 reference 也重估则 25.286，MA IoU 3.12。它否定的是这次重估操作适合该 D/reference 检查点的主张，不定位 v3 的 20k 异常成因，也不等价于训练期 BN freeze 的受控消融。
  - 证据：HRDecoder_MA残差头与集成_20260915-16/01_结果报告/最终报告_两轮实验汇总.md:38-46；HRDecoder_MA残差头与集成_20260915-16/HANDOVER_完整交接文档.md:574-579。

## 20k 检查点相对 8k 能证实的变化

下表来自同一份 CPU 只读报告：best_val 为 8k，last 为 20k；两者各有 1707 个浮点模型张量、307 个 BN 方差 buffer，模型/优化器非有限张量列表为空、负 BN 方差为空。脚本只枚举张量有限性、running_var 和指定输出层权重行范数，不执行 forward 或 SGD；所以“PASS”不是健康推理认证。

| 项目 | best-val @8k | last @20k | 能确认的事 |
|---|---:|---:|---|
| backbone.layer1.0.bn3 running_var 最小值 | 2.2149e-16 | 6.2694e-16 | 极小方差 8k 已有，20k 仍在同数量级 |
| backbone.layer1.0.downsample.1 running_var 最小值 | 4.8972e-16 | 1.7000e-16 | 该层也同数量级变化；不是20k独有 |
| backbone.bn1 running_var 最小值 | 1.8235e-14 | 1.3831e-14 | 两个节点均有很小的通道方差 |
| backbone.transition2.2.0.1 running_var 最小值 | 5.7174e-13 | 2.8524e-13 | 20k低约一半；单个快照不指示异常原因 |
| conv_seg.weight 各输出行范数（EX/HE/SE/MA） | [1.7948, 1.7900, 1.6293, 1.8699] | [3.1514, 1.8841, 3.4567, 2.0063] | EX、SE行范数变化较大，HE、MA较小；没有范数阈值或对应干预证明它们造成零预测 |

20k val mIoU 为 15.0509%；EX、SE 无任何正预测，而 HE、MA 仍有预测。小样本 CPU FP32 复核检查了两张图，其中 007-3873-200 含 EX/SE 正像素 580/1168，输出最大概率仅 3.6772e-6/4.4216e-7；该图原始 base logits 已为 -10.5520/-12.3418，新增 residual 也为负。这确认至少该图存在极低概率输出，但不能确定训练机制原因，也不支持只归因于 residual。健康审计 JSON 自己也声明，有限参数和 buffer 不证明推理输出有限，也不能确定验证崩塌原因。

- 证据：HRDecoder_ArchitectureDesign_20261008/cache_execution/server_receipts/reports/diagnostics/checkpoint_health_20000.json:3-13,100-184,625-634,721-805,1243-1248；HRDecoder_ArchitectureDesign_20261008/training_execution/checkpoint_health_audit.py:15-68；HRDecoder_ArchitectureDesign_20261008/training_execution/validation_analysis/val_20000.md:1-9；HRDecoder_ArchitectureDesign_20261008/02_架构规格.json:11。

## 诊断边界

现有历史支持把冻结参考概率、阳性锚定 crop、valid/ignore 定义、BN buffer 与 affine 参数状态、优化器/调度和输出 logit/概率范围分别记录；不支持仅凭“参数有限”“BN 方差非负”或某个权重范数判断训练健康，更不支持从历史配方相关性直接宣告 v3 20k 崩塌由 BN、蒸馏、类别稀疏或单一模块导致。
