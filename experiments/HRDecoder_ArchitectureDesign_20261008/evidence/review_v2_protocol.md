# HRDecoder-CF v2 窄范围协议复审

审查对象：`D:\dataset\experiment\HRDecoder_ArchitectureDesign_20261008\04_v2_修订架构.md`。相对路径 `FinalDecision_20261005/...`、`lesion_segmentation/...` 均相对于 `D:\dataset\experiment\`，`evidence/...` 指本设计目录下的 evidence。只读复审；未训练、未运行模型推理、未修改 v2 主文档。

## 结论

v2 比 v1 更清楚地限定了比较口径：40k 是同批次数/主网更新数，不是等 FLOPs；H4 已统一到 2048 工作网格；teacher 软图在原图坐标构造后与学生共享几何，软目标插值、有效域和 KD 权重顺序也有说明。历史 42.20 与 45.8604 的身份、官方 split 教师污染和历史 test 非盲测限制均被如实区分。**104.90 GiB 可从原始图像尺寸完整复算，不建议删除精确数值；应注明它是未压缩 FP32 概率张量负载，不是实测磁盘占用。**

剩余的公平性限制是：旧 B0 的42.20为历史 best@32k，而不是可复核的本次40k checkpoint；v2 的 padding 有效域修正又改变训练目标，所以 40k 只能称相同图像批次/更新预算，不能说训练配方或计算完全相同。新鲜独立测试尚未指定。另，目录中的 `02_架构规格.json` 仍是 v1 契约，和 v2 有多处直接冲突；实现者若仍将它视为规范，会按错方案实现。

## 复核结果

### 40k 公平口径

v2 明确规定 40,000 个全局批次、40,000 次主 SGD，batch=2；H4 每个相同批次做一次 Adam，不额外取图或再跑 HRNet，并明说这不等于相同 FLOPs（`04:27,165-175`）。这支持“相同图像曝光/更新预算”的说法，不支持“等计算”。H4 现在从第1步随主网络训练，验证节点均包含已训练 H4；移除了 v1 中早期 best-val checkpoint 没有训练 H4 的歧义（`04:169-173`）。

历史本地 B0 的42.20%是自定义 454/76/227 划分上的两位小数报告，使用 val 选出的 32k 权重；完整训练虽到40k，但本次审计未找到该 checkpoint 或逐图 test 计数。它不是 HRDecoder 论文另一协议下的32.25%，v2对此区分正确（`04:29-30`；`HRDecoder_FinalDecision_20261005/结题报告/evidence/original_HRDecoder_baseline_evidence.json $.historical_test.inference_reported`, `$.baseline_checkpoint_found_in_this_audit=false`, `$.reported_split_sizes`）。所以应并列报告学生40k固定端点和 scheduled best-val；不要将学生40k对42.20描述成“同选点规则的复现比较”。

还有一项重要的配方差异：v2 要求新主损失、KD、H4 全部屏蔽源图外 padding，并要求公平对照也采用同一 `v_geo`（`04:181-183`）；历史 B0 则把旋转/补边标签记成背景0（`autodl100/work/HRDecoder/configs/_base_/datasets/hr_ddr_2048.py:31-35`）。v2已声明保留42.20的历史口径、不称重新复现，并规划带相同 mask 的公平对照（`04:182,231`）。因此只有重跑同一 mask/teacher 预算的 B0 对照，才能将改进归因于新架构；历史42.20可作工程参照。ImageNet W48 的“同文件”哈希仍待核验，v2明确写出这一点（`04:85`）。

### 原图教师缓存与几何

几何协议相较 v1 实质变完整：T0/T1/T2先在同一原图坐标构造，再应用同一学生几何；teacher soft probability bilinear、离散 GT nearest；先变换再算连续 KD 可信度；单独维护源图有效域并排除双线性采样支撑越过源图边界的目标像素，同时保留真实黑背景（`04:179-185`）。这与“transform teacher map 不等于重新在增强图上运行 teacher”区分一致，未发现教师图、标签和有效区的概念混淆。仍需把同一组变换参数及逐操作边界规则落实到数据流水线；这属于实现复现记录，不能将软图搬运称为增强图教师真值。

v2的train缓存要求也正确限制为自定义 train 的454张；明确禁止把 val cache 或 test count 当训练目标，并指出 train 预测是 in-sample、非 OOF（`04:179,187,189`）。现有证据仍没有证明这454张 T0/T1/T2 完整概率已经导出（`evidence/teacher_protocol_audit.md:46-51`）；正式训练前应先核实缓存 ID、shape、class order 和教师/代码/config/probability SHA，且不得把缓存扩展到 val/test。

### 104.90 GiB 复算

这次重新核对了原始 JPG 尺寸，而不只沿用旧估算。清单来源为 `D:\dataset\experiment\HRDecoder_FinalDecision_20261005\execution\server_results\reports\data_identity.json $.splits.train.samples[].{sid,hw}`。完成证据脚本核验图像/标签 SHA，并断言标签尺寸等于 `hw`（`HRDecoder_FinalDecision_20261005/结题报告/analysis/audit_completion_evidence.py:39-45`），但它本身没有读取 JPG 尺寸。本次另用 PIL 打开 `D:\HRDecoder\data\DDR_split_613\images\train\<sid>.jpg` 全部454个 JPG 头信息；与清单 `hw=[H,W]` 对比为454/454一致，0缺图、0尺寸不符。这补足了原脚本只核验 label shape 的部分。

复算公式：`Σ(H_i×W_i)=2,346,569,020 px`；T0/T1/T2各4类FP32为 `3×4×4=48 bytes/pixel`；故未压缩 tensor payload 为 `112,635,312,960 bytes = 112.63531296 GB = 104.89980966 GiB`，四舍五入 **104.90 GiB**。固定每张2048²时为 `454×2048²×48 = 91,402,272,768 bytes = 85.125 GiB`。故 v2 `04:187` 的两个数都正确。实际文件占用取决于容器、元数据和压缩，未测量；不要把104.90 GiB写成实测落盘量。

### 历史成绩与独立测试

45.8604%是自定义 test 227张上的完整融合教师 T2，非单模型结果；v2的递归式 T0=A（旧系统含 H4）、T1=P0、T2=最终系统与已锁配方一致（`04:38-47`；`evidence/teacher_protocol_audit.md:7,13-22`）。42.20%是本地历史 HRDecoder 报告，类序 EX/HE/SE/MA、全图像素累计、严格 `>0.5` 的评价契约与现有固定划分一致，但基线记录四舍五入、checkpoint/逐图 counts 不存，因此仍是历史参照，不是新复现（`04:29-30,182`；上列 baseline JSON 的 `$.comparison_grade`）。

v2正确声明历史 test 已多次参与研究反馈，不能称新盲测；官方 383/149/225 发表协议需隔离重建 teacher。之前核对的旧自定义 train 与官方 valid/test分别重叠80/142张（`04:189,225`；`evidence/teacher_protocol_audit.md:23-31`）。目前 v2 只把“外部隔离证据”列为未来发表需求，没有指明具体新数据集、患者级划分或独立评估锁定方案；因此现阶段只能报告固定历史 test 上的工程对比，不应宣称独立验证。患者级互斥也仍未证实。

### 版本一致性（实施前需处理）

设计根目录现有的 `02_架构规格.json` 仍是 v1：例如固定概率融合与学生分支权重（`02 $.student_fusion.weights`），旧的 `M=valid AND ((T>0.5)==Y)`（`02 $.loss.distillation_support`），H4仅最后1,600步更新及BN末段切换（`02 $.H4.active_global_updates_inclusive`, `$.training.freeze_original_BN_running_statistics_from_update`）。这些分别与 v2 的逐级logit残差、连续GT一致性KD、H4全40k且不切换BN直接冲突（`04:103-113,148-157,165-175`）。`04:3,5,34`已宣布v2取代v1，但若继续将旧 JSON 当可执行规格，协议会互相矛盾；尤其`04:210`又声称复核了文档/JSON/探针一致性，与当前顶层规格内容不符。此审阅不改 JSON；实施前应明确由 v2 文本取代旧规格，或另行维护与 v2 同步的规格文件。

## 结论分级

- **已核实正确：** 104.90 GiB的尺寸和字节计算；85.125 GiB固定2048估算；40k批次/主更新与40k H4更新的计数；45.8604、42.20的指标身份区别；teacher train-only 和官方协议隔离声明。
- **文档已承认、不可越界解读：** 40k并非等计算；历史B0不是checkpoint级复现；teacher搬运目标不是增强图教师推理；历史test非独立盲测。
- **仍缺执行资产/规范：** 454张train软概率缓存；实际同起点文件SHA；新独立测试/患者级隔离证据；与v2一致的机器可读规格。
