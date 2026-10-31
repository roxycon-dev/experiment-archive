# runs 历史实验与机制证据

审查范围是 `D:\dataset\experiment` 下的运行配置、日志、评估记录、停止/恢复记录及结果表；目标是核对当前单个 ImageNet HRNet-W48、40k 更新、global batch 2、454/76/227、融合标签协议的架构设计证据。未启动训练/SGD，未连接服务器，未读取任何原始图像或标签像素，未加载或执行 checkpoint/pickle。Archive 中 H0 的小型 metadata 包只流式读取了文本日志、CSV、JSON 和停止码。

## 覆盖与解析方式

`runs_coverage.json` 是机器覆盖清单；`runs_file_hash_inventory.json` 记录逐文件 SHA 去重。5,526 个文本类物理文件合并为 3,813 个唯一 SHA 组，重复拷贝 1,713 份。502 个 `.log`、12 个 `.err`、3 个 `.out` 共 517 个文本日志文件逐行扫描：337,720 条物理行、285,801 条去重行，抽取 82,559 个结构化事件/指标记录。提取是全文件结构化扫描；没有把全量日志声称为逐句人工阅读。

3,942 个 JSON 文件按唯一 SHA 解析。2,634 个唯一内容是标准 JSON；3 个 `.log.json` 内容实际为 JSONL，恢复后合计 1,829 条记录：`D_0913_retry` 1,011 条（1,000 train、10 val、一条环境记录），`trial_a` 和 `trial_b` 各 409 条（400 train、8 val、一条环境记录）。余下 1 个标准 JSON 无效 SHA 是空字节文件，重复出现在 22 个物理占位符中，不能据此推断运行是否发生或结果如何。另有 69 个 `.jsonl` 文件，按 SHA 去重后 41 组、4,483,322 条记录；连同上述恢复日志，共 72 个物理 JSONL 流、44 个唯一 SHA、4,485,151 条记录。

68 个 CSV 中 63 个唯一 SHA，结构化读取 50,940 行；11 个无表头遥测文件按记录处理。205 个 YAML/YML 中 95 个唯一内容均经 safe-load 读取。114 个 Python 配置候选只做 AST 读取，均未 import 或执行。另 2 个 TOML 与 2 个 CFG/INI 文件已全文读取。`runs_validation_nodes.csv` 给出 11 个架构运行的 34 个验证节点以及 EX/HE/SE/MA 的 IoU、TP、FP、FN；`runs_experiment_table.csv` 保留各节点的完整 mIoU 与逐类 IoU 轨迹；`runs_historical_experiments.csv` 单列历史基准、H0、H4 和产品融合，避免把它们混成 W48 结果。

60 个委派 TXT 的 SHA 清单见 `delegation_split_20261016.csv`。本审阅者负责的 30 项均已全文人工阅读；`remaining_materials_luna` 负责的 30 项中，29 份完整文件全文阅读，另 1 份文件读到物理 EOF，但自身在 Python 表达式中途截断，无法重建缺失尾部。故 59 份完整正文已人工全文阅读，另 1 份只可能核验其现存全部字节。60 项中没有与旧 Markdown 完全同 SHA 的重复副本。状态分别记在 `runs_delegation_read_status_historical.csv` 和 `delegation_coverage_remaining.csv`。未读凭据文件 `HRDecoder_Next_20261003/ops/ssh_public_key.txt` 被明确排除。代码与压缩包源成员由 source research 另行核对；35 个 archive 的目录/hash 纳入 inventory，但没有为读取数据像素而展开大型数据/资产包。

## 纯 W48 基线与产品目标不是同一类比较

目前可核验的最强纯 HRNet-W48 单模型 test 参照是 B0：ImageNet W48、2048 输入配置、SGD lr 0.005、Dice、batch 2、40,000 次更新；最佳 val 在 32k（39.390%），test 使用该 32k checkpoint。test 42.20% 与 EX 51.56%、HE 43.13%、SE 49.98%、MA 24.11% 都是历史报告的两位小数。原始 B0 checkpoint 与逐像素 TP/FP/FN 未找到，所以表中保留 NA；不能把 round 值伪装成精确重算。数据审计确认当前固定 454/76/227 分配和报告所述分配一致，但 B0 不是本轮重训。

H0 是另一个纯 W48 的部分运行：metadata tar 中有 3k/6k val，6k 时 val mIoU=36.922642%；6k 验证的 EX/HE/SE/MA IoU 为 41.8771/43.5195/41.4545/20.8394%。TP/FP/FN 依次为 EX 146435/58875/144368、HE 678491/353125/527433、SE 105484/53053/95920、MA 21540/34078/47744。`exit_code=143` 且正式更新停在 6k；配置意图中计划 12k，没有 test。它的有效 batch 为 8，训练裁块/优化器/冻结 BN 与 B0 也不同，因此不能用作同预算对照。

当前用户给出的融合系统验证/测试参考为 45.321831227388% / 45.860383832668%。精确 test count 审计显示 P0=44.780891% 及最终锁定融合=45.860384%：最终对 P0 的逐类 IoU 差为 EX +0.524、HE +0.996、SE +2.795、MA +0.0035 个百分点。P0 counts（TP/FP/FN）是 EX 1072861/390087/502080、HE 1609384/506944/1444094、SE 354891/138077/215627、MA 80222/84314/109821；最终融合是 EX 1084309/392193/490632、HE 1623523/460808/1429955、SE 363816/117499/206702、MA 78485/78341/111558。最终配方是 0.75 P0 + 0.25 S20k EMA，P0 本身亦含异构模型，所以这是产品系统目标，不是纯 W48 基线。对 P0 的差额主要落在 SE、其次 HE，而不是 MA；不能据此断言单 W48 要用哪条分支。

完整核对表见 `runs_historical_experiments.csv`。其中 `D_old` 的 42.576398% 是局部 student 与冻结 B 参考融合的分数，不是一个 W48 输出；H4 是叠加到冻结 D+reference 系统上的 MA-only head；最终产品融合、P0、D_old、H4 全都不是纯 W48 对照。

## W48 训练轨迹中的反例和崩塌

ArchitectureDesign 目录的所有正式实验曲线均来自固定标签/划分与 2048 canvas 的 W48 任务线，但许多是 fresh ImageNet initialization、短 screen 或不同 loss/梯度路由；它们不能替代同一原始 B0 训练配置的对照。`runs_experiment_table.csv` 包含 11 个 arm 的每一条验证节点，而不是只取首尾：

- 完成 40k 的 CET：29.0527、32.8091、37.6001、38.7279、37.9997、36.2690、38.7222、40.4876、37.8112、38.3138（4k 至 40k）。没有 total collapse，也没有 test；最佳 32k 仍低于本次 val 参考。该 run 使用额外 GT/KD/aux 目标，不能把“能跑完”归因于某单一架构机制。
- 旧 arm `v3`（名称与 2026-10-16 NativePyramid 方案不同）val 为 29.4859、37.9882、36.9134、34.9849、15.0509（4k 至 20k），实际停止更新数 21,930。20k 时 EX 与 SE 为零检出：TP/FP/FN 分别为 EX 0/0/290803、SE 0/0/201404；HE 41.8451 IoU、MA 18.3585 IoU。其 40k 占位 JSON 为空，停止记录显示用户在崩塌后停止，没有 24k 之后的验证或 test。
- `detail` val 为 32.0338、34.9232、36.7358、13.0748、15.2038（4k 至20k），停止更新约21,850。原始 `detail_execution/validation_series.json` 的4k节点包含同一checkpoint的private/base head计数：MA private TP/FP/FN=21810/63322/47474（IoU16.4472%），base=25245/1326940/44039（IoU1.8081%）；private少3435 TP、多3435 FN，但少1263618 FP。此为同checkpoint双头输出对照，不是独立训练基线；它显示MA IoU增益主要来自压低假阳性，并伴随召回下降，不能归因于分辨率本身。16k 时 SE TP=0/FN=201404；20k 时 EX、SE 均 TP=0 且所有真阳性像素漏检。HE 与 MA 仍有预测。
- `SACD` 是当前四尺度直达类别预测想法的近邻，而不是同一结构：旧实现是每尺度 128→32 的非线性 action readout，含共享 L/G/teacher 路径。val 为 27.8256、33.4699、36.8538、31.4726、37.1848（4k 至20k），20k 时 EX/HE/SE/MA IoU 为 35.2282/41.5563/51.1930/20.7617。其曲线中有 HE FP 峰值：16k HE FP=1,881,983，20k 降至417,270。此 run 只到20k，无 test，不证明四个独立线性 1×1 类别读出失败或成功。
- `adapter` 在4k mIoU 30.4442、同一节点 base mIoU 26.8082；`semantic` 为32.8890、base33.2550；`residual` 为29.4758@4k/37.2124@8k；`aligned`、`foundation`、`iou`、`expert` 只跑短 screen。均不能称为40k检验，也没有一个等同当前 v3 的隔离对照。

这些轨迹支持“W48/2048/全局 clip/某个私有损失的组合有时发生晚期类别性崩塌”，不支持“高分辨率本身必然损坏 HE/SE”“零初始化本身必然崩溃”或“冻结 BN 是原因”。

## 高分辨率、零初始化、损失和 BN：因果证据边界

1. **分辨率与类间 tradeoff。** B0 在2048设置下达到历史纯 W48 42.20%；同时，fresh W48 2048 screen/训练里存在高分早期节点和后期 EX/SE 零检出。Detail 4k存在同checkpoint private/base双头对照：四类IoU都提高，MA改善最大；HE/SE的TP略有下降而FP减少。它证明该private输出头在该节点的错误结构，不是独立训练或分辨率消融；不能把变化归因于high-res stem/detail。崩塌表象是EX/SE全类FN，而不是可解读为只损边界/内区；没有空间分区计数可支持interior机制。
2. **零末层与 branch start。** `detail`、旧 `v3`、SACD 都有零初始化或近零初始化路径；旧 v3/Detail 后段仍退化，故零初始化不是充分的稳定保证。它可能保护初始预测，但第一次梯度是否通过零末层、支路是否有逐类响应，需实际梯度/参数轨迹。当前没有失败 update 的 pre/post global-clip 模块梯度占比，所以“全局 clip 竞争”只是猜测。没有成功/失败的更新级梯度竞争记录。BASE_GRADIENT_DIAGNOSTIC 是 12 个 train-only batch、正式 optimizer updates=0：39 个正类 batch 例中16个当前阈值零 recall，饱和 Dice 正 logit 梯度约1e-13–1e-11，而平衡 BCE 正梯度约1e-6–1e-5。它说明局部梯度尺度差异，不能证明分支失败或崩塌由负例压力导致。
3. **固定 BN。** `BN_CROSSOVER_COMPLETE.json` 是状态诊断，不是 optimizer 对照：W12/BN12=36.73585、W21/BN21=16.36646、W21/BN12=0.14957、W12/BN21=4.28315；另报告 `strong_buffer_mismatch_support=false`，不支持把 running-buffer mismatch 当成强根因。B0 用 `norm_eval=false` 仍取得成功分数，故固定 BN 并非已证必要条件；E4→B 还同时改变了 low-LR、frozen BN、MA auxiliary，测试41.918低于B0，不能把任何部分单独归因。没有历史证据证明新 head channel-LN 会修复崩塌。
4. **Dice/OHEM。** B0 的确切历史强结果使用 Dice。当前 v2 的 OHEM/positive-negative balancing 组合没有匹配40k成功对照；IOU screen 同时换成 Lovasz+BCE 且明显退化，不构成 OHEM-only 反证。局部梯度诊断提醒 Dice 对高置信正例可变得很小，但没有更新级因果验证。v2 loss可为了让结构对照少变一个因素而暂时保留，不能描述为已验证优于 Dice-only。
5. **2048与训练协议。** 2048 在 B0 可工作，也不足以保证 CET、旧 v3、Detail、SACD 达到目标或稳定。H0 的2048/1024-crop、effective batch8、12k planned/6k stopped；CET/旧 v3 的40k计划、batch2、custom loss；B0 的全程40k、batch2、原Dice；D_0913 的 B@3k warm task weights、20k low-LR、batch2与 reference fusion。把这些 score 排成架构排行榜会混淆训练预算、初始化、loss、推理系统和选点。

## 可用的正向机制与可迁移边界

- **最可靠 W48 先验仍是简单 W48 控制架构与训练协议本身**：B0 已证实四尺度 HRNet-W48 + 原 FCN decoder、ImageNet init、2048 输入配置、Dice-only、40k batch2 可以达到42.20 test。原 checkpoint 与逐类 test count 缺失，无法从 B0 的报告反推出哪里尚有多少可学增益。新方案应把这套作为最小参考/复现基线，而不是将它描述为无多尺度语义。
- **H4 的 MA-only 校正有真正局部的正向证据**：固定的 D+reference 输出被冻住，仅校正 MA。227 test counts 上，MA 从 TP/FP/FN=69270/90073/120773 到80641/96545/109402，MA IoU 24.7290→28.1383；EX/HE/SE count tuples bit-identical，系统 mIoU42.5764→43.4287。训练 head 约2k，dev val 22.617→25.221@1.6k→25.080@2k。这个证据支持“固定预测底座上的窄 MA 校正可以改善 MA，伴随更多 MA FP”，不支持在线修改 W48 主干、不支持同40k单模型增益，且 test 已被历史使用。
- **一次很短的 teacher-KL负/正样本归一化 probe** 在相同设置的500更新对照中，stratified arm mIoU 比 control 高1.632pp，但 MA precision 约-3.13pp、recall约+11.26pp，没通过预设“精度提高且召回不降”门槛；它说明局部 tradeoff，不适合直接搬到新主模型。
- **旧决策/日志中 D_0913_retry 是真实 formal 20k，不是“所有 plus3 没有训练”。** 原始 summary/state 与可恢复 JSONL 相符：best val41.615%@2k，final40.476%@20k，未满足连续两节点42.4 gate，无 test。它从 task-trained B@3k 初始化，使用 frozen BN 并在 inference 融合冻结 whole-image B；因此是一个 W48-derived student/reference 系统，不是本任务所限的 ImageNet-only W48。早峰后滑落反对将早期高点当稳定增益。

## 直接用于当前设计决策

历史没有运行过完全相同的“F1–F4 各自无 bias 1×1→四类、上采样到同一个 S 再求和”的 W48-only 分类头，也没有运行过“新 head channel-LN 替代 GN、固定 backbone BN、保留 v2 loss、无 RGB、PixelShuffle 零残差”的整个组合。因此这个候选是可测试假设，不能写作已支持的提升。四尺度 feature 早在旧 HRDecoder 720通道输入就已有；当前直接路由的动机只能说改变梯度/分类路径，并非补上从未存在的尺度信息。

保留结构的最低风险理由：单一 ImageNet W48；减少新分支数；主路直接受监督；附加残差从零开始；实验训练/损失/40k预算锁定。风险仍包括语义和最终头的双重监督改变梯度、四条直接 classifier 的尺度间 logit 标定、零末层使内部残差在初始阶段无任务梯度、PixelShuffle 相位残差后续幅度失控、通道 LN 未实证、固定 BN 的域偏移、OHEM negative pressure、全局clip与两损失头的梯度组合。应把每项作为启动前结构验收或训练中的观测量，不应提前承诺能过45.3218/45.8604门槛。

源码是否真实运行、原始论文和各自实现身份请配合 source research 的代码 SHA/source manifest；此处 `protocol.json` 是运行身份与超参记录，目录中存在某个候选 Python 文件本身不代表它已经在相应 run 中执行。
