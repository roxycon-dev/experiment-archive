# 教师输出契约与协议公平性审计

审计对象：`D:\dataset\experiment\HRDecoder_FinalDecision_20261005` 的最终报告与证据、原始 B0 证据和日志，以及 Ready、Next、NextPlan、Q 归档中的最新有效结果。只查阅报告、清单、JSON 和文件元数据；未读取权重张量，未训练、未推理，未改动历史实验。遵循 `D:\dataset\AGENTS.md` 的数据目录说明。

## 结论与教师目标

最终 45.8603838%（报告四舍五入为45.8604%）是227张固定 test 上的**概率融合系统**成绩，不是单模型成绩。锁定模型为 S@20k EMA，S 单支只有 val 42.2703%；固定融合 val 为45.3218%。test 四类 IoU 为 EX 55.1213%、HE 46.1978%、SE 52.8789%、MA 29.2435%。逐图 test 计数、类别计数及配方见 `D:\dataset\experiment\HRDecoder_FinalDecision_20261005\结题报告\evidence\final_locked_test_evidence.json` 的 `$.locked_test_miou_pct`, `$.test_per_class`, `$.gain_vs_R_pp`, `$.conditional_bootstrap_from_original_final_test`；正文见 `D:\dataset\experiment\HRDecoder_FinalDecision_20261005\结题报告\HRDecoder项目结题报告.md:7,15`。

教师概率的固定分解可直接按三个输出目标表达：

```text
T_final = 0.75 P0 + 0.25 T_S20
P0      = 0.75 T_old(H4后) + 0.25 T_U0
        = 0.5625 T_old(H4后) + 0.1875 T_U0 + 0.25 T_S20
```

因此用户提出的 `T_old(H4后) / T_U0 / T_S20` 和 `.5625/.1875/.25` 是报告配方的等价概率空间展开。`T_old(H4后)` 仍是历史 HRDecoder 集成及后融合 MA 修正，不是一个权重文件；`T_U0` 是独立 ConvNeXt-S；`T_S20` 是 DINOv3 空间交互分支。配方出处：`D:\dataset\experiment\HRDecoder_FinalDecision_20261005\最佳模型技术报告\报告正文.md:77,81-93,101,105,227`；最终锁定字段见 `D:\dataset\experiment\HRDecoder_FinalDecision_20261005\execution\server_results\reports\final_selection_lock.json` 的 `$.arm`, `$.update`, `$.formula`, `$.checkpoints.42.sha256`, `$.inference`。

## 数据、指标与公平性边界

本项目固定自定义 split 为 train/val/test = 454/76/227，assignment SHA256=`79181851e8026125bf900b6d1129ca7946b6cda56dfdb5708b27364b1173ea9f`。DDR 标签索引为 `0=背景, 1=EX, 2=HE, 3=SE, 4=MA`；输出通道为 `EX/HE/SE/MA`，背景不计入宏平均。预测规则是 FP32 sigmoid 概率对齐原图后固定权重融合，以严格 `>0.5` 二值化，按全数据像素累计四类 TP/FP/FN 后等权平均 IoU。B0 使用2048画布并在HRNet前缩放到1024；S使用原生1024窗口、中心768、halo128、RGB ImageNet归一化。证据：`D:\dataset\experiment\HRDecoder_FinalDecision_20261005\结题报告\HRDecoder项目结题报告.md:55,63,65,90-98`；`D:\dataset\experiment\HRDecoder_FinalDecision_20261005\execution\server_results\reports\final_selection_lock.json` 的 `$.inference`。

**旧自定义 teacher 不满足官方 383/149/225 协议的隔离条件。** 我将本轮冻结自定义身份清单 `D:\dataset\experiment\HRDecoder_FinalDecision_20261005\execution\server_results\reports\data_identity.json` 中 `$.splits.{train,val,test}.samples[].sid` 与 `D:\dataset\lesion_segmentation\train\image`、`valid\image`、`test\image` 的文件名 stem 做了只读交集计数。结果如下：

| 自定义 split | 官方 train | 官方 valid | 官方 test |
| --- | ---: | ---: | ---: |
| train（454） | 232 | 80 | 142 |
| val（76） | 45 | 11 | 20 |
| test（227） | 106 | 58 | 63 |

官方图像数分别为383/149/225；自定义划分脚本明确将原集合合并后再按 seed 42 重分为454/76/227，见 `D:\dataset\lesion_segmentation\01_板块一_数据集划分脚本_ddr_split.py:4-18,23-30`。旧教师在自定义 train 上训练，故其训练中见过142张官方 test 和80张官方 valid；不能把这些模型或其输出用于宣称官方 split 的无泄漏比较。若后续要报告官方协议结果，教师也必须在官方 train 383 范围内隔离训练，旧自定义 train teacher 不能跨协议复用。图像 ID 交集不提供患者级互斥证据。

`42.20%` 是同一自定义454/76/227 split 上的历史 HRDecoder 测试报告值，不是论文原始 DDR mIoU；原论文 Table 1 的 HRNet 版 mIoU为32.25%、mF为48.21%；其它复现记录中的47.81%不可混称该论文表值。B0 日志显示完整训练40,000步、最佳val为32,000步；最终测试使用val选出的32k checkpoint。checkpoint、逐图 test counts 和历史样本哈希快照在当前审计中均未找到，不能称为本次复现。证据：`D:\dataset\lesion_segmentation\实验日志_json\01_板块一_baseline.json:328-329,409-410`；`D:\dataset\lesion_segmentation\01_板块一_baseline实验配置.py:55-58`；`D:\dataset\lesion_segmentation\01_板块一_划分统计报告_baseline_report.md:30-42`；`D:\dataset\experiment\HRDecoder_FinalDecision_20261005\结题报告\HRDecoder项目结题报告.md:33,35,57,59`；`D:\dataset\experiment\HRDecoder_FinalDecision_20261005\结题报告\evidence\original_HRDecoder_baseline_evidence.json` 的 `$.reported_split_sizes`, `$.historical_test`, `$.baseline_checkpoint_found_in_this_audit`, `$.historical_test.per_image_counts_found_in_this_audit`, `$.split_identity_and_evaluation_code_match_to_current.comparison_grade`。因此“同起点、同40k”应按完整训练预算比较，不能把B0最佳checkpoint说成40k权重。按当前用户限定，**学生初始权重仅为 ImageNet HRNet**；教师既有的其他预训练来源不转作学生初始化。

## 教师训练来源及预训练

| 输出目标 | 已确认的训练谱系 | 预训练与外部监督 |
| --- | --- | --- |
| `T_old(H4后)` | D/E/B/E4 历史成员按3:2:1:1及翻转TTA融合，随后 H4 仅修正 MA。分支预算/所用节点分别为：B0 40k/32k；E4 40k/36k；B续训8k/3k；D_old 20k预算/2k；E_old 6k预算/1.5k；H4头选1.6k。它们是有分支继承关系的历史模型，不能加成一个教师总步数。 | HRNet-W48 ImageNet预训练；使用固定自定义 DDR 划分。 |
| `T_U0` | ConvNeXt-S U-Net，在DDR训练12k；独立 val 38.4464%；以25%权重进入P0。 | ImageNet初始化；最终U0无IDRiD/MAPLES/TJDR像素监督。 |
| `T_S20` | Q_DDR@16k → F12追加12k → S空间交互@20k EMA；任务谱系累计48k更新，但不是与B0同预算比较。 | DINOv3 ViT-B/16，LVD1689M通用预训练；最终S链没有Q_MIX、IDRiD、MAPLES或TJDR像素监督。预训练身份见 `D:\dataset\experiment\HRDecoder_Q_20261003\server_results\reports\pretraining_identity.json` 的 `$.pretraining`, `$.sha256`, `$.bytes`。 |

主要训练来源详见 `D:\dataset\experiment\HRDecoder_FinalDecision_20261005\最佳模型技术报告\报告正文.md:33,49,57,61,73,77,81-93,99-105,123-127,181,185-188,221-227`、`D:\dataset\experiment\HRDecoder_FinalDecision_20261005\结题报告\HRDecoder项目结题报告.md:112,188,196,218,228,230,486` 和历史 ledger `D:\dataset\experiment\HRDecoder_FinalDecision_20261005\结题报告\evidence\historical_experiment_ledger.json:1588-1616,1619-1668,1700-1750,1786-1845,1952-1974`。U0的ImageNet初始化也见 `D:\dataset\experiment\HRDecoder_Ready_20261002\reports\HRDecoder_完整实验报告_20261003.md:45`。旧成员 checkpoint 的文件名、哈希、字节数记录在 `D:\dataset\experiment\HRDecoder_Ready_20261002\best_pipeline\local_weight_manifest.json` 的 `$.weights`；S20权重文件为 `D:\dataset\experiment\HRDecoder_FinalDecision_20261005\execution\server_results\runs\S_seed42\checkpoint_u20000.pth`，锁定SHA见上文。以上仅核验目录和metadata，没有读取大权重。

## train 概率资产、缓存与仍缺项

**目前没有证据表明454张自定义 train 的三路 teacher 概率已导出。** 现有 `D:\dataset\experiment\HRDecoder_FinalDecision_20261005\execution\server_results\runs\S_seed42\cache.json` 中 `$.P0.images=76`、`$.P0.float32_original_probability=true` 是 P0 的val概率缓存；同文件 `$.training.images=893`、`$.training.storage="RGB uint8 + mapped label uint8; read-only pages shared by fork"` 是图像/标签读缓存，不是train soft target。可见的 S20 图是 `D:\dataset\experiment\HRDecoder_FinalDecision_20261005\execution\server_results\runs\S_seed42\val_u20000\` 下76张val `.npz`。test概率未保留：`D:\dataset\experiment\HRDecoder_FinalDecision_20261005\结题报告\evidence\nontraining_server_work_plan.json` 的 `$.server_work_after_global_ack_only[1].source_note` 和 `$.reviewed_implementation_evidence["reports/single_seed_test.py"].test_probability_retention=false`。项目完成状态 `D:\dataset\experiment\HRDecoder_FinalDecision_20261005\结题报告\evidence\completion_audit_status.json` 的 `$.new_inference_or_training=false`。

因此，现有 teacher 权重及推理实现支持后续对**自定义 train 454 IDs only**导出概率，但输出资产仍待生成；不可把val缓存或test整数计数当训练缓存。应保留三路概率的ID集合、通道顺序、原图尺寸、teacher权重与代码SHA、推理配置SHA，并明确这些是教师在其训练图像上的in-sample软目标，不是独立泛化预测。若转为官方383/149/225协议，须另行保证teacher训练只见官方train；上面的交叉表说明旧teacher不符合该条件。

尚未证实的事项：B0历史 checkpoint及逐图test计数；历史B0逐样本SHA；患者级独立性；DINO LVD预训练与DDR样本级重叠；三路train概率图及其ID/shape校验。

## 可配对结果与负结果（保持数据集与指标身份）

- **高分辨率，Ready matched val：** Ready配对记录在共同1k节点，固定融合由43.3663%到43.7328%（+0.3665pp）；共同2k由43.3383%到43.3329%（−0.0054pp）。机器记录：`D:\dataset\experiment\HRDecoder_Ready_20261002\round2_highres\reports\paired_validation_observation.json` 的 `$.matched_nodes`。
- **高分辨率，Next G：** Next报告中的U0高分辨率固定25%候选val为43.8679%，锁定test为44.6345%；对照旧P0 test 44.7809%低0.1464pp，未取代旧赢家。该G为不同评估身份，不与Ready的matched val配对混算。来源：`D:\dataset\experiment\HRDecoder_Next_20261003\reports\HRDecoder_严格实验最终报告_20261003.md:3,5-8,21-23`。
- **额外监督与微调，NextPlan/Q：** Q_MIX曝光记录 `D:\dataset\experiment\HRDecoder_Q_20261003\server_results\reports\stream_Q_MIX_seed42.summary.json` 的 `$.exposures` 显示24k内DDR窗口曝光低于Q_DDR；NextPlan E_NEG24k单分支37.0094%、固定融合44.1643%，F12在追加12k节点为40.2418%单分支、44.5888%固定融合。外部数据、DDR曝光、额外头和计算未被分离，不能将结果归因于“外部数据本身有害”。来源：`D:\dataset\experiment\HRDecoder_NextPlan_20261004\execution\server_results\reports\最终实验报告.md:16,20-29`；`D:\dataset\experiment\HRDecoder_FinalDecision_20261005\结题报告\HRDecoder项目结题报告.md:228,230`。
- **空间交互，FinalDecision matched val：** 同父权重、DDR曝光和日程下，S−C单分支在20k/32k分别+1.2459/+1.0451pp；固定25%融合仅+0.1456/+0.0974pp。D−C单分支+0.8838/+0.5062pp，但融合−0.0753/−0.0904pp。它说明single增益不能替代融合系统收益。来源：`D:\dataset\experiment\HRDecoder_FinalDecision_20261005\结题报告\HRDecoder项目结题报告.md:287,293,299-304`。

FinalDecision报告明确项目历史 test 反馈已被多阶段实验使用；锁定配方阻止本轮 test 驱动重选，但不构成新盲测。最终相对R的+0.1388pp条件paired图像bootstrap区间为−0.2626至+0.6519pp，且不含种子方差。出处：`D:\dataset\experiment\HRDecoder_FinalDecision_20261005\结题报告\HRDecoder项目结题报告.md:7,106`；`D:\dataset\experiment\HRDecoder_FinalDecision_20261005\结题报告\evidence\final_locked_test_evidence.json` 的 `$.conditional_bootstrap_from_original_final_test`。

