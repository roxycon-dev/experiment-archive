# 融合目标与原 HRDecoder 对照证据

本审计复用 review_20261016 与 FinalDecision 的已完成运行审计，只补查目标配方、逐类记录和本地 completion 归档；未训练、未连接服务器、未加载权重或重算像素。

## 目标成绩与系统组成

45.321831% 是 S_seed42@20k EMA 在 76 张 DDR val 上的 fixed_fusion；45.860384% 是同一锁定配方在 227 张历史 test 上的一次系统评估。公式为：

P_final = 0.75 × P0 + 0.25 × S_EMA20k  
P0 = 0.75 × C_old + 0.25 × U0

总式可展开为 0.5625 C_old + 0.1875 U0 + 0.25 S，但 C_old 含学习型 H4，不能当作纯线性四模型集成。

- C_old：D_old、E_old、B、E4 概率按 3:2:1:1 加权，成员做水平翻转平均，再由 H4 对 MA 通道作学习型非线性残差修正。H4 不是固定标量权重。
- U0：独立训练的 ConvNeXt-S / U-Net 类模型；单模型 val mIoU 38.4464%，没有独立 test 授权结果。
- S：DINOv3 ViT-B/16 通用预训练 → Q_DDR16k → F12 追加12k → S 空间交互20k EMA。task 父链约48k更新，阶段间可训练范围、优化器和窗口曝光不同；S 没有 TJDR 像素监督。
- P0 test 为44.780891%；最终系统较 P0 高1.079493 pp。该差值是新增分支接入产品系统的结果，不是 W48 单模型或单一空间模块的独立贡献。

| 类别 | 目标 val TP/FP/FN | val IoU | val Precision | val Recall |
|---|---:|---:|---:|---:|
| EX | 185,859 / 78,183 / 104,944 | 50.3702% | 70.3899% | 63.9123% |
| HE | 695,246 / 337,679 / 510,678 | 45.0405% | 67.3085% | 57.6526% |
| SE | 140,247 / 32,959 / 61,157 | 59.8418% | 80.9712% | 69.6347% |
| MA | 26,296 / 31,719 / 42,988 | 26.0349% | 45.3262% | 37.9539% |

| 类别 | 目标 test TP/FP/FN | test IoU | test Precision | test Recall |
|---|---:|---:|---:|---:|
| EX | 1,084,309 / 392,193 / 490,632 | 55.1213% | 73.4377% | 68.8476% |
| HE | 1,623,523 / 460,808 / 1,429,955 | 46.1978% | 77.8918% | 53.1696% |
| SE | 363,816 / 117,499 / 206,702 | 52.8789% | 75.5879% | 63.7694% |
| MA | 78,485 / 78,341 / 111,558 | 29.2435% | 50.0459% | 41.2985% |

test 配方在评估前已锁定，但该历史测试集在项目多个阶段被使用，不是新的独立盲测。

## 原 HRDecoder B0 的同口径边界

B0 是单一 HRNet-W48 + HRDecoder FCN：ImageNet W48、2048 输入画布、SGD lr .005、Dice、batch 2、40k 更新。best val 在32k，报告 mIoU 39.390%；历史 test 42.20%使用该32k best-val checkpoint。四类 test IoU 只保存到两位小数：EX 51.56%、HE 43.13%、SE 49.98%、MA 24.11%。没有 B0 原始 checkpoint，也没有逐类 val 或 test 的 TP/FP/FN，不能补造。

B0 配置与划分报告指向自定义 DDR_split_613 的454/76/227；指标同为原图像素、sigmoid严格大于.5、累计四类全局 TP/FP/FN 后宏平均、背景排除。当前审计确认数据划分与文档定义，但 B0 没有等同的历史图像/标签字节身份快照；故此为报告级同划分对照，不是checkpoint和历史标签的字节级复现。

相对 B0，融合目标 val 高约5.9318 pp、test高约3.6604 pp。这是单模型到多年累积异构系统的总差额，不能归功于新 S。C_old重导出 test44.2110%、P0 test44.7809%和上一参考R test45.7216%也都不是纯HRDecoder单模型成绩。

## 对本轮 W48 的逐类目标缺口

本轮 v3 Z@40k 主结果与锁定融合 val 仅作描述比较：

| 类别 | v3 Z@40k | 融合 val | 差额 |
|---|---:|---:|---:|
| EX | 46.6211% | 50.3702% | +3.7491 pp |
| HE | 40.9440% | 45.0405% | +4.0965 pp |
| SE | 49.6390% | 59.8418% | +10.2027 pp |
| MA | 20.4981% | 26.0349% | +5.5368 pp |

当前最大差距是 SE，其次 MA、HE、EX；不能把 B0→融合 test 的最大累计类别差距 MA，误当作当前 W48 到融合 val 目标的瓶颈。目标 val 相对 P0 的 IoU 差为 EX +2.6337、HE +2.5342、SE +2.7725、MA −0.6039 pp；目标 test 相对 P0 分别为 +0.524、+0.996、+2.795、+0.0035 pp。已见的系统增量主要在 HE/SE，MA test 净增近零。

## 归因证据

- 输入/尺度：B0在2048画布整图推理；S用原生1024窗口、中心768、halo128和ImageNet RGB normalization。最终系统还混合C_old历史局部/整图路径与U0原生整图输出。没有只改变输入尺度的同模型配对实验，不能把系统分差归给高分辨率或窗口。
- 推理：B0报告无TTA；C_old保留水平翻转平均和H4，之后以固定权重作原图尺度FP32概率融合，严格>.5。目标与B0的推理图不同，报告没有逐项隔离各操作的百分点贡献。
- 标签/额外监督：使用同名自定义DDR 454/76/227划分及四类指标；B0旧标签字节快照缺失。胜出S父链未用TJDR/IDRiD/MAPLES像素标注。D/SD使用去重后的439对TJDR TRAIN，但没有胜过S；这只能说明该监督包未胜出，不能推断额外病灶标注普遍无效。
- 权重/预算：B0是ImageNet W48起点、40k更新；最终P0冻结多个历史HRNet体系权重并加入独立ConvNeXt-S，S则有DINOv3预训练和Q/F12/S任务父链。融合系统还有各分支历史训练成本，不能与单个40k W48按更新数或FLOPs等同。
- 最近的匹配正证据：S20k−C20k单分支val +1.2459 pp，但固定25%融合仅+0.1456 pp；32k融合差+0.0974 pp。它支持当前DINO父链与预算下空间交互包改善验证、传到强伙伴后增益较小；不能外推为W48收益。H4冻结底座测试对照只改MA，MA IoU 24.7255%→28.6558%，其他类计数不变；这是窄MA后融合证据，test已复用，不是W48主干更新。
- 负向或混杂证据：E4含RGB细节支路、test约41.626%，低于B0的42.20%，但训练与模型变化不能隔离细节支路。U0高分辨率延续2k/4k固定融合val分别约−0.1214/+0.2451 pp；另一G记录test44.6345%，低于P0。身份/几何不同，不能拼成因果曲线。D/SD额外监督包亦未胜出。
- 新目标相对旧参考R的test增量仅+0.1388 pp，paired image bootstrap 95%区间−0.2626至+0.6519 pp且不含seed方差；不应称为显著系统突破。

## completion 日志归档

本地 completion_20261017 共30个文件（29 JSON、1 Markdown），未见ZIP/TAR/GZ等压缩包，也未见train.jsonl；本地完成包无归档可展开恢复该日志。这不证明训练代码没有写日志，也不证明远端没有副本。本次未连接远端；后续日志回收须单独核身份、覆盖和时间范围。

## 证据路径

- 目标val逐类记录：HRDecoder_FinalDecision_20261005/execution/server_results/runs/S_seed42/val_u20000/metrics.json 的 fixed_fusion 字段。
- 目标test、配方SHA和推理设置：HRDecoder_FinalDecision_20261005/结题报告/evidence/final_locked_test_evidence.json 与 execution/server_results/reports/final_selection_lock.json。
- P0成员与公式：HRDecoder_FinalDecision_20261005/execution/server_results/configs/p0_identity.json；P0 test计数：结题报告/evidence/test_p0/p0_manifest.json。
- B0成绩与身份缺口：HRDecoder_FinalDecision_20261005/结题报告/evidence/original_HRDecoder_baseline_evidence.json；复用审计：HRDecoder_NativePyramid_20261011/review_20261016/runs_findings.md。
- 谱系、输入/推理、配对正负证据：HRDecoder_FinalDecision_20261005/结题报告/analysis/report_source.md，evidence/four_arm_saved_validation.json 和 evidence/historical_experiment_ledger.csv。
- 本轮W48结果：HRDecoder_NativePyramid_20261011/review_20261017/results_findings.md、results_curve.csv。
- 日志归档清单：HRDecoder_NativePyramid_20261011/server_execution_20261016/completion_20261017/。
