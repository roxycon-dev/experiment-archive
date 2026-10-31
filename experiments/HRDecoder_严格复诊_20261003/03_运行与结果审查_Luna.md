# HRDecoder 运行与结果审查（Luna）

范围：只读核对完整报告、原始 eval JSON/CSV、完成摘要、campaign state、融合与锁定 test 聚合计数及文件清单。没有 SSH/服务器访问、训练、推理或读取 test 概率；test 仅从冻结 TP/FP/FN 重算。

## 总体判断

23 个 val/dev 评估节点均从原始 JSON 提取，CSV/JSON mIoU 匹配情况见文末。U1 两阶段和 H1 外部预训练正常完成；H1 DDR 实际跑到 12,000 updates 后按低优先级停止，最佳在 9,000；A0 在 6,197 更新边界由操作者停止，最佳 6,000。Round-2 U0 continuation 在 2,000 时因无配对收益停止，U0 high-res 完成 4,000，最佳均在 1,000。
历史 U0 是 native whole-image；U1 用 resize 2048 加 sliding window 1024/768。该几何差异混杂 U0/U1 的直接比较。

## 验证节点逐类指标

百分数；主训练段包含 6 张 dev 图的预训练评估节点与 76 张 DDR val 节点，round2为76张 val；每类单元格为 IoU / precision(PPV) / recall；类别顺序 EX、HE、SE、MA。TP/FP/FN 在紧凑 JSON 中逐节点保留。

| 运行 | split | update | mIoU | EX I/P/R | HE I/P/R | SE I/P/R | MA I/P/R | 推理几何 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| A0_ddr_finetune | val | 3000 | 6.95 | 17.07 / 25.87 / 33.42 | 0.01 / 0.05 / 0.01 | 10.71 / 15.43 / 25.94 | 0.00 / 0.00 / 0.01 | [2048, 2048]; sliding_window(window=1024,stride=768) |
| A0_ddr_finetune | val | 6000 | 9.68 | 24.14 / 37.00 / 40.97 | 0.00 / 0.01 / 0.00 | 14.58 / 21.95 / 30.25 | 0.00 / 0.00 / 0.00 | [2048, 2048]; sliding_window(window=1024,stride=768) |
| H1_ddr_finetune | val | 3000 | 32.09 | 37.16 / 50.08 / 59.03 | 42.98 / 55.29 / 65.87 | 33.31 / 49.71 / 50.25 | 14.91 / 22.79 / 30.13 | [2048, 2048]; whole_image |
| H1_ddr_finetune | val | 6000 | 33.34 | 38.93 / 59.40 / 53.04 | 41.54 / 69.02 / 51.06 | 36.82 / 72.31 / 42.86 | 16.06 / 25.62 / 30.10 | [2048, 2048]; whole_image |
| H1_ddr_finetune | val | 9000 | 35.60 | 39.61 / 54.25 / 59.48 | 43.10 / 61.90 / 58.66 | 42.11 / 60.91 / 57.70 | 17.56 / 29.14 / 30.64 | [2048, 2048]; whole_image |
| H1_ddr_finetune | val | 12000 | 35.40 | 40.39 / 55.04 / 60.28 | 40.62 / 61.83 / 54.21 | 43.22 / 67.02 / 54.90 | 17.38 / 27.38 / 32.25 | [2048, 2048]; whole_image |
| H1_external_pretrain | external_train_derived_dev | 300 | 4.68 | 4.24 / 4.29 / 80.94 | 14.38 / 35.83 / 19.36 | 0.00 / 0.00 / 0.00 | 0.08 / 0.08 / 2.34 | [2048, 2048]; whole_image |
| H1_external_pretrain | external_train_derived_dev | 600 | 26.24 | 39.05 / 78.78 / 43.64 | 20.78 / 52.09 / 25.69 | 36.79 / 48.26 / 60.74 | 8.34 / 11.01 / 25.58 | [2048, 2048]; whole_image |
| H1_external_pretrain | external_train_derived_dev | 1200 | 40.58 | 52.59 / 69.24 / 68.62 | 22.65 / 66.60 / 25.56 | 57.17 / 72.48 / 73.03 | 29.90 / 44.66 / 47.50 | [2048, 2048]; whole_image |
| H1_external_pretrain | external_train_derived_dev | 2400 | 43.00 | 55.78 / 74.66 / 68.81 | 25.38 / 64.62 / 29.47 | 57.13 / 69.41 / 76.35 | 33.72 / 50.01 / 50.85 | [2048, 2048]; whole_image |
| U0_continuation2048 | val | 1000 | 39.09 | 45.00 / 64.74 / 59.61 | 39.69 / 57.13 / 56.52 | 50.49 / 70.89 / 63.70 | 21.17 / 32.65 / 37.58 | whole_image |
| U0_continuation2048 | val | 2000 | 37.80 | 42.67 / 59.83 / 59.81 | 39.97 / 57.58 / 56.66 | 49.44 / 71.84 / 61.31 | 19.12 / 29.95 / 34.60 | whole_image |
| U0_highres | val | 1000 | 39.61 | 45.16 / 65.59 / 59.18 | 39.62 / 55.38 / 58.19 | 51.58 / 71.82 / 64.66 | 22.07 / 32.63 / 40.55 | whole_image |
| U0_highres | val | 2000 | 37.49 | 40.34 / 57.48 / 57.51 | 38.26 / 55.46 / 55.24 | 50.50 / 74.71 / 60.91 | 20.84 / 32.20 / 37.15 | whole_image |
| U0_highres | val | 4000 | 38.49 | 44.19 / 62.83 / 59.83 | 38.86 / 55.07 / 56.90 | 49.79 / 70.47 / 62.91 | 21.14 / 30.61 / 40.58 | whole_image |
| U1_ddr_finetune | val | 3000 | 24.80 | 27.23 / 32.17 / 63.97 | 35.73 / 51.11 / 54.28 | 24.70 / 36.21 / 43.73 | 11.53 / 14.05 / 39.22 | [2048, 2048]; sliding_window(window=1024,stride=768) |
| U1_ddr_finetune | val | 6000 | 31.50 | 34.72 / 41.87 / 67.03 | 38.23 / 56.43 / 54.24 | 37.17 / 53.53 / 54.89 | 15.87 / 19.99 / 43.54 | [2048, 2048]; sliding_window(window=1024,stride=768) |
| U1_ddr_finetune | val | 9000 | 34.28 | 40.50 / 50.13 / 67.84 | 34.08 / 48.37 / 53.58 | 43.97 / 63.24 / 59.06 | 18.56 / 26.92 / 37.41 | [2048, 2048]; sliding_window(window=1024,stride=768) |
| U1_ddr_finetune | val | 12000 | 36.69 | 43.04 / 55.63 / 65.54 | 36.80 / 52.82 / 54.82 | 46.99 / 66.01 / 61.99 | 19.94 / 28.82 / 39.28 | [2048, 2048]; sliding_window(window=1024,stride=768) |
| U1_external_pretrain | external_train_derived_dev | 300 | 1.83 | 3.98 / 3.98 / 94.67 | 3.33 / 3.44 / 50.80 | 0.00 / 0.00 / 0.01 | 0.03 / 0.03 / 7.00 | [2048, 2048]; sliding_window(window=1024,stride=768) |
| U1_external_pretrain | external_train_derived_dev | 600 | 8.92 | 14.84 / 15.38 / 80.70 | 13.95 / 18.07 / 38.00 | 6.41 / 6.88 / 48.52 | 0.48 / 0.49 / 40.93 | [2048, 2048]; sliding_window(window=1024,stride=768) |
| U1_external_pretrain | external_train_derived_dev | 1200 | 21.98 | 33.96 / 35.29 / 89.96 | 24.84 / 36.33 / 43.98 | 27.06 / 30.09 / 72.90 | 2.07 / 2.11 / 56.58 | [2048, 2048]; sliding_window(window=1024,stride=768) |
| U1_external_pretrain | external_train_derived_dev | 2400 | 27.16 | 41.34 / 44.58 / 85.06 | 25.85 / 43.23 / 39.14 | 38.37 / 45.57 / 70.83 | 3.07 / 3.15 / 55.71 | [2048, 2048]; sliding_window(window=1024,stride=768) |

## 初始化、批量与数据曝光

公共设置：AdamW，weight decay 0.01，5% warmup 后 cosine 至 base LR 的 0.01；seed 42、crop 1024、四通道独立BCEWithLogits+Dice等权。增强包括随机/前景混合 crop、尺度 0.5–2.0、翻转、±45°旋转和轻量颜色变化。DDR 为 454 train / 76 val / 227 test。外部 cohort 为 48 train 图；6 张从 IDRiD train source 派生的 dev 图使用用户 polygon mask，并非官方像素标注。
crop exposure = completion_summary.micro_steps × micro-batch；混合采样下它不是唯一图像数或 epoch。实际 micro-step 略多于 update×accum；代码计数包含 AMP 跳过优化步后的微批次。effective batch 相同并不令 microbatch Dice 的梯度完全等价。
| 运行阶段 | 初始化 | micro×accum=batch | LR（主干/编码器；head） | updates / micros | crop曝光 | 状态；最佳点 |
|---|---|---:|---|---:|---:|---|
| U1_external_pretrain | ImageNet ConvNeXt-Small, then external supervised pretraining | 2×4=8 | backbone 1e-4; head 3e-4 | 2400 / 9600 | 19200 | complete；best u2400 27.16% |
| U1_ddr_finetune | U1 external-pretrain best checkpoint; stage-local AdamW | 4×2=8 | backbone 1e-4; head 3e-4 | 12000 / 24016 | 96064 | complete；best u12000 36.69% |
| H1_external_pretrain | ImageNet HRNet-W48 + modified project HRDecoder; BN running stats frozen, affine trainable | 4×2=8 | backbone 1e-4; head 3e-4 | 2400 / 4804 | 19216 | complete；best u2400 43.00% |
| H1_ddr_finetune | H1 external-pretrain best checkpoint; stage-local AdamW | 8×1=8 | backbone 5e-5; head 2e-4 | 12000 / 12008 | 96064 | low_priority: best=0.355950, recent_gain=-0.001919, no_class_advantage；best u9000 35.60% |
| A0_ddr_finetune | random init; adapted HACDR with project 4-logit decoder; no ImageNet/external init | 8×1=8 | encoder 6e-5; head 9e-4 | 6197 / 6200 | 49600 | operator_stop_at_update_boundary；best u6000 9.68% |
| U0_continuation2048 | historical U0 complete-model checkpoint; fresh AdamW | 4×2=8 | backbone 2e-5; head 6e-5 | 2000 / 4008 | 16032 | paired_no_gain_or_learning；best u1000 39.09% |
| U0_highres | same historical U0 complete-model checkpoint; fresh AdamW | 4×2=8 | backbone 2e-5; head 6e-5 | 4000 / 8010 | 32040 | complete；best u1000 39.61% |

历史 U0：ConvNeXt-Small/U-Net、ImageNet 初始化、seed42、12k updates、24k microsteps、microbatch4（约96k crop presentations）；native-whole val 导出 mIoU 38.4469%，checkpoint 元数据 38.4464%，相差约 0.00045 个百分点。H0 仅有历史约6k结果，无可恢复 checkpoint，未重训。
Round 2 两臂均从 U0 历史完整模型 warm-start、fresh AdamW、batch8、LR 2e-5/6e-5；control 使用 long-side2048，high-res 随机2048/2560并有0.5–2.0尺度扰动。u=1k high-res固定融合较control +0.3665pp；u=2k差 −0.0054pp；仅验证集。control 首个u=1k因缓存API参数错误未成功；维修后同一配置/checkpoint在继续训练前重新完整评估76张。恢复后 DataLoader RNG 流不保证位级相同。

## 融合流程、候选偏差与缓存

val champion mIoU 42.6691%；U0 alone 38.4469%；固定 .75 champion + .25 U0 为 43.4877%（+0.8186pp）。完整val固定blend逐类IoU：EX 47.7365%、HE 42.5062%、SE 57.0692%、MA 26.6388%。U1固定blend 43.1056%（+0.4364pp）；H1 42.6425%（−0.0266pp）；A0 41.3958%（−1.2734pp）。候选提名先用全76张val的候选/固定blend表现，提名U0、H1；H1虽均值略负，仍可能凭单类gain/4的打分项入选。
三折 alpha cross-fit 本身按GT分层、seed42，25/25/26，patient_grouping=false；代码以两折选alpha、留一折评分且full_val_rescan=false。固定 P=.75 Pc+.25 Pu 只有 Pc∈(1/3, 2/3] 的像素可能被 Pu 改变阈值判定；若有 Pc 缓存，逐类理想上界可按 TP*=sum(y且Pc>1/3)、不可避免 FP=sum(~y且Pc>2/3)、IoU上界=TP*/(GT+FP) 计算。当前无本机连续 Pc 数组，不能给出实际 oracle 数值。OOF global 43.2483% 对 OOF base 42.6691%，+0.5791pp；classwise OOF 43.1422%。部署选择 U0/ConvNeXt global alpha .25，全val fixed blend 43.4877%。但候选和checkpoint均受全val筛选，因此该OOF不是独立外层验证。
本轮Ready_20261002备份树内没有连续概率或二值预测数组，也没有cache/val目录；融合JSON只留了压缩NPZ、FP32原坐标等缓存元数据、远程路径和部分聚合计数。根代理另外盘点整个D:\dataset\experiment的npz/npy/h5/hdf5：发现1901个旧FullCompetition互补分析/合同检查/临时测试文件，抽查真实champion缓存仅含packed二值mask、n_pixels、orig_hw，不能恢复连续Pc。因此仍无法重建本轮最终full-val固定blend逐类TP/FP/FN，亦不能算固定 .75/.25 的概率oracle。champion、U0单模和OOF global/classwise的聚合计数仍可用，见JSON。

## 锁定 test 计数复核

只从final_test.json锁定计数重算。旧/新mIoU 44.2110%→44.7809%，报告差值一致：+0.5699pp。总 ΔTP=+7679、ΔFP=-75352、ΔFN=-7679。
| 类 | old TP/FP/FN | new TP/FP/FN | ΔTP/ΔFP/ΔFN | IoU old→new | PPV old→new | recall old→new |
|---|---:|---:|---:|---:|---:|---:|
| EX | 1058087/389535/516854 | 1072861/390087/502080 | +14774/+552/-14774 | 53.86→54.60 | 73.09→73.34 | 67.18→68.12 |
| HE | 1603987/550006/1449491 | 1609384/506944/1444094 | +5397/-43062/-5397 | 44.51→45.20 | 74.47→76.05 | 52.53→52.71 |
| SE | 368502/169247/202016 | 354891/138077/215627 | -13611/-31170/+13611 | 49.81→50.08 | 68.53→71.99 | 64.59→62.21 |
| MA | 79103/85986/110940 | 80222/84314/109821 | +1119/-1672/-1119 | 28.66→29.24 | 47.92→48.76 | 41.62→42.21 |

解释：总FP下降75,352，净TP增加7,679。HE/SE改善主要由FP下降驱动；EX主要靠补TP（FP小幅上升）；MA两者皆有。final_test声明该split曾在历史实验查看，并非独立外部验证。

## 来源链差异与限制

H1 DDR campaign_state hash 为 2bf116be1cd68dabe78b7685d5d2afe416d1c5b781329648a036616dd9347dac，completion hash 为 1b2de9719d17d9ba1a1ea9b9ba20c2dc839f0ea1f2e573ad1f5b0d4a1c5965f5，本机唯一对应的 resolved YAML hash 也是 1b2de9719d17d9ba1a1ea9b9ba20c2dc839f0ea1f2e573ad1f5b0d4a1c5965f5；该来源链差异未能解释。completion、12,000 updates/microsteps以及u=3k/6k/9k/12k eval JSON均佐证实际运行至12k，因此不能仅据哈希差异否定训练执行。
CSV/JSON对照：23 个节点匹配；0 个CSV缺行或mIoU不一致。紧凑JSON含全部节点指标/计数、执行曝光、锁定test差值、融合统计、缓存后缀清单及哈希。
test split曾被查看；U0/U1推理几何混杂；H0不可恢复；外部dev mask并非官方像素mask；round2只看val。审查未复制概率缓存。
