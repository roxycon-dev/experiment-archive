> 收尾已核实完成：两条本任务临时认证已精确移除，原有直连公钥保留；本任务隧道/监督已停止，45分钟automation-2已删除，两台服务器保持开机。最终收据：results/final_delivery_receipt.json。

# 论文最小补证实验

**最终实验验收（2026-10-24 17:34）：整体补证已完成。** 原HRDecoder从找回工程恢复，DDR mIoU复核42.1962%（四舍五入42.20%）；四次新增C/S固定20k、全部完整终点/EMA/优化器/RNG/采样身份本地核验、DDR227/IDRiD-derived27预测与GT逐图重算、真实三组统计、MA三方/固定九例、同硬件成本均已完成。最终5653份输入证据全SHA审计通过，报告、5份CSV、7张TeX表及7页PDF已阅读/逐页视觉核对，results/campaign_completion_receipt.json为COMPLETE_ALL_AUTHORIZED_EXPERIMENTS_AND_LOCAL_EVIDENCE / overall_campaign_complete=true。见[最终报告](论文补充实验结果.md)、[最终表格](results/final_campaign/)、[表格PDF](results/final_campaign/paper_tables.pdf)。

DDR固定融合S−C三组为+0.0652、+0.0561、−0.0969个百分点，均值+0.0081±0.0911，未证明稳定改善；外部固定融合均值+0.4289±0.1352，两个图像区间包含零。保留H4增加FP、外部去ConvNeXt分数更高及2024部分MA下降等负结果。这里只衡量同一父模型条件下适配波动，不是独立从头训练方差。

本段覆盖下方所有“进行中/等待3407/原基线缺失”等历史状态。阶段2的v1错误尝试与v2部分尝试均保留，但不作主基线；不要重训或重跑已完成评估。两台服务器保持开机，后台队列和备份监督均已自然完成。临时认证及监测清理以evidence/completed_campaign_auth_cleanup.json及最终监测收据为准。

**IDRiD分数含义：** 本轮是DDR训练模型在IDRiD-derived保留27图上的冻结外部验证，没有进行IDRiD train54上的基线重训；不得把旧v1的38.44%写成原论文IDRiD 51.94%的同协议复现。见 `plan/idrid_comparison_scope.md`。

**历史继承关系核查（2026-10-24）：** 原42.20%的HRDecoder@32k此前缺失，现已在找回工程中核验恢复；E4/B/D/E及F12父模型也实际存在。本轮C/S由F12续训，重训比较基线不构成它们的历史父模型。已从原E4日志及四个实际权重提取初始化证据：E4记录ImageNet初始化且load_from/resume_from为空，B加载E4@36k，D/E加载B。原42.20%权重→E4这一边尚未得到证据支持，不得写成已证明的直接续训。见 `evidence/historical_hr_lineage_audit.json`。

**基线验收更正（2026-10-24）：v1的40.0124%不能作为忠实原基线复现验收。** 用户质疑后，CPU逐张量验证确认原ImageNet预训练确实加载且完整初始化后1830份骨干张量逐位一致；但发现原输入配置的 `scale_ratio=(0.75,1.25)` 被构造器pop后从JSON丢失，v1重建时误用了默认1。原数据/日志/权重全部保留；v2修正重训已在找回原模型后于2980步停止，两组C/S与3407现有训练不改。见 `plan/baseline_reconstruction_amendment.json`。下方“基线阶段完成”仅指旧v1执行/计数完成，不再表示该基线符合原协议。最终报告现以原模型找回后的统一推理、MA、成本及完整SHA门槛验收，不再要求v2训练终点。

以下保留追加授权时的历史执行说明；当前整体已完成。用户已追加授权四次新种子的C/S配对训练及缺失原基线的重训，全部在两台已授权服务器执行。本地不进行长训练，不做阈值或融合搜索，不修改历史模型或论文原档案。下方早期零训练阶段不等于四次新增适配训练完成；基线最终使用找回的原权重，重训尝试单独披露。

以下为原基线找回前的历史快照，不能替代当前状态。2026-10-24 07:00：C2024/S2024及v1错误40k基线（当时按3/5记录）已经完成评估、本地逐图复算与完整检查点核验。现有seed42与新2024两组配对已齐，三组汇总仍标PARTIAL；C3407约5480/20k、S3407继续排队。新基线DDR227/IDRiD27、MA三方/九例、同4060成本及完整best/40k核验均已完成，见 `阶段2_基线重训与MA补证.md`。2024的单模型DDR提升较大，但固定融合收益小且区间跨零，部分MA IoU下降，全部保留。实际最新状态见 `results/training_live_status.json`，完整训练备份见 `results/terminal_backup_status.json`、`evidence/terminal_checkpoint_validation/summary.json`，逐图重算与配对统计见 `results/paired_seed59/*/analysis/verified_summary.json`、`results/paired_seed_summary/summary.json`。

零训练阶段的DDR227四行对照、MA分析/真实例图、完整系统计时、IDRiD27冻结评估及本地逐图复算均已完成。外部服务器另独立核验81份FP32概率缓存和全部四行GPU整数计数，PASS。

新增训练协议见 `plan/additional_training_protocol.json`。两新种子固定2024/3407，各C/S两臂、相同F12_seed42父模型、原32k日程运行到20k；只衡量该父模型条件下的适配变化。原基线最终使用找回并验证的历史32k权重冻结推理；v1的40k错误尝试与v2的2980步部分尝试独立保留，不用其结果替代原模型。主服务器实时状态为 `training_queue_progress.json` 和各训练run的status/completion；2080Ti基线状态另见远程logs/baseline_queue_status.json。脚本、计划或任务启动不表示完成。

分阶段记录见 `阶段1_推理与离线分析.md`、`阶段3_原基线找回与复核.md`；最终 `论文补充实验结果.md` 与 `results/final_campaign` 已生成并审阅。旧 `results/paper_tables.tex` 仅含阶段1，最终7表请使用 `results/final_campaign/paper_tables.tex`。

## 冻结协议与证据

- `plan/protocol.md`：DDR四行对照、MA连通/匹配/分组、固定例图规则、bootstrap。
- `plan/runtime_amendment.md`：本地参考MSDA新环境与历史CUDA结果的区别。
- `plan/external_protocol.md`：用户确认隔离的IDRiD_55–81、类别完整性来源及当前存储坐标。
- `evidence/local_asset_audit.json`：757张DDR图像和标签与原训练/评估清单SHA一致。
- `evidence/fixed_assets_identity.json`、`matched_lineage_audit.json`：A/U/S权重和匹配C/S身份。
- `evidence/external_holdout_manifest.json`：固定27图、108份逐类GT；原图/TXT/掩膜SHA、RGB及近似重复审计复核。
- `evidence/external_preprocessing_audit.json`：固定首图、最大尺寸图、末图的原有legacy预处理核查，CPU-only。
- `evidence/archive_checkpoint_audit.json`：原HRDecoder基线追溯。不同划分28000不可替代原split613 32000；后续用户已追加授权重训，原先允许跳过不再适用。

## 运行身份

- `ddr_test_local_v1`：227图、完整/去H4/去ConvNeXt/C20替换四行，共用相同A/U/S概率。
- `full_system_profile_v1`：一次warmup另记，固定validation前三图的完整A+U+S实际前向；没有C20额外前向。
- `idrid_holdout_v1`：27图，在新增SSH服务器完成的固定配方评估；保留全部四行。
- `compatibility_v1`：仅一张验证图的运行兼容性诊断及PREVIEW，不得进入论文作为正式样本。

DDR及完整系统计时使用既有本地环境：

```powershell
$analysisPython = 'C:/Users/GuanXingRen/Miniconda3/python.exe'
$legacyPython = 'C:/Users/GuanXingRen/Miniconda3/envs/M2MRF-Lesion-Segmentation/python.exe'
```

旧HRDecoder在Python3.8/torch1.10+cu113/mmcv1.7.2中运行；S/C在Python3.13/torch2.6+cu124中运行，BF16前向、FP32参考MSDA采样/概率。权重严格完整加载；本轮不声称逐位重放历史CUDA实现。DDR各行共享相同分支概率。

IDRiD外部27图迁至用户新增的RTX4080 SUPER服务器。A/U在Python3.8.10、torch2.0.1+cu117、mmcv1.7.1中运行；S/C在Python3.11.16、torch2.7.1+cu128中运行，仍用BF16及reference MSDA。每个分支先记录固定验证哨兵图的环境漂移，未根据外部结果调参。四行共用S/C/U缓存及专家前向；缓存阶段时间不作为完整延迟。模型源码847份身份一致，专家权重无损重建逐张量校验。详见 `plan/remote_runtime_amendment.md`、`evidence/remote_canary_receipt.json` 和remote审计记录。

用户后续要求先不关服务器，当前保持开机，未执行或安排关机；操作指示见 `plan/server_operation_update.md`。

脚本依赖本工作区的历史源代码和原权重；结果包不包含图像或模型，也不是独立安装包。DDR根目录为 `D:/HRDecoder/data/DDR_split_613`；IDRiD输入为 `D:/dataset/best_segmentation`。各manifest记录完整依赖身份。

## 从已保存掩膜重算，无需GPU

仅在各正式run有completion.json后执行：

```powershell
& $analysisPython D:/dataset/experiment/paper_supplement_20261023/scripts/analyze_fixed_run.py D:/dataset/experiment/paper_supplement_20261023/results/ddr_test_local_v1
& $analysisPython D:/dataset/experiment/paper_supplement_20261023/scripts/analyze_cohort_run.py D:/dataset/experiment/paper_supplement_20261023/results/full_system_profile_v1
& $analysisPython D:/dataset/experiment/paper_supplement_20261023/scripts/analyze_cohort_run.py D:/dataset/experiment/paper_supplement_20261023/results/idrid_holdout_v1
& $analysisPython D:/dataset/experiment/paper_supplement_20261023/scripts/finalize_evidence.py
```

所有掩膜NPZ均为bool，形状4×H×W，通道EX/HE/SE/MA；逐图JSON记录原图/GT/掩膜哈希和int64计数。DDR标签ID1/2/3/4，0背景、255忽略；IDRiD为四份独立二值GT，允许类别重叠，空类按历史用户完整性确认计为阴性。分析脚本从掩膜重新计数并与推理记录核对，不能只依赖总分。

## 限制

推理/离线阶段新增训练0次。用户后续追加的四次C/S配对训练正在执行，基线已在2080Ti的2/3号卡实际训练。0/1号卡在其他用户任务结束后，已接入本任务共同A/U推理缓存；三组C/S同精度评估自动接续。预计超过三小时的训练不放本地；本地仅做脚本、资产和结果复核。

统一配对推理见 `plan/paired_evaluation_runtime.md`；服务器端 `publish_paired_evaluation_jobs.py` 在每个固定20k终点到达后无损导出EMA并验哈希传到.59，当前唯一调度器 `queue_supplement_evaluation59.py` 只在卡空闲时执行。旧 `queue_paired_evaluation59.py` 已退役。新旧六个C/S均按FP32/reference统一重评，早期BF16结果单独保留。`scripts/training_remote_control.py` 会同时采集训练、环境、缓存及评估进度。

21:15调度器已升级为 `queue_supplement_evaluation59.py`，继续接管既有推理子进程，并在40k及全部预定val节点完成后追加原hook选定的新基线评估。共同缓存的765文件独立审计PASS；历史C_seed42已本地回收并从掩膜/GT复算。`recover_completed_evaluations.py` 仅回收已完成掩膜及元数据，模型checkpoint须另行备份；`analyze_recovered_training_results.py` 在完整三组出现前始终明确标为PARTIAL。

DDR内部test历史上用于开发；新增重评不消除该历史。IDRiD testing27的隔离依据本轮用户明确确认。项目其他分支使用过IDRiD training54，但进入本次评估的S/C血缘是DDR-only Q_DDR→F12。无患者ID，不能声称患者级隔离；图像bootstrap不包括训练种子变异。IDRiD为当前方形存储图和用户确认来源的转换GT，不能冒充官方原始几何基准分数。
