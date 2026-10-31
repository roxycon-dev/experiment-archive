> 收尾已核实完成：两条本任务临时认证已精确移除，原有直连公钥保留；本任务隧道/监督已停止，45分钟automation-2已删除，两台服务器保持开机。最终收据：results/final_delivery_receipt.json。

# 最终实验验收完成（2026-10-24 17:34，覆盖全部下方过程快照）

四个新增适配固定20k、完整状态/EMA本地核验、最后S3407两队列评估/768结果文件回收与独立GT计数全部PASS。三组summary为COMPLETE_THREE_PAIRED_SEEDS。最终5653输入SHA审计PASS；报告/5CSV/7TeX表已读，PDF七页实际view_image全部审阅后写final_report_review，再次finalize已得到COMPLETE_ALL_AUTHORIZED_EXPERIMENTS_AND_LOCAL_EVIDENCE / overall=true。真实原基线42.1962%复核42.20%。请看论文补充实验结果.md、results/final_campaign及campaign_completion_receipt。

不要再启动训练、推理、备份watcher或旧队列。临时认证/隧道只作本任务精确清理，实际结果以evidence/completed_campaign_auth_cleanup.json为准；服务器保持开机。下文均为历史过程记录，包含当时缺失/进行中状态，不是当前待办。

# 连续执行入口（原模型找回后更新，优先于下方所有v1/v2历史说明）

**16:28完整备份核验更新：四个新增完整终点+EMA以及原基线均已本地备份/CPU全状态核验PASS**，summary为ALL_REQUIRED_TERMINALS_AND_ORIGINAL_BASELINE_VALIDATED，backup all_terminal_backups_complete=true。S3407的1908个模型/EMA/优化器张量有限、EMA导出逐位、原源码/协议/160000采样游标和后8行/RNG身份全PASS。watcher45360已在全部备份完成后正常退出（terminal_backup_detached_exit.json NORMAL_EXIT），此时缺PID是预期，不重启。剩余工作只有S3407最后两队列推理/固定融合、结果回收与独立GT复算、真实三组统计和最终报告/表格实际审阅，仍overall=false。

**16:26最新：四次新增C/S训练均达到固定20k，尚未整体完成。** 最后S3407完整终点SHA3967757a712413ebc68b868ccb81d52cbdcf624863aced71bcdd9a77405a0380，全部1000个记录节点loss/grad有限；保留原32k日程与EMA20k，发布器六任务全部发布。S3407完整1.62GB模型已本地验分段/整SHA，EMA405MB仍由健康watcher45360回收，先等其完整receipt再CPU全状态核验，不重复启动watcher。最后S3407统一2080Ti FP32评估PID1952799/GPU0已正式运行，16:25到DDR50/227，随后还需IDRiD27与CPU固定融合；尚未completion，不打包未完成结果。root训练卡目前无剩余训练任务，保持开机、不空跑；.59其他卡空闲，不再沿用旧他人占卡状态。MiKTeX的pdflatex/xelatex/lualatex已在本机可用，最终真实表格生成后可编译并实际看PDF，不用安装新依赖。

**14:13执行准备更新：** 最后S3407约13.2k/20k仍健康，剩余适配约2.15h；.59 GPU0此时空闲，其他1/2/3卡仍留给其他用户，不预占。只读核清了两条本任务临时认证，未删除任何条目，见temporary_auth_cleanup_inventory.json。最终生成器新增真实交付审阅保护：全部模型证据齐全后第一次--finalize只写RESULTS_GENERATED_AWAITING_REPORT_REVIEW/overall=false；实际阅读报告和CSV/TeX后才写evidence/final_report_review.json（PASS_FINAL_REPORT_REVIEW、实际时间/方法、完整reviewed_outputs_sha256），再--finalize输出SHA完全匹配才overall=true。详见plan/final_delivery_and_auth_cleanup.md。新代码已编译，缺S结果时仍实际返回WAITING；完整生成/审阅分支尚未执行，不能假称已验收。

**13:26资源更新：** S3407实际10780/20k，全部539个记录点的loss/grad有限，root训练正常。此时.59四张2080Ti由其他账户studentuser003的进程使用（PID1701298–1701301），不是本任务重启v2，也不是待结束的旧评估；严禁干预。既有评估队列仍健康，S终点未到；等实际空闲卡，不提前混用其他硬件/重复评估。当前主训练不受影响，详见evidence/monitor_health_20261024_1325.json；不能再照旧提示认定四卡一直空闲。

**2026-10-24 10:28最新：新增C3407固定20k已训练、完整状态回收/CPU核验及两队列统一FP32推理/本地逐图计数复算全部PASS，新增已核验3/4。** C3407完整SHA7dd1740295b41f002865b59bc8cb065078aed5d364bdfde63539a309d4a9e158，EMA SHA d809e2686588e6e6ae96fd60ef126b45f3b2fc0ba8a5d40c310b14f1b21446f8；1278模型/EMA/优化器张量有限，完整RNG与160000采样游标/后8行及导出逐位PASS。768文件/15015354字节结果包SHA4a31eefa8d62e03e7e2a337cd88e88a69527eba3c5a3e070176da933af07ef32。尚无S3407终点，三组统计仍PARTIAL（仅42/2024两组），不能把新C单臂加入配对均值。S3407已接续约1260/20k，日志损失/梯度均有限，预计适配约余5.9h；主队列/发布器/.59调度均健康。

本地旧备份watcher33400在C3407备份后失去进程、无stderr；训练/评估未受影响，未丢结果。已通过WMI隐藏pythonw独立后台恢复为**PID45360**，以results/terminal_backup_controller.json实际PID为准；新日志logs/terminal_backup_detached_20261024.*.log、退出证据logs/terminal_backup_detached_exit.json（健康运行不应存在）。见evidence/local_backup_watcher_recovery_20261024_1028.json。健康时不重复启动；如再次消失先查实际进程/文件锁/退出证据，不猜训练失败。

**原基线已完成补证；不要重启v2，也不要重跑已完成原基线推理/计时/九图。** `plan/original_baseline_recovery_amendment.json` 与 `results/original_baseline_phase_receipt.json` 为当前执行入口。原模型来自用户找回的工程文件，完整SHA446fc7b598e9680a1470059d2aba3a6f9c2222b66658a5eccef6059b1362850f，原seed42/split613/best32k/val39.39018219%；原配置与日志逐字节匹配，1844状态与923优化器槽有限，完整本地备份与推理导出逐位一致。

原基线本地统一推理完成val76=39.39255813%、DDR227=42.19623780%（四舍五入42.20）、IDRiD-derived27=35.59566480%；330图掩膜及计数独立GT复核PASS。结果 `results/baseline_original_local`，666文件本地结果包SHA599f0e8fa6ff9103e644a33052f762aa134508ab085fe853d345f822c33d3a1e。MA三方 `results/baseline_original_threeway`，九图已实际通过view_image审阅，新的QA `evidence/baseline_original_threeway_visual_review.json`。原模型同4060相同3val成本0.3196801秒/图，另记warmup；`results/baseline_original_profile_local`。阶段报告 `阶段3_原基线找回与复核.md` 已按实际数据生成；SE49.9876与历史49.98有约0.01pp小差异，不冒称逐位历史运行时重放。阶段2仅v1历史，不作主基线。

v2监督器及其全部自有后代已确认退出，停止时2980步，保存iter_2000完整checkpoint/日志未删。不是训练失败，不是完成40k；是原权重找回后不再必要，见 `evidence/baseline_v2_superseded_by_recovered_original.json`。一次身份guard因DataLoader名为multiprocessing而拒绝，在任何操作前退出；补查父子链后才停止，全部留痕，未动他人/C/S/服务器电源。预算记v1 40k错误尝试、v2 2980步部分尝试和四个20k C/S，不是两个40k完成。

用户追问程序是否写错：v1 scale_ratio遗漏已确认；v2恢复原值，额外detail/context/learnable fusion/SE slide均未启用。对checkpoint标记Git cfc6c88的核心代码进行固定BN/小合成几何/相同随机数下前向与反传，损失与923组梯度差均0，见 `evidence/baseline_core_equivalence_audit.json`。无优化器更新。该检查不是完整双卡SyncBN/采样/RNG/梯度通信一致性证明，不能承诺v2必达42.20。原模型找回不建立E4历史父边。

四个适配任务不变，继续root唯一队列/publisher8854、.59唯一dispatcher453129。备份watcher现 **PID33400**，`results/terminal_backup_controller.json` 与 `logs/terminal_backup_watch_original_20261024.*.log`；已改成原基线感知，不再等待v2终点。CPU核验器也使用原模型身份，要求四个新C/S完整状态+原模型备份，summary状态齐全后为 `ALL_REQUIRED_TERMINALS_AND_ORIGINAL_BASELINE_VALIDATED`。目前仍缺3407两臂，必须PARTIAL。不要重复启动健康进程。

**2026-10-24 09:50已接通最终汇总，仍缺3407真实结果：** `finalize_training_campaign.py` 已将全部门槛/审计/报告/CSV/TeX切换原模型阶段receipt、baseline_original_local、baseline_original_threeway、新QA/profile、原日志、导出与恢复的原32k。旧v2临时assert已用完整原模型身份/状态/日志/SHA门槛替代，不是简单删除保护。修正了旧“外部去H4低于新基线”的失效解释，改为真实原九例中部分目标被原基线检出而full仍漏检；不沿用v1为主基线的结论。报告预算明确v1错误40k、v2部分2980、四个适配20k及原基线只推理。新 `--audit-available` 已实际核验4095文件PASS，仅已就绪证据，不冒充最终全审计；`--finalize` 也实际拒绝在缺3407/三组/完整模型时生成总报告。真实三组到齐后仍须执行完整分支、阅读数字/科学解释、检查CSV/TeX及总receipt，不能把这次代码准备算作总报告完成。

v2部分尝试已额外本地回收2k模型（525854986字节）、原日志/状态/配置8个文件并验SHA；2767个模型/优化器张量有限，见 `evidence/superseded_baseline_v2_recovery.json`、`evidence/superseded_baseline_v2_history`、`models/superseded_baseline_v2_u2000.pth`。原工程8权重也已清点SHA和元数据，见 `evidence/recovered_project_checkpoint_inventory.json`，未推理/挑选其他变体。主C3407实际19000/20k，S3407仍待队列，约余6.46小时适配（不含评估/回收）；两端健康，.59等待新终点。新的整体粗估18:00–20:00见campaign_ETA_latest，但实际只按验收门槛判断完成。

两台服务器保持开机；等待新终点时无意义空跑禁止。后续按实际日志/损失/NaN/OOM/磁盘、回收/独立复算/完整权重核验推进，只有三组及总报告真正完成才收尾。本文件以下均为旧执行历史，不再要求v2到40k或重跑已完成原基线阶段。

---

## 以下为原模型找回前的历史说明

用户后续质疑IDRiD分数，已核原文表1/第3.1节与官方配置：51.94%为IDRiD train54→test27域内三次均值；本轮旧38.44%是DDR train454→IDRiD-derived27冻结评估，未做IDRiD监督训练。修正v2仍是DDR基线。训练域/归一化/2048 whole与官方2880×1920 slide/当前存储几何皆不同，不能直接比较或把差距全归因域偏移。报告需显式写DDR→IDRiD，不自动追加新的IDRiD训练以追高分。详情 `plan/idrid_comparison_scope.md`。

**当前基线未完成验收。** v1错误从构造后JSON重建，丢掉原输入配置的随机HR裁块尺度(.75,1.25)，误用默认1。预训练已实际核验1830个骨干张量逐位相同；错误不是没加载ImageNet。见 `plan/baseline_reconstruction_amendment.json`、`evidence/baseline_initialization_and_constructor_audit.json`。v1权重/日志/预测/成本全部保留，旧阶段2仅是该错误尝试的历史执行记录，不作忠实原基线主结果。预算必须披露四次适配和两次40k基线尝试。

修正v2已在.59双GPU2/3运行，supervisor PID1327017，真实进度见 `scripts/training_remote_control.py` 输出 `baseline.baseline_v2`。路径 `/home/studentuser001/paper_supplement_20261023/baseline_v2`，日志 `logs/baseline_v2_queue_status.json`、`baseline_v2_train.log`。保持原40k/十val节点/原hook best、FP32全局batch2 SyncBN；只恢复原scale范围和crop_num4，不按test搜参。原单4090 BN2与双2080Ti SyncBN资源差异仍披露，不能保证42.20。训练后supervisor调用 `finish_baseline_v2_59.py` 并自动以固定best评估到 `evaluation_workspace/paper_supplement_20261023/results/baseline_reconstructed_v2_59`。健康时不重复启动。

原C/S训练不变：同一F12 raw父模型，原32k日程到20k，保留全优化器/EMA/RNG/采样身份。42与2024两组已复算；3407两臂尚未齐，三组统计仍PARTIAL。root发布器8854、.59唯一旧评估调度453129保留，不重启退役队列。

本地备份watcher已重新加载为 **PID40604**，见 `results/terminal_backup_controller.json`、`results/terminal_backup_status.json`，日志 `logs/terminal_backup_watch_v2aware_20261024.*.log`。现等待v2的 `baseline_v2_seed42.json`、`baseline_v2_best_u<节点>.pth` 与 `baseline_v2_endpoint_u40000.pth`，不以v1充数。CPU核验器也已v2感知，MMCV的40k meta.iter39999仍是经源码证明的保存语义。

**尚需技术接通并真实执行的v2下游：** pack/recover脚本识别新名 `baseline_v2_seed42` 与新结果目录；独立GT/掩膜重算；新v2日志及十val节点回收；用新best在同4060/相同三val计时；v2/noH4/full MA三方及原固定9例加入v2列，再实际查看9图并保存新的v2审阅记录。既有隔离依赖可复用，v1计时/预测/原日志/视觉QA不能当作v2。最终生成器已有v2强制门槛与临时旧路径拒绝保护；必须完整切换其输入、门槛、SHA审计、报告与两次基线尝试预算后才能解除临时assert。禁止只删除保护。

**历史血缘也须区分：** `evidence/historical_hr_lineage_audit.json` 是从实际E4/B/D/E权重及原E4 JSON日志提取的证据。E4记录ImageNet初始化、load_from=None、resume_from=None；B记录加载E4@36k；D/E记录加载B。暂未证明原42.20% HRDecoder@32k→E4。缺失原基线文件不等于后续改进权重缺失；F12原父模型确实存在且是本轮C/S父模型。新重训比较基线不能追认为历史父模型。不要虚构这条边或据此改写旧训练。

两端保持开机；不干预他人，不为空闲卡制造工作。每45分钟核验实际健康、推进可执行回收和技术准备。只有四次适配和修正v2基线、全部原图预测/计数/完整状态本地验SHA、MA九图、同硬件成本、三组统计与总报告真实完成才整体完成。临时认证仅清理本任务条目且须在真正收尾后，绝不擅自关机。

---

## 以下为v1验收撤回前的历史说明，不作为当前执行指令

**2026-10-24 07:00更新：S2024已完成完整模型回收/全状态核验及两队列逐图复算，42/2024两组配对已齐。** 后续只欠3407两臂训练及其回收/评估、3组总体统计/最终报告。不要重复运行既有推理。2024固定融合总体增益小且图像区间跨零，部分MA IoU下降；保留这些结果，不调参。最新状态仍须实际查询。

**2026-10-24 06:12更新：下面的“基线就绪后”全部已实际完成。** 原40k基线及C2024/S2024均到终点，基线best选16k，DDR227/IDRiD27逐图复算、MA三方、九例实际视觉审阅、同4060计时、原日志/完整best和40k备份均PASS。见 `阶段2_基线重训与MA补证.md`、`results/baseline_phase_receipt.json`。不要重复跑已完成的本地GPU计时或重选例图；保留已有QA。后续聚焦S2024完整模型回收/配对复算及3407两臂。S2024在CPU融合，C3407已接续训练，S3407仍排队，三组整体未齐。

核验器已按实际原MMCV源码校正：文件iter_40000的meta.iter为39999（self.iter+1命名、更新后递增前保存），不是少训一步，未改训练/检查点。证据 `evidence/baseline_checkpoint_iteration_semantics.json`。计时缺prettytable的导入故障也已用实际59环境的纯Python包隔离恢复并留痕，已有完成计时不能重跑。

本文件是后续执行说明，不是完成证明。以实际状态与逐文件SHA为准，不能用本文数字代替下一次查询。全部五次新增训练结束后还要完成评估、备份、MA、成本、统计与报告。

## 先检查健康

使用 `C:/Users/GuanXingRen/Miniconda3/python.exe scripts/training_remote_control.py`。S2024正在主服务器训练，C2024已完成；3407两臂仍按既有队列等待。.59原40k基线继续GPU2/3，0/1已做完当前有用缓存/评估后等待新终点。不得为了利用率启动无意义计算，不得关闭机器或干预别人进程。

主服务器训练源码 `training_v1` 有哈希身份，不要修改。评估发布器PID8854，新调度器 `queue_supplement_evaluation59.py` PID453129；旧调度器不可重启。旧U失败已修复，结合completion而非历史首尝试判断。损失/梯度、NaN/OOM、磁盘、最新步数与进程均需检查。

## 本地回收与复算

所有下列命令的工作目录是本项目根，使用上面的Python，除明确写出的旧环境计时外。

1. `scripts/recover_completed_evaluations.py`：只打包已完成结果，回收原图掩膜及计数/来源并验全部文件SHA。历史C/S42和新C2024当前都已完成。
2. `scripts/analyze_recovered_training_results.py`：从本地GT和掩膜重算；未齐3组必须PARTIAL。不能将未配对C2024加入S−C均值。已完成成员不会重复做推理。
3. `scripts/verify_terminal_checkpoints.py`：已回收完整last+EMA才做CPU状态/采样流/源代码/RNG/优化器/有限性/逐位导出检查。C2024已实际PASS。

完整模型由 `scripts/backup_terminal_assets.py --watch` 后台持续回收，单实例文件锁。当前controller在 `results/terminal_backup_controller.json`，最新PID36124；旧PID36384不在且无Python错误日志，于01:49恢复并保留旧日志。先看进程和 `results/terminal_backup_status.json`，健康时不要重复启动。如果失效，留痕并用Start-Process -WindowStyle Hidden恢复；不能把修复本地watcher写成训练失败。新日志 `logs/terminal_backup_watch_restart_20261024.*.log`。四连接固定身份SSH分段/全文件SHA与前缀续传已实际测试，C2024的1.4GB完整last回收约233秒。大模型不要退回默认SFTP或无预取串行下载。

## 基线就绪后

基线必须等40k、全部10个val节点和原hook的唯一best选定。不会使用DDR test或IDRiD选权重。

- 运行 `scripts/recover_baseline_training_evidence.py`，回收原训练日志、配置和所有val节点。当前实际执行返回WAITING，未假称日志已完成。原源码/数据清单另有本地冻结身份。
- 完整best本地备份验SHA后，以 `C:/Users/GuanXingRen/Miniconda3/envs/M2MRF-Lesion-Segmentation/python.exe scripts/profile_retrained_baseline_local.py` 跑同4060、同三张val、一次不计入warmup的实际成本。运行很短，允许本地；不要重训、不要重跑完成的计时。精确原源码、隔离纯Python依赖和CPU模型预检已完成，不需再pip或升级Torch。
- 基线两队列结果回收/重算后运行 `scripts/analyze_baseline_threeway.py`，保存基线/noH4/full的MA分组、检出/FP、区间和沿用固定9例的基线列。
- **实际用view_image看完生成的9张图**（可先contact sheet再检查细节），确认标题、对应GT/预测、目标白框、无遮挡/截字；只修布局，不重选案例或改预测。然后才手工保存 `evidence/baseline_threeway_visual_review.json`：

```json
{
  "status": "PASS",
  "reviewed_at": "实际UTC时间",
  "method": "Actual visual inspection with view_image",
  "selection_changed": false,
  "figures": [{"path": "每张实际绝对路径", "sha256": "该文件实际SHA256"}]
}
```

figures必须9条，完整覆盖example_provenance.json，不可在没有看图时预填PASS。

## 总体收尾

`scripts/finalize_training_campaign.py` 无参数仅检查门槛；加 `--finalize` 在门槛未齐时仍只输出 `results/campaign_gate.json`，不会写总报告。已实际测试当前缺11项证据时不会生成总报告。全部齐全后会重新验所有模型/结果包/逐文件SHA、阶段1整数总计、三组均值/样本SD、同硬件/同图与真实图审阅，再生成：

- `论文补充实验结果.md`；
- `results/final_campaign/` 下总表、三组逐项/均值SD、MA三方CSV、booktabs TeX、输入数字来源SHA；
- `results/campaign_completion_receipt.json`，仅真正通过所有门槛才overall_campaign_complete=true。

生成前后需要读报告与表格，修复实际代码/文件差错并留痕，不能绕过核验。当前新finalizer已编译且“拒绝提前完成”分支实际验证；完整分支尚无数据可执行，不能冒称已通过。将最终报告的科学解释同实际新种子结果核对，保持方向混合/区间跨零和负结果，不夸大稳定性。

训练总预算不能冒称完全相同：本轮4×20k适配、一次40k基线之外，最终系统另有既有专家/父模型/ConvNeXt/H4训练。注册参数包括复制reference/辅助模块，不是去重参数或FLOPs。三种子只测同父模型条件适配，不是完整系统从头波动。图像bootstrap不是种子SD。

最后保留DDR test历史开发使用、IDRiD隔离为用户确认、原方形存储与转换GT缺官方原始独立比对、无患者ID，以及外部去ConvNeXt更高/H4增加FP等负结果。实验完成不等同TMI/MedIA创新达标。

两台服务器一直保持开机，完成后也不能擅自关机。仅清理本任务新增的临时认证条目，不能删别人key。完成通知与停止不再必要的本任务自动化需在真实整体完成后处理；此前45分钟心跳持续。
