# SACD服务器训练启动记录

最后一次完整训练：固定2048、batch2、40000，fresh ImageNet HRNet-W48 seed42，private20261008。目标固定40k76图val>45.32183122738841%，锁定40k一次227图test>45.860383832668%。截止2026-10-10 02:00北京时间。仅此一次，无付费对照。服务器保持开机，不控制外部诊断。

CET已于2026-10-09 13:30左右完成，40k=38.313758887%，best32k=40.487642992%；76图、counts与checkpoint持久加载核验通过，所有点已报告，自有训练/控制器/RAM已结束。日志/指标归档cache_execution/server_receipts/cet，不下载checkpoint，未过val不测test。

SACD服务器原torch2.0.1+cu117 CPU12项数值/路由/前向梯度等价检查PASS，SGD动量/RNG/跨epoch恢复PASS。真实GPU2048/batch2、12次丢弃更新、一次主干调用/步、FP32损失、所有梯度有限、私有L/G/头非零梯度、fresh初始化PASS。优化后身份8f6fd8dfc7a032f335511168a4ae831f4c9a69f5aaaf00876842c11dc20c11cf。首版测速不够时间，未启动正式更新；分组执行保持四类独立参数，优化后mean1.075423699秒。

截止门槛以mean×40000+1200秒计算：320秒10次val，120秒保存/初始化，180秒test，580秒余量。未通过不得开始。控制器01:57仍训练则只停止自己创建的子进程，保留有效原子检查点且不关机。40k完成val达标后固定last.pt锁定并一次test，未达val明确记录跳过。test只计算评价，不选权重。

训练启动状态正在由reports/sacd/TRANSITION_STATUS.json及GPU_CAMPAIGN_RUNNING.json核验；后续追加真实PID与时间，不用准备状态冒充已训练。

## 已核验正式启动

启动北京时间 2026-10-09T13:42:15.600832+08:00；训练PID 25190，控制器PID 25187。FINAL_ROUND_STARTED登记一次，DEADLINE_GATE=PASS；含1200秒预留的保守完成时间 2026-10-10T01:59:12.457199+08:00。GPU smoke更新丢弃，正式从fresh ImageNet重建，不读取CET任务checkpoint。当前详细回执见cache_execution/server_receipts/sacd，交接已切换SACD_RUNNING。

启动后80步实测mean0.985028秒/步，数据等待0.000029秒，更新和损失正常。该阶段lambda_K=0；1500步后启用辅助反传，不能把前80步的短窗口0.985速度外推整轮。40k估计仍采用非零辅助GPU smoke1.075424秒/步及后续实测窗口，预计01:45–01:55完成训练和最终核验。

最新归档 2026-10-09T13:45:00.075904+08:00：completed=100，performance update=140，三个自身进程存活，无新val或错误，GPU utilization.gpu [%], memory.used [MiB], memory.total [MiB], power.draw [W] 100 %, 20931 MiB, 24564 MiB, 312.17 W 。监控仍45分钟一次，每个新验证点逐点报告。

最终test另有02:00前15秒本进程截止看门狗；超时仅结束test进程，写TEST_DEADLINE_STOP，不关机。失败或中断后只允许同一已锁定40k checkpoint及相同评价代码重试；成功test存在则拒绝重复。test锁包含评价代码SHA，结果记录完成时间，不影响当前训练身份。
