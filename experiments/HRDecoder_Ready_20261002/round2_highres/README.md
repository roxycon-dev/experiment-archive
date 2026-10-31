本轮从历史U0最佳权重进行2048与2048/2560配对续训，最多4000步/臂，按固定75%旧冠军＋25%候选的完整val选择best。上一轮227图test已完成，本轮不再评价该test。协议与配置在protocol.json、configs/；实时进度在reports/进度.md；新best本机镜像在server_results/work_dirs/。

尺度为基准长边，两臂均保留0.5–2.0随机缩放扰动。后台每分钟同步健康和结果，模型按每小时巡检。

<!-- HRD_AUTOSHUTDOWN_20261003 -->

用户已授权整轮完成后关闭服务器。后台收尾控制器等待队列全部结束、完整验证和工作区汇总落盘、两臂 best 和历史部署权重本机 SHA 校验通过，并确认服务器无实验进程/GPU任务后，再执行 AutoDL 的 /usr/bin/shutdown。工程异常或备份未完成会保持开机；不会释放/删除实例。实际执行状态见 reports/shutdown_state.json，条件见 shutdown_policy.json。
