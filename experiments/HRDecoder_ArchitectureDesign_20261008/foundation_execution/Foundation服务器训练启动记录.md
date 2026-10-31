# FoundationDistill服务器训练启动记录

已在2026-10-10T11:12:46.043569+08:00核验新fresh训练，controller 54053，train 54823，identity cc3a215c3e867779f4c5e2a6feedb5921c95b4cf48bfc9fe610003216be73bf8。原torch2.0.1+cu117的11项CPU检查PASS，真实2048/batch2非零辅助12步GPU smoke PASS，12步全部丢弃后重新fresh ImageNet W48 seed42/private20261008开始正式训练。没有旧task checkpoint输入。

已核验正式150步；实际供数等待0.00003809秒，96GiB RAMowner39434保留。非零辅助smoke均值1.0722秒/步；当前早期窗口1.1005秒/步不外推整个40k。

预计4k验证与交接2026-10-10T12:24:33.954864+08:00，若screen通过，同一fresh继续40k，预计训练与验证2026-10-10T23:24:52.605933+08:00（估算含20分钟验证余量，test另留，不保证）。4k门槛仍为30.475812328119%及private收益等原数值条件，未看到结果后修改。新增完整val严重崩塌保护在保存checkpoint后立即停止，避免再等4k。

服务器一直开机，外部诊断不控制，旧detail已停止且durable21000/best12000保留。目标仍固定40k完整val与锁定40k独立test均超过融合。当前没有任何val/test结果，不宣称达标。
