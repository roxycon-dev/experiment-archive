# HRDecoder Q：本轮独立 DINOv3 系统

本目录执行 `HRDecoder_Breakthrough_Plan_20261003/01_成熟突破方案.md` 与 `02_执行与验收规范.md`。这些文件优先于旧研究笔记。当前阶段为准备；准备完成与正式长训分开记录。

主线为官方 DINOv3 ViT-B/16 LVD、四层 DPT 式重建、stride4 RGB 空间支路，1024 输入、halo128、中央768，EX/HE/SE/MA 独立通道。P0 固定为 `0.75*C+0.25*U0`；新固定融合完整展开为 `0.5625*C+0.1875*U0+0.25*Q`。C 的成员、残差头及源锁身份见 `configs/p0_identity.json`。

`prepare_local.py` 仅锁定既有 454/76/227 的 ID/SHA，并用新指标代码重现 P0 的逐图计数。test 仅哈希文件，不解码标签像素。`acceptance_cpu.py` 验收几何、ignore 损失、阈值、调度与恢复。`make_stream.py` 固定24000更新的训练 GT 采样序列，4均匀窗口、4前景类别，保留原生比例与增广账本。

`preflight_gpu.py` 只做必要验收：真实1024末四层反传100更新、冻结身份及四类头梯度、最多200更新固定真实小批接线学习、真实断点恢复、完整76图val计时及P0精确计数；全部试跑权重丢弃，正式运行重新初始化。不会运行test，不会把预检val当候选。独立服务器环境位于 `/root/miniconda3/envs/hrdecoder_q_20261003`，旧环境不升级。

正式启动命令（仅在 `reports/formal_readiness.json=READY` 后）：

```bash
cd /root/autodl-tmp/HRDecoder_Q_20261003
/root/miniconda3/envs/hrdecoder_q_20261003/bin/python train_run.py --arm Q_DDR --seed 42 --endpoint 16000
```

`train_run.py` 每200更新写可恢复last，完整节点为2k/4k/8k/12k/16k，扩展时20k/24k；仅正式晚期节点保留FP32概率与checkpoint。resume严格核对来源、配置、数据、权重、采样、RNG、优化器与LR，技术故障暂停且不自动无限重启。中断在验证节点时先补齐该节点再训练。

`advance_campaign.py` 写状态，不自行开训练。8k三项严重失败条件同时成立才停架构；12k→16k单模增益≥0.3点才触发共同24k；相邻正式节点同公式持续P0+0.5点才允许2024复核。两种子均值与一致方向条件通过后一次冻结选择，再集中test。没有持续信号就停止自动追加。

Q_MIX 必须官方 IDRiD 原图/像素标签及 MAPLES/MESSIDOR-1 原图均完成许可、逐ID匹配、uncertain ignore、训练/测试隔离、与DDR757图去重和至少20张原坐标叠图审查后才能 READY；仅旧54张多边形或MAPLES标签包不够。数据不齐先Q_DDR，不替换混合配方。

本轮服务器由用户明确分配；SSH凭据不落盘，巡检每45分钟。用户已说明费用无上限，但仍报告实测耗时、GPU小时和实际费用，不因此无限追加。清理限当前实例上本轮不用的可再生成历史非入选概率缓存，保留P0、数据、全部权重与日志；冷存中未本机回收的历史证据保留。每条正式run所需证据和权重先同步本机并核验SHA再关服务器。关机请求回执与平台OFF分开记录。
