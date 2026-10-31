# 2026-10-05 最后一轮方案与证据

先读 [00_最后一轮决定性实验方案.md](00_最后一轮决定性实验方案.md)，实现时对照 [01_执行协议.json](01_执行协议.json)。

本目录包含本地指标审计、代码审查、反方意见及两次CPU折外融合诊断。本次没有启动训练、使用GPU或联系服务器。TJDR尚未下载验收，新模块尚未实现。

目录：

- `notes/`：原始指标与历史路线核验、实现复审、反方审查、负责人裁决。
- `analysis/fusion_probe.py`：本次事后四折权重/阈值诊断脚本。
- `analysis/fusion_probe_result.json`：R+F4结果。
- `analysis/F12/fusion_probe_result.json`：R+F12结果。
- 两份`fusion_probe_protocol.json`与histogram NPZ：诊断拟合网格、折分与中间直方图；最终计数从原始FP32数组重读精确计算。

目标为test47.2115%，当前test两种子均值45.6993%。不要把val、test、单模型、融合系统、两种子分数均值与概率集成混为同一个指标。
