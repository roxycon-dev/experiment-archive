# HRDecoder重新诊断资料包

2026-09-26。本轮交付是诊断、经验整理和研究方案；没有新增GPU训练结果。

建议依次阅读：

1. [重新诊断与主方案](01_重新诊断与主方案.md)：根因、纠错、主方向、对照矩阵、验收标准。
2. [历史实验台账审核版](02_历史实验台账_审核版.md)：70条记录，包含训练、推理候选、旧计划及工程失败，**不是70次独立训练**。
3. [集成与尺度候选原表](02b_集成与尺度候选原表.md)：9月15日至16日完整22行test、5行val候选表。
4. [文献筛选与证据](03_文献筛选与证据.md)：21项论文/数据/代码来源，说明检索范围、读取深度、选择及限制。
5. [审查修正与证据链](04_审查修正与证据链.md)：本次如何纠正旧报告和WorkBuddy初稿，以及哪些事实仍不完整。

证据文件：

- `history_ledger_reviewed.json`：台账机器可读版。
- `local_file_catalog.jsonl`与`coverage.json`：3862个本地文件索引、范围和排除项；不等于逐文件内容核验。
- `synthetic_audit_results.json`、`audit_synthetic.py`：CPU合成检查，使用真实ep19精修器权重，证明补零行为不一致及错误饱和输出的局部零梯度；不能换算成实际val增益。
- `candidate_audit_summary.json`、`expert_key_evidence.json`：原始验证集分析的摘取。
- `asset_presence.json`：现冠军关键推理权重的本地存在性；本轮没有重新完整推理或重算其全部哈希。
- `E4_initialization_evidence.json`、`H0_metadata_readback.json`：从原始训练日志/压缩包核对初始化与执行状态。
- `delegation/`：WorkBuddy DeepSeek v4.1 Flash的只读委派任务和原始答复。文件名含`unreviewed`的是原始中间稿，**不得越过审核版直接采纳其结论**。

历史目录未被改动。工具脚本仅用于审查及文件整理，不是下一轮训练代码。未实现的新方案不应被登记成实验已执行。
