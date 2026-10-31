# 私有纠错正负监督对齐训练启动记录

2026-10-10T12:42:45.781546+08:00已核验fresh ImageNet W48 seed42/private20261008新训练150正式步；没有旧task权重。train57901/controller57131，identity 637a5e77db60fecb85c569b7bbbb9cdececdef317e8a62136f49a1f9ef1a5f04。固定2048/batch2/40000，原torch2.0.1 CPU13项PASS，真实非零辅助12步GPU smokePASS并全部丢弃后重新fresh开始正式更新。

只改Foundation私有GT BCE与私有教师KL的fullGTpositive/negative固定.5/.5归一化。网络、base保护、BN、采样、LR4x、单SGD、全局clip5和严重崩塌自动保存停止保留。教师仅训练，推理无GT/teacher/TTA/H4。数字screen门槛未变，新结果前注册。

8个train-only无SGD梯度探针完整归档；支持增加阳性权重方向，不证明参数空间唯一因果或最终fresh泛化。注册JSON中的predecessor字段保留更早adapter/detail历史，immediate Foundation拒绝记录已在spec与独立REGISTRATION_METADATA_ERRATA明确，无数值/训练源身份改动。

新GPU smoke非零辅助均值1.1033秒/步，正式早期窗口1.1136秒/步，数据等待0.00004995秒。自己96GiB RAMowner39434复用，服务器开机，外部诊断不控制。

预计4k筛查与交接2026-10-10T13:55:23.331243+08:00；若screen通过同一fresh继续40k，预计训练/验证2026-10-11T01:20:32.847944+08:00（含20分钟余量，test另留，仅估计）。全76val每4k汇报，原阈值全部保留；失败自主诊断下一版，严重崩塌不等下4k。固定40kval过融合阈值才锁定同一checkpoint一次custom227test；当前没有新val/test，目标尚未达。
