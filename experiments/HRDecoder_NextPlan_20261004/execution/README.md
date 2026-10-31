# HRDecoder 下一轮执行包（2026-10-04）

用户已要求严格执行上级目录中的方案；本目录是可执行实现，不改历史训练文件。
本地准备阶段不SSH、不开服务器、不训练、不读新test标签、不检查机器配置。

固定顺序：E_NEG42 → F4_42 → F12_42。E_NEG比较终点24k，F臂比较终点12k。
F臂2k与4k同时满足规定的双重退化条件才科学止损；正常一次回落不改方案。
同一公式在规定的两个晚期节点均比对应R提高至少0.5点才晋级；F任一臂晋级，两臂一起补2024。
最终只选两个完整种子的固定终点，原始均值精确排序，相等才按单模→25%→50%、F4→F12→E_NEG决胜。
仅锁定的一个配方及公式满足test门槛后运行test227，两种子均保留，部署固定seed42。

阴性修复：只恢复IDRiD的28个SE空类及1个HE空类。图像、mask、polygon、MAPLES uncertain和其他unknown不改。
SE受影响窗口为seed42=10761、seed2024=10762；HE为354、336。
受影响窗口完整身份、增广记录、哈希见reports/affected_windows_seed*.jsonl。
F4/F12共同的新阶段流直接冻结既有DDR流12001..24000窗口，避免重新扫描454图建立组件池；
这符合原采样分布，包含此前见过的窗口，不能声称窗口全部全新或数据独立。

E_NEG原始初始化由已核验历史代码、相同seed、相同官方预训练权重重建。
历史没有单独保存u0张量指纹。服务器验收会独立重建两次并核验一致，明确记录这一证据边界；
不能将其写成“已与历史存档u0张量逐项比较”。若历史代码、随机顺序、资源或采样身份无法核验，暂停E_NEG，
重新明确对照与成本，不静默追加24k对照或冒称严格配对。

服务器开启后，复用已安装环境与已有资产，执行：

```powershell
& 'C:\Users\GuanXingRen\Miniconda3\python.exe' -u ops\deploy.py --launch
```

本轮用户已开启的连接：root@connect.nmb1.seetacloud.com:40920，沿用已有SSH密钥。
若用户提供新连接，先更新configs/runtime.local.json中server字段及known_hosts；不存储密码。
新远端目录：/root/autodl-tmp/HRDecoder_Finetune_20261004。
旧资产目录：/root/autodl-tmp/HRDecoder_Q_20261003。
训练Python：/root/miniconda3/envs/hrdecoder_q_20261003/bin/python。
旧P0推理Python：/root/miniconda3/bin/python。

必要GPU验收仅限：E_NEG初始化身份重建、F4/F12各100次稳态完整有效更新（另10次预热）及各完整val76。
若micro2×4发生OOM，两F臂正式前统一micro1×8，记录Dice微批语义变化；E_NEG仍保持原micro2×4。
不自动添加激活检查点、改变损失或继续尝试更多微批方案。
正式训练重建初始权重，预检更新不进入正式步数。GPU测速后按实测修订耗时与费用。

开机前粗预算仅据历史：发现阶段约8–16 GPU小时，全部晋级约16–32 GPU小时；
这是规划区间，不是F12实测报价，最终test及传输另计。费用=实耗小时×用户服务器小时价；价格未提供。
不会为准备工作租新服务器或下载大规模数据。

磁盘不足时可按用户授权清理确认无关数据。自动清理限定于旧失败G分支cold_storage/G，
仅当剩余空间不足20GiB时清理该目录中的退役数据/权重，校验真实绝对路径、目录类型且不是链接；
所有服务器实验脚本、日志及结构化证据也保留，不整目录删除；
保留P0成员、DDR、IDRiD、MAPLES、DINO预训练与Q16k及本轮依赖。
少于8GiB则暂停启动；正式运行保留2GiB断点写入余量，不越过空间错误继续训练。

45分钟巡检已设为本聊天heartbeat，状态不变保持安静。巡检命令：

```powershell
& 'C:\Users\GuanXingRen\Miniconda3\python.exe' -u ops\mirror.py
```

每个技术暂停先保留日志，不自动重试。修复并核实身份后可远端显式运行campaign.py --resume-paused。
初始化、资源、源代码、采样流和断点游标须一致；不能改配置后续跑同一名称。

部署时同步启动隐藏的本地收尾进程。它等待本轮终态，不开展新实验：
终态后回收全部必要模型/optimizer/RNG、配置、逐图计数、来源有效像素审计、概率与SHA；
验证回收后才调用已确认的AutoDL /usr/bin/shutdown方式。关机请求写入收据且不重复提交。
断线不代表平台OFF；巡检应通过已登录平台页面/授权API或用户平台观察确认OFF，再暂停heartbeat。
若本地收尾进程中断，巡检可用ops/mirror.py --full补回收后运行ops/shutdown.py。
任何情况下均不得自动重新开服务器。

产物：reports/最终实验报告.md、scientific_comparisons.json、final_selection.json、
本地server_results/、final_backup_receipt.json和shutdown_receipt.json。
