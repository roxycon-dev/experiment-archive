# IDRiD分数的训练与测试口径核对（2026-10-24）

当前补充实验没有训练“IDRiD训练集上的HRDecoder基线”。旧v1的IDRiD mIoU 38.4406333117%来自DDR_split613训练454图、DDR验证76图选择best，再冻结设置直接评估IDRiD-derived保留27图。修正v2仍重训DDR基线，不是新增IDRiD监督训练。

原HRDecoder论文表1中HRNet版的IDRiD mIoU为51.94%，是该数据集54张训练、27张测试的域内任务；表注为三次重复的平均值。实现细节使用2880×1920输入、1920×1920训练裁剪和滑窗推理。官方IDRiD配置还使用IDRiD归一化统计。本轮DDR模型的外部评估沿用DDR归一化、2048方形单次整图推理，当前存储图像也不是官方4288×2848原始几何。

因此38.44%与51.94%的数值差约13.50个百分点，不是同协议IDRiD重训复现误差；不能把差值全部归因为域偏移，也不能据此认定漏加载预训练。除训练域/输入/推理差异，旧v1本身还存在原HR裁块尺度重建错误，已单独撤回验收。预训练确实加载的证据和v2修正见baseline_reconstruction_amendment.json。

| 项目 | 原论文IDRiD HRDecoder | 当前DDR模型的IDRiD外部评估 |
|---|---|---|
| 有监督训练 | IDRiD train54 | DDR train454 |
| 选择/训练日程 | 原文/官方实现设置 | DDR val76固定节点选择best，原40k |
| 测试 | IDRiD test27 | 当前IDRiD-derived保留27图 |
| 输入与推理 | 2880×1920、1920滑窗 | DDR 2048方形、whole、无TTA |
| 数值含义 | IDRiD域内性能，三次均值 | DDR→IDRiD冻结泛化，单次权重 |
| 本轮状态 | 查阅的文献参考数字 | v1已撤回忠实基线验收，v2待训练/评估 |

要与原文数值直接比较，需另行建立同IDRiD监督训练/测试协议，分别训练基线与改进方法并统一评估；目前不声称已完成，也不预设结果一定超过论文。既有外部冻结验证不改为在测试集上选权重/阈值。

核对来源：

- 原论文第3.1节及表1：https://papers.miccai.org/miccai-2024/paper/4020_paper.pdf
- 官方训练配置：https://github.com/CVIU-CSU/HRDecoder/blob/main/configs/lesion/hrdecoder_fcn_hr48_idrid_2880x1920-slide.py
- 官方数据设置：https://github.com/CVIU-CSU/HRDecoder/blob/main/configs/_base_/datasets/hr_idrid_2880x1920-slide.py
- 本地实际运行：results/baseline_retrained59/manifest.json、scripts/evaluate_retrained_baseline59.py、results/baseline_retrained59/analysis/verified_summary.json。
- 冻结外部协议：plan/external_protocol.md；当前存储几何及标签溯源限制继续保留。
