# Q 后续路线反审（2026-10-04）

## 已有证据

Q= DINOv3 ViT-B/16末4层5e-6微调+DPT-128+stride-4 RGB末端融合，2 seed/24k。val16k，seed42/2024：EX 45.69/45.09、HE 39.18/35.81、SE 47.43/48.83、MA 19.28/19.46；P0 为 47.74/42.51/57.07/26.64。Q 单模每类均低于 P0；MA precision 27.6/32.6%、recall 39.0/32.6%，P0 为44.4/39.9%，不是只缺召回。测试低权重融合有约+0.9pp互补收益，但不能据此选结构；外部混合约-5.6pp，泛化仍是硬风险。IDRiD polygon 是官方像素标签无损转换，标签格式不是根因。

## 建议主线：(1) 更深适配，不添模块

固定 DPT/RGB头，只改变 ViT 适配深度：全层分组低 LR（早层更小），LoRA 可作独立对照。改动低至中、最容易归因；Q 已有互补信号，先检验是否因只调末4层而欠适配。固定并记录 MIX 各来源曝光，按来源验证。相同 2 seed、val16k 作门槛；测试只做最终锁定评估。

致命风险：小样本全层更新遗忘通用特征、继续扩大 MIX 掉点；Q 当前 val 全类落后 P0，深化更新没有直接收益证据。达不到稳定 val 改善就停；47.2115 是目标，不能保证。

## 另两条线

**(2) ViT-Adapter 式早期空间交互：机制最贴题，工程中高。** ViT-Adapter 在编码器分组间注入卷积空间先验并抽取多尺度特征；与 Q 末端拼接有实质区别。Glasses 直接适配 DINOv3，在 Synapse/ISIC 报告 83.52/91.60 Dice，但仅腹部、皮肤；评审认为提升有限、数据集少。迁移需接 DINOv3 token/位置编码和多层交互，过拟合与调试面是主风险。若试，仅做轻量中层注入，不照搬双编码器。

**(3) DINOv3 ConvNeXt-Small 金字塔：真多尺度，互补未证。** 官方实现有 LVD 预训练50M模型和四阶段特征图；接多尺度 decoder 为中高工程，且新增显存/推理成本。致命风险是换架构不等于错误独立；Q 对 P0 的融合证据不能证明 ConvNeXt 对 Q 有增益。先用固定 val 查逐源指标、错误相关，再决定是否训练。

## 文献边界

MedDINOv3 的 blocks 2/5/8/11聚合、提高分辨率在 AMOS22 分别报告+2.10/+2.06 DSC，支持多尺度假设；但它基于3.87M CT切片做域预训练，不是眼底适配。CT-3M列入 AMOS22/BTCV/KiTS/LiTS，亦是其评测集；主文未说明预训练排除评测折，不能视为干净外部泛化证据。论文/代码未见 DDR、IDRiD 预训练记录。DINOv3的网页预训练也不等于已知DDR污染，公开资料无法排除图像重合。

## 原始论文/官方代码

- [DINOv3论文](https://arxiv.org/abs/2508.10104) · [Meta代码](https://github.com/facebookresearch/dinov3)
- [ViT-Adapter论文](https://arxiv.org/abs/2205.08534) · [官方代码](https://github.com/czczup/ViT-Adapter)
- [Glasses MICCAI 2026论文页](https://papers.miccai.org/miccai-2026/0429-Paper1781.html) · [作者代码](https://github.com/friay/Glasses)
- [MedDINOv3论文](https://arxiv.org/html/2509.02379) · [作者代码](https://github.com/ricklisz/MedDINOv3)
