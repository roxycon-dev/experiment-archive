# 第三轮教师身份核查：S@20k

核查日期：2026-10-08。只读源码、JSON 身份/协议和文件目录；未打开或载入大型权重文件，未训练、未运行模型推理。

## 判定

S@20k 推理时仍运行 DINOv3 ViT-B/16。它不是仅用 DINO 软目标训练出的轻量 HRDecoder：S 是沿用 F12 的 DINO 分割模型，在其前向中加入可训练的 RGB 空间金字塔、四段空间交互和空间残差，再接回原 DDR 解码器。S 阶段的监督是 DDR 分割标签的 masked BCE/Dice；训练代码里没有 DINO 教师 logits 或蒸馏损失。最终锁定记录把 S@20k 作为 25% 分支，与历史 P0 做固定融合，所以“最终系统”也不是单独的 S 模型。

## 图、参数与初始化

- [final_model.py](D:/dataset/experiment/HRDecoder_FinalDecision_20261005/execution/server_results/final_model.py:49) 的 `FinalModel` 保留 `parent.backbone`；第 84–110 行执行 DINO token 化、四次空间注入、原 ViT blocks 0–11 和四次空间提取；第 112 行把多层 ViT 特征及空间图送入 decoder。第 74–112 行的 `forward` 只返回分割 logits，没有 feature-export 返回值。
- [spatial.py](D:/dataset/experiment/HRDecoder_FinalDecision_20261005/execution/server_results/spatial.py:59) 实现 12-head、每头4点的可变形注意力；第 121–161 行定义四段交互和卷积空间金字塔。因此这部分不是只在训练时使用的 DINO 监督器，部署前向会执行它。
- [final_selection_lock.json](D:/dataset/experiment/HRDecoder_FinalDecision_20261005/execution/server_results/reports/final_selection_lock.json) 的 `$.arm="S"`、`$.update=20000`、`$.checkpoints["42"].path`、`$.formula="fixed_fusion"` 指向 S@20k。该锁还标明 `$.seed_scope=["42"]` 和 `$.single_seed_exploratory=true`。 [final_test.py](D:/dataset/experiment/HRDecoder_FinalDecision_20261005/execution/server_results/final_test.py:38) 第 38–44 行严格装入该 checkpoint 的 `ema` 后测试；[validation.py](D:/dataset/experiment/HRDecoder_FinalDecision_20261005/execution/server_results/validation.py:53) 第 53–56 行给出 `fixed_fusion = 0.75*P0 + 0.25*S`。
- S 身份记录 [identity.json](D:/dataset/experiment/HRDecoder_FinalDecision_20261005/execution/server_results/runs/S_seed42/identity.json) 的 `$.initialization` 写明 DINOv3 ViT-B/16、LVD1689M、strict pretrained、无 missing/unexpected keys，backbone 参数 85,669,632；`parent_completed=12000`、`parent_optimizer_loaded=false` 和 parent SHA 指向 F12@12k。预训练身份 [pretraining_identity.json](D:/dataset/experiment/HRDecoder_FinalDecision_20261005/execution/server_results/reports/pretraining_identity.json) 记录权重 SHA `73cec8be…`。
- 身份中的 `total_parameters=88,263,748` 是 Q/F12 父模型的计数，不是 S 全图计数：[`qlib/model.py`](D:/dataset/experiment/HRDecoder_FinalDecision_20261005/execution/server_results/qlib/model.py:79) 第 94–97 行先构造 Q 并计数；之后 [experiment.py](D:/dataset/experiment/HRDecoder_FinalDecision_20261005/execution/server_results/experiment.py:36) 第 36–42 行才构造 `FinalModel`。按源码定义和张量维度静态相加，S 新增空间金字塔/交互 12,721,184 参数、四个空间残差投影 303,616 参数，故 S 全图为 **101,288,548 参数**。这是未载入实例权重的源码算术结果，不是 checkpoint 中现成的参数统计字段。
- [final_model.py](D:/dataset/experiment/HRDecoder_FinalDecision_20261005/execution/server_results/final_model.py:57) 第 57–69 行用隔离 seed 初始化新模块，复用旧 backbone/decoder；backbone 全部可训练，只有 `mask_token`、`cls_norm`、`local_cls_norm` 冻结。空间残差投影零初始化见第 18–22 行；注入/提取模块各自初始化见 [spatial.py](D:/dataset/experiment/HRDecoder_FinalDecision_20261005/execution/server_results/spatial.py:67) 第 67–83 行。

## 训练来源与日程

- S 的父权重是 F12@12k。 [experiment.py](D:/dataset/experiment/HRDecoder_FinalDecision_20261005/execution/server_results/experiment.py:21) 第 21–34 行校验并严格载入 F12 checkpoint，记录 `parent_completed=12000`，不载父 optimizer。F12 自身由 Q@16k 模型权重起步：[HRDecoder_NextPlan_20261004/execution/train.py](D:/dataset/experiment/HRDecoder_NextPlan_20261004/execution/train.py:33) 第 33–51 行校验和载入 `checkpoint_completed=16000`，并设 `old_optimizer_loaded=false`；[configs/protocol.json](D:/dataset/experiment/HRDecoder_NextPlan_20261004/execution/configs/protocol.json) 的 `$.F4_F12.new_stage_updates=12000`、`$.F4_F12.optimizer="Fresh AdamW"` 与 DDR-only 数据定义对应 F12 阶段。因此链路是 DINO LVD1689M → Q_DDR@16k → F12 再训12k → S 阶段。
- S 阶段在 [train_run.py](D:/dataset/experiment/HRDecoder_FinalDecision_20261005/execution/server_results/train_run.py:31) 第 31–37 行新建 optimizer、scheduler、完整模型 EMA；第 68–77 行训练到32k并在注册节点验证。 [configs/protocol.json](D:/dataset/experiment/HRDecoder_FinalDecision_20261005/execution/server_results/configs/protocol.json) 的 `$.training.updates_per_arm=32000`、`$.training.warmup_updates=500`、`$.validation_and_selection.deployment_candidate_updates=[20000,32000]`；最终锁选 u20000。 [schedule.py](D:/dataset/experiment/HRDecoder_FinalDecision_20261005/execution/server_results/schedule.py:7) 第 7–47 行记录 AdamW 参数组：DINO block 峰值 LR 按 `1e-5*0.8^(11-b)`，旧 decoder `5e-5`，空间模块和空间残差 `1e-4`；第 50–55 行是500步 warmup 后余弦降至峰值的1%。EMA 是完整模型副本，衰减0.999，见 [state.py](D:/dataset/experiment/HRDecoder_FinalDecision_20261005/execution/server_results/state.py:10) 第 10–27 行。
- S 不使用 TJDR。协议 `$.arms.S.TJDR_package=false`；[train_run.py](D:/dataset/experiment/HRDecoder_FinalDecision_20261005/execution/server_results/train_run.py:39) 第 39–47 行常规加载 DDR 数据，只在 D/SD 臂建立 TJDR loader；[engine.py](D:/dataset/experiment/HRDecoder_FinalDecision_20261005/execution/server_results/engine.py:8) 第 8–26 行对 S 路径用 DDR 图像、标签和有效像素执行 `masked_loss`。因此共享缓存记录中出现外部数据，不等于 S 用其训练。S 的任务训练数据是 DDR train（协议 split 454 张）；另有 DINO LVD1689M 通用预训练来源。

## 训练集输出资产

- 当前**没有可直接复用的训练集 DINO 特征、空间图或 S@20k 软概率缓存**。 [runs/S_seed42/cache.json](D:/dataset/experiment/HRDecoder_FinalDecision_20261005/execution/server_results/runs/S_seed42/cache.json) 的 `$.training` 明确是893张的 `RGB uint8 + mapped label uint8` 原始数据共享缓存；`$.P0` 是76张原始分辨率概率。 [inputs.py](D:/dataset/experiment/HRDecoder_FinalDecision_20261005/execution/server_results/inputs.py:57) 第 57–85 行说明 `cache_sources` 只缓存图像和标签，`Windows` 第 95–111 行再按窗口/增强生成训练输入及标签。
- S@20k/32k 的 `.npz` 位于 `runs/S_seed42/val_u20000` 和 `val_u32000`，每个各76张验证概率；其数据身份见相应 `metrics.json` 的 `split=val`、`n_images=76`。它们不能作为 DDR train 的软目标。目录抽查只发现这些验证 `.npz` 和 checkpoint，没有 train 概率或特征文件。
- 如要生成 T_S20 的 DDR train logits，可用锁定 checkpoint 的全模型 EMA 对 DDR 训练图像作一次新导出；若要隐藏 spatial/ViT 特征，则还要给当前只返回 logits 的 `FinalModel.forward` 加导出路径或挂钩。仅缓存的 RGB/标签不能代替这一步。由于 S 中 backbone 和 spatial adapter 都可训练，DINO 预训练输出、Q/F12 旧输出或早期 S 特征都不是 S@20k EMA 的等价缓存。本审计没有执行该导出或任何模型推理。

## 对“约10万参数头”的解释边界

本轮事实足以纠正教师身份：S@20k 是约101.29M参数的 DINOv3 分割全模型，而不是一个100k decoder head；其模型输出还在最终锁定系统中占25%。但“学生约100k的新头能否吸收三路异构教师行为”不能从教师参数量直接判定。若100k只指学生新增头，学生整体还须加上 HRNet 主干参数；能否匹配三目标需要在独立验证协议上测量。当前目录也没有可供三目标直接训练的 train 输出缓存。
