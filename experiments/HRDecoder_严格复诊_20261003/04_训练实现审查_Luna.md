# HRDecoder 训练实现审查（Luna）

审查范围：新根 D:\dataset\experiment\HRDecoder_Ready_20261002 与历史根 D:\dataset\experiment\HRDecoder_FullCompetition_20260920 中的训练引擎、损失、初始化/优化器、检查点、随机流、数据增强和相关运行日志。只读审查，并使用了一个小型 CPU 合成张量对照；没有启动训练、连接服务器或修改代码。下文未重复根前缀的路径均相对新根，注明“旧根”的路径相对历史根。模型几何本报告不作判断。

## 结论先行

确认了一项会影响有效 batch 对照解释的数学语义：Dice 在每个 microbatch 独立算比率后再平均，所以相同 effective batch 的 8×1 与 4×2 不是同一个 loss/gradient。它不是自动构成缺陷；若比较组固定同一种 microbatch 分组，语义一致。新引擎已经把旧引擎“累积梯度等价于大 batch”的错误注释改正。

没有发现 A0 训练 loss 持续发散或 logits 非有限的证据。A0 loss 从 1.663@u100 降到 0.937@u3000、0.891@u6000；验证 mIoU 从 0.0695 升到 0.0968，随后运行在 u6197 被明确标记为操作员停止。因此不能据这段日志判定已收敛，也没有证据说自动停止误判了它。随机初始化、decoder LR 为 backbone 的 15 倍、BN 运行统计更新都值得用小对照检查，但目前只是可能解释。

停止规则属于探索性预算筛选，并不等于收敛检验。U0 2048 对照在两个验证点后提前停止有后续反弹未被观察到的风险，但现有数据不能证明它若继续会恢复。H1 则暴露一个确定的状态标签问题：完整跑满 12,000 updates 后，最终检查点触发低优先级规则，summary 仍写成 status=stopped。

## 可确认的实现语义与风险

### 1. 梯度累积的 Dice 不是全局有效 batch Dice

trainer/engine.py:193-203, 244-264 对每个 microbatch 单独调用模型和 loss，再除以 grad_accum 后反传；trainer/loss.py:61-85 在当前 batch 上计算有效类别数、BCE 归一和 Dice 交并比。实际目标是“microbatch 标量 loss 的均值”，不是把累积窗口拼成大 batch 后再算一次 Dice。有效类别在不同 microbatch 出现与否时，类别权重也会不同。

CPU 合成对照通过附件 audit_dice_accumulation.py 直接调用项目 masked_bce_dice_loss。第一例 shape=[2,4,1,2]，microbatch mean loss=0.80032232、pooled loss=0.82365143、最大梯度差=0.09799211；但此例各 microbatch 的有效类别不同，BCE mean=0.45140591 与 pooled BCE=0.46058407 也不同，所以总差异不能单独归给 Dice。第二例 shape=[2,4,1,4]，两 microbatch 的 4 类全部有效且像素数相同，前景密度分别为 0.5 与 0.25；microbatch mean BCE=0.58673617、pooled BCE=0.58673620（浮点舍入差约 3e-8），而 mean Dice loss=0.50378725、pooled Dice loss=0.49261928（差 0.01116797），最大梯度差=0.00417471。这单独验证了相同 BCE 下，Dice 比率的 microbatch 均值与 pooled ratio 不等价。脚本固定全部输入并把两例及梯度写入 audit_dice_accumulation.json。新引擎 engine.py:254-256 已明确注释“不代数等价”；历史引擎 D:\dataset\experiment\HRDecoder_FullCompetition_20260920\trainer\engine.py:230-232 曾声称等价，但计算代码仍是同样的逐 microbatch loss 累积。新旧行为在此没有变化，变的是说明准确性。

本批实验中，U0 高分辨率与 2048 对照都是 microbatch 4、accum 2；A0 与 H1 DDR 是 8、1；H1 外部预训练是 4、2。只匹配 effective batch 不足以匹配目标函数。最低成本做法是固定关键对照的 microbatch/accum；若不得不改分组，应在报告中明确这是目标函数变化。若要实现真正的大 batch Dice，需要跨整个累积窗口池化 Dice 统计，不能只把现有标量 loss 平均。

### 2. H1 的训练调用和评估调用是两条不同路径

训练引擎 trainer/engine.py:193-203 调用 self.model(image)，并只把返回 logits 交给共同 loss。HRDecoder adapter 的 _Student.forward 在 trainer/models/hrdecoder_adapter.py:164-209 返回训练 logits；独立的 eval_forward 从 :211 起实现。评估器在 trainer/eval.py:141-145 使用 eval_forward 或滑窗版本。静态调用链确认：H1 的训练没有调用 eval-only 路径，也没有额外的 HR representation/aux crop loss。

这说明现有 H1 结果评估的是当前 adapter 的训练/推理组合，不能当作对未完成原始 RGB stride1/2 全局局部特征融合方案的否定；本审查不判断这两条路径的几何是否正确。

### 3. 真实权重加载：基础权重有直接日志；H1 病灶外部权重有强路径证据但缺少逐键日志

- U0 ConvNeXt 运行日志明确记载官方编码器加载 340 个键、missing 为空，unexpected 仅为分类头/归一化头：round2_highres\server_results\work_dirs\U0_highres\train.log:7。U0 对照在 D:\dataset\experiment\HRDecoder_Ready_20261002\round2_highres\server_results\work_dirs\U0_continuation2048\train.log:7 同样如此。H1 HRNet ImageNet 载入 1,830 个键、missing 为空，见 server_results\work_dirs\H1_ddr_finetune\train.log:9。
- H1 DDR 的 resolved 配置 server_results\configs\resolved\H1_hrdecoder_ext_pretrain.yaml:20-22 指向 H1_external_pretrain/ckpt_best.pth 并指定 SHA256；本地文件 SHA256 3aa43faa5ad38f0c6409ba8b518cb8783d319f5b8a5e4b8dce7ccfd25cfaf408 与配置值吻合。该配置文件 SHA256 1b2de9719d17d9ba1a1ea9b9ba20c2dc839f0ea1f2e573ad1f5b0d4a1c5965f5 与 H1 DDR 完成日志 train.log:139 的 config_file_sha256 一致。trainer/models/__init__.py:47-61 对 DDR 外部初始化执行同家族、正式阶段和 SHA 校验，加载失败会中止；运行已完成 12k updates。因此这是强路径证据，且很可能已走过载入；但现存材料不能确认该特定 lesion checkpoint 已成功写入模型。此外，campaign_state.json:85-89 记录 H1_ddr_finetune.config_sha256=2bf116be1cd68dabe78b7685d5d2afe416d1c5b781329648a036616dd9347dac，与 completion_summary.json:13、train.log:139 和本地 resolved YAML 的 1b2de9719d17d9ba1a1ea9b9ba20c2dc839f0ea1f2e573ad1f5b0d4a1c5965f5 不一致；本地材料没有实际启动配置副本，无法消解来源链冲突。限制是 train.py:396-399 只打印 meta["weight_report"]，没有打印 external_weight_report，故日志没有外部 checkpoint 的逐键加载数。
- A0 的 hacdr_adapter.py:44-48 构造 backbone 后调用 init_weights()；A0 resolved 配置 server_results\configs\resolved\A0_hacdr_adapted.yaml:14 的外部预训练 checkpoint 为 null，训练日志 A0_ddr_finetune\train.log:5-6 记载 init cfg None 且 norm_eval=False。这支持“没有载入 lesion/外部预训练权重”的结论，但不等于逐参数验证了每个初始化分布。

### 4. A0 loss、极端 logits 与 LR/BN 证据

A0 实际日志中 micro=8, accum=1，frozen_bn=0，见 server_results\work_dirs\A0_ddr_finetune\train.log:3,10；config 的 backbone/decoder LR 分别为 6e-5 和 9e-4（server_results\configs\resolved\A0_hacdr_adapted.yaml:40-53），比值为 15。hacdr_adapter.py:46 使用 BN，且该 run 的 norm_eval=False，因此训练态会更新 BN 统计量。

server_results\reports\A0_train_crop_diagnosis.json:510-629 中选定 crop 的 eval_fp32 与 batch_stats_fp32 都是 finite；其 logits 极值分别约为 [-137.88, 935.08] 和 [-155.50, 1008.16]。这是极端但有限的 logits；单凭范围无法归因于 LR、BN 或模型错误。trainer/loss.py:49-54, 58-85 强制 logits/target/valid 转 fp32，用 binary_cross_entropy_with_logits 并在 fp32 做 Dice reduction，因此未见“半精度像素级 reduction 溢出”的实现证据。极端 logits 会使 sigmoid 概率接近饱和，但这里没有梯度统计证明这导致训练失败。

低成本验证：在同一固定训练 crop 上记录 logit 的分位数、loss/梯度范数；分别在 train/eval BN 状态运行一次前向并比较；再只改变 decoder LR 或 BN 统计策略做短程受控校准。现有 A0 运行是操作员在 u6197 停止（train.log:76），不是基于这些假设的消融，故不能进一步归因。

### 5. U0 warm-start、重启优化器和早期峰值

U0 第二轮配置以 U0_historical_best.pth 做模型 warm start，设置 4,000 updates、warmup 200、backbone/decoder LR 为 2e-5/6e-5、microbatch 4×accum 2：见 round2_highres\server_results\configs\U0_highres.yaml:26-66 和对应对照配置。trainer/models/__init__.py:47-73 明确把 warm-start 视为只加载模型权重；train.py:420-429 随后新建 optimizer，trainer/optim.py:66-71 建立新的 AdamW。因此这是“模型 warm-start + fresh AdamW/scheduler”，不是从历史训练状态续训。历史 U0 原配置用 1e-4/3e-4（旧根 configs\U0_convnext_unet_ddr_only.yaml:47-60）；当前 LR 反而较低。

结果方面：历史 U0 的验证 mIoU 从 u3000 的 0.1748 增至 u6000 的 0.3500、u9000 的 0.3744、u12000 的 0.3845（旧根 runs\train.log:40-134）；新 U0 高分辨率在 u1000 为 0.3961，u2000 为 0.3749，u4000 为 0.3849（U0_highres\train.log:21,32,53）。新对照在 u1000 为 0.3909，u2000 为 0.3780 后被筛选停止（U0_continuation2048\train.log:49-62）。

fresh AdamW 状态和重置短 schedule 是合理待测因素，但不能单凭“早峰”判为根因：新旧阶段本身不是匹配试验，当前高分辨率组与对照还改变 resize side。多 side 配置在 augment.py:74 多消耗一个 RNG 抽样，其后随机 resize ratio、裁剪和增强随机数也会错位；相同 seed 不表示两组逐样本增强完全配对。要归因 fresh optimizer，应在同一模型初始权重、同一 batch 序列、同一 LR schedule 下，只对照恢复 AdamW moments 与 fresh moments。

### 6. AMP、微步计数和数据曝光/随机流

引擎只在 CUDA 且配置开启时进入 fp16 autocast（engine.py:116-121, 193-206）；loss reductions 转 fp32。优化器更新前 GradScaler unscale 后裁剪；scale 下降时跳过 optimizer update，再取新的 microbatch 窗口（engine.py:216-235, 251-270）。完成日志中 micro_steps 高于 updates×grad_accum：例如 A0 的 6,197 updates 对 6,200 micro_steps，U0 高分辨率 4,000 对 8,010。这与少量 scaler skip 相符，但单靠 summary 不能完全证明每个额外微步的原因；amp_skips 目前未写入 summary/checkpoint，建议持久化计数，避免以后只能倒推。

seed.py 保存 Python/NumPy/Torch/CUDA 全局 RNG；fast-kernel 配置会把 cudnn deterministic 关闭并启用 benchmark/TF32（train.py:382-387，seed.py:16-24），因此“同 seed”不保证位级可复现。新 DatasetRuntime 共享 epoch 并缓存 raw array，DDR 增强 RNG 用 worker seed、样本 index、epoch（common.py:114-155，ddr_dataset.py:116-121）；这是改善，不是问题。另一方面，checkpoint 保存 epoch/global RNG，但不保存 DataLoader sampler 的当前排列和游标，engine.py:147-183, 413-437 也未持久化 external dataset 每 worker 的 draw 计数（external_dataset.py:206-225）。从非 epoch 边界恢复不能精确重放剩余样本/增强序列。建议要么保存 sampler/generator 与游标，要么在报告中明确 resume 是状态续训但非逐样本重放。

外部预训练使用 source-balanced 权重 1/count(source) 与 WeightedRandomSampler(replacement=True)（external_dataset.py:273-281, engine.py:150-154），其含义是平衡数据源，而不是让所有图像等频。H1 外部预训练实际 48 张图、2,400 updates、micro 4×accum 2（H1_external_pretrain\train.log:2-3,11），名义上是 19,200 次 draw；运行 summary 的 4,804 micro_steps 对应约 19,216 次取样（含被跳过 update 窗口的取数）。较小数据源会因此被更频繁重抽，这是明确的设计曝光差异。记录每个 source/image draw 次数可帮助排除曝光量解释。

## 停止门槛：能确认什么、不能确认什么

trainer/screening.py:2-23 至少要求两个完整验证节点，且 current update 达到 total 的 50% 才允许 stop；U0 continuation 分支比较最近两次 fusion/solo 增益，阈值为 0.001。U0_continuation2048/train.log:49-62 恰在 1k、2k 两点后停止。这是小样本验证节点上的探索性预算筛选，不含统计误差界，存在停止过早的风险，但两点不足以证明它错误地判作未收敛或若继续会恢复。历史 U0 的长训练曲线说明“训练可能在较后阶段仍有提升”是合理假设，不是该新对照的反事实证据。

H1 的最终验证在 u12000 后产生 low_priority 并触发 break；engine.py:352-360 无论实际是否已到总步数，只要 stop_reason 非空都写 status=stopped。H1 日志 H1_ddr_finetune/train.log:137-139 表明它实际 updates_done=12000、已跑完整计划，故这里是确定的 summary 标签不准确，不是提前中止训练。

另外，engine 初始化 eval_reports=[]（engine.py:127），恢复只还原 update/micro_step/epoch/best（engine.py:441-458），checkpoint 的 extra 仅传 best/epoch（engine.py:413-437），但筛选依赖内存中的历史 eval reports（engine.py:345-346, 389-395）。重启后旧验证点没有灌回 stop rule；现有的“边界验证缺失则补跑”只处理当前恢复点对应的 eval JSON，不会恢复更早点。建议从 eval_history.csv 或 checkpoint 恢复历史验证序列，并在真正到达 total updates 时报告 complete，把“筛选触发”作为单独字段。

## 与历史 U0 的训练实现差异摘要

1. 新引擎修正了旧引擎关于 Dice 累积等价大 batch 的错误注释；逐 microbatch loss 的实现语义保持不变（新 engine.py:254-257 vs 旧 engine.py:230-232）。
2. 新 loss 对 logits/target/valid 都显式转 FP32（新 loss.py:49-54）；旧 loss 只把 target/valid 转为 logits.dtype，所以 autocast 时 reduction 有机会落在半精度。这是有利的数值保护。
3. 新数据路径把 DDR targets 存成 uint8，添加共享 epoch/cache runtime（data/common.py:114-155）；新增强增加多 resize side 选择（data/augment.py:72-78），会改变随机流消费。
4. 新 ConvNeXt 把原来 checkpoint_sequential(..., segments=1) 换成逐 block checkpoint（新 convnext_unet.py:158-163；旧实现 :156-160），修复了旧写法基本不起 checkpoint 的问题；但 U0 这轮配置 gradient_checkpointing=false，不能解释本轮曲线。
5. 本次 diff 中 optim.py 和 checkpoint.py 的核心实现未见新旧差异；fresh optimizer 是 warm-start 入口/调用顺序的设计，而不是本轮刚出现的 AdamW 算法改动。

## 建议的低成本后续核验

- 将关键实验的 microbatch 与 accumulation 固定，或显式记录 loss 分组语义；保持 effective batch 相同并不足够。
- 用附件中的 CPU 合成梯度对照作为 regression check；all-valid 案例隔离出 BCE 相等而 Dice 不等，但不证明模型收敛性。
- 把 external_weight_report、AMP skip 计数、eval history 与 data sampler state 持久化；明确区分 warm-start、optimizer-state resume 和 exact sample-stream replay。
- A0 先做固定训练 crop 上的短诊断，观察梯度范数和 BN 统计；随后一次只改 decoder LR 或 BN 策略。现有信息不足以单独降低头部 LR 或冻结 BN 并声称已定位原因。
- 将 screening 输出解释为“预算优先级”而非“未收敛/已收敛判断”；至少恢复历史验证点并正确区分“正常跑满”与“提前筛停”。



