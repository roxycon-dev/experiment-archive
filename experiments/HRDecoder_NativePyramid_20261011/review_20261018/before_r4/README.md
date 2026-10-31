# HRDecoder：v3实验记录与JointContext架构设计

**当前设计r3（2026-10-18）：** [07 HRDecoder-JointContext修订方案](07_HRDecoder_JointContext_最终架构方案_20261017.md)。主代理本人进行架构专项复审，将两尺度共用的720→64投影拆成同值初始化、独立训练的P_G/P_N；保留同一W48、分类前共同/差异融合及统一两图平方Dice。修订解除共同通道子空间限制，并明确函数族包含关系与40k指标提升之间缺少的论证。**完整新模型尚未实现或训练，当前Python仍是已完成的NativePyramid v3，不能按新架构名称直接启动旧入口。** r3是设计修订号，不是该已训练v3。

当前交付：[r3机器规格](review_20261018/jointcontext_spec_r3.json)、[09架构专项审查与修订](09_架构专项审查与修订_20261018.md)、[架构代数复算](review_20261018/architecture_capacity_audit.json)。本次未运行模型或优化器，没有新mIoU，不能将结构复算视作达标验收。

历史交付：[v2机器规格](review_20261018/jointcontext_spec_v2.json)、[08主代理复审](08_主代理反复审查与修正_20261018.md)、[v2小型CPU计算图检查](review_20261018/root_adversarial_checks.json)及[v2原稿快照](review_20261018/before_r3/07_HRDecoder_JointContext_最终架构方案_20261017.md)。这些检查不包含r3独立投影模型，不能自动移作r3运行验收。10月17日的12项代数检查属于v1历史；[首次修订前快照](review_20261018/before_revision/07_HRDecoder_JointContext_最终架构方案_20261017.md)已保留。[05候选方案](05_HRDecoder_DualScale_架构定稿_20261017.md)及[06主代理独立审查](06_主代理独立审查_20261017.md)仍为历史，原NativePyramid v3执行记录和源码不改写。

用户授权后，已从原服务器只读处理完整40,000条`train.jsonl`，核验连续更新编号并保存关键字段投影及原文件SHA；相关结果见[数学与训练日志证据](review_20261017/redesign_math_evidence.md)。

**2026-10-17最新状态：** 实际v3运行`run_seed42_launch02`完成40,000次更新，固定终点raw val mIoU **39.425551%**，低于45.321831%目标 **5.896280个百分点**；best-val为36k的40.277046%。test未访问。本配方未通过性能验收，原样v3不再作为下一轮默认训练推荐。

最新结论见[40k实测复盘](04_v3_训练复盘_20261017.md)，原始结果见[训练完成记录](server_execution_20261016/completion_20261017/训练完成记录.md)。下列内容保留2026-10-16的准备与执行协议；“尚未训练”等措辞属于当时状态。运行时设计文档02属于源码身份清单，保留原字节，最新实测状态以本段和复盘为准。

## 2026-10-16准备记录

2026-10-16。当前 Python 已按 [v3 最终设计](02_v3_最终架构设计_20261016.md)实现，执行细节继承 [v2 完整协议](00_最终架构方案.md)。本轮只准备代码、运行合成输入的单元检查及子代理审查；没有运行真实数据学习门、2048 SGD 或正式训练，没有新的 val/test 成绩。

本地验收、独立审查及待服务器执行的事项见 [准备记录](preparation_20261016/准备完成与服务器交接.md)。原 v1 说明和本轮改动前的相关文件保留在 `preparation_20261016/legacy_v1/`；历史 `evidence*` 不作为 v3 预检证据。

## 已实现的固定配方

- 一个 ImageNet HRNet-W48，2048 画布直接入主干；主干 BN running statistics 固定、affine 可训练。
- 四尺度无 bias 线性分类，共享唯一四类 bias；逐级非线性语义解码及 stem 细节支路。C/D 末层零初始化，无裸 RGB 分支，解码器采用 FP32 逐像素通道 LayerNorm。
- 解码器 **246,132** 参数，全模型 **65,571,252**；1830 个主干张量逐键加载，明确排除 126 个分类张量，31 处融合改双线性。
- SGD 40,000 更新、global batch2、80,000 图次；正式运行 micro batch1，两个微批共享完整 batch 的 P/K/A/V 分母。学习率、warmup、增强、损失及 seed 由 `protocol.json` 锁定。
- 正常反传记录各模块及四尺度×四类别裁剪前后梯度范数；每250更新固定8张训练整图探针，每2000更新完整val76。S/Z分别累计四类TP/FP/FN，探针另记阳性logit分位、直接分类幅度及C/D幅度。
- 探针和验证分别建立连续两节点的学习证据，再判断连续丢失；持续未学会和验证回落按设计保存完整边界并 `PAUSED`。数值/OOM/状态异常写终态失败标记，禁止从旧 `last.pt` 回滚续跑。

标签为 0背景、1 EX、2 HE、3 SE、4 MA、255忽略，沿用454/76/227融合标签划分。唯一接受的数据清单放在 `preparation_20261016/data_identity.json`，SHA256为 `7c53b86bcb27ccd121042f667cc58e39b98716a8c477bfd1f1657fa126575719`。指定 W48 文件 SHA256为 `d2186c55782308392ae14cf179ac4840143b09abdbba81aa3f9274c795ebc960`。

## 本地结构检查

使用已有依赖，或按 `requirements.txt` 准备环境。服务器的 Python、依赖实际版本、CUDA/cuDNN、GPU和数值开关会由预检重新记录，并与正式运行精确比较。本地 Windows 结果不能代替服务器预检。

```powershell
python verify.py --static-only --output preparation_20261016/static_checks_new
```

此入口不做真实数据 SGD，只输出 `STATIC_CHECKS_PASSED_SERVER_GATES_PENDING`，不能取得正式训练所需的 `PASS`。输出目录必须全新。合成单元测试中的小模型 SGD 用于验证微批与状态恢复，不计为真实诊断或正式训练。

## 服务器上的执行顺序

服务器提供后先检查 GPU/BF16、空间、数据与权重，使用全新的目录运行下面的真实预检。以下命令是交接说明，本轮没有执行。

```bash
python verify.py --data-root /path/to/DDR_split_613 --identity preparation_20261016/data_identity.json --pretrained /path/to/hrnetv2_w48-d2186c55.pth --output /path/to/v3_preflight
```

预检必须全部通过：

1. 单元检查、实际训练/验证文件 SHA、预训练逐键身份、固定 BN、重算检查。
2. 实际 W48 在 FP32/BF16 下，初始零残差与确定性非零残差 fixture 的 S/Z/loss、全部参数梯度：无重算、重算、完整batch及微批对照。fixture不做SGD，随后丢弃。
3. 固定8个真实256裁块，每类至少2个阳性裁块；最多256更新，224/256两节点每类IoU均≥20%，末16次loss均值低于首16次。
4. 固定真实2048图组保留四类，每类至少2张阳性图；8次global batch2、micro1更新。记录首次C/D末层与主干直路任务梯度、末层启动后内部任务梯度及参数变化、BN buffer不变、BN affine变化、逐类响应和实际显存峰值。

诊断权重全部丢弃，更新和失败成本另记。任何门失败都保留原状态和证据；不得提高步数、降低门槛或静默换精度后沿用同一预检结果。

只有同一源码/数据/权重/依赖/硬件身份下完整预检 `PASS`，正式入口才允许开始：

```bash
python train.py --data-root /path/to/DDR_split_613 --identity preparation_20261016/data_identity.json --pretrained /path/to/hrnetv2_w48-d2186c55.pth --preflight /path/to/v3_preflight/preflight.json --output /path/to/v3_seed42 --micro-batch 1
```

正式输出目录也必须全新。逐更新 `progress.json` 和 `train.jsonl` 区分已执行更新数、已持久化检查点数、尚未确认的更新尝试；源码、验证器、测试、协议和依赖清单共同参与 SHA 身份锁。

## 暂停、失败与终点

只有保存完整 SGD 边界的 `PAUSED` 可以在诊断后原配方续接。将 `result.json` 中当前的 pause_id 传入，其他参数保持原值：

```bash
python train.py --data-root /path/to/DDR_split_613 --identity preparation_20261016/data_identity.json --pretrained /path/to/hrnetv2_w48-d2186c55.pth --preflight /path/to/v3_preflight/preflight.json --output /path/to/v3_seed42 --micro-batch 1 --resume --resume-pause-id CURRENT_PAUSE_ID
```

存在 `failure.json`、未解决的 in-flight 更新、执行账本与检查点不一致、源码/环境/参数变化时均拒绝续接。进程意外中断不能退回旧检查点补跑并仍称恰好40k；已花费与不确定成本要单列。

若第40000更新正好触发健康告警，同样保存为 `PAUSED`，保留40k raw及完整val，并标记 `endpoint_finalize_only`。诊断后以当前pause_id确认续接时，剩余采样为空，只完成终点锁定，不增加SGD、不重复val，也不自动执行test。`run_active.json`为独占运行租约；硬杀遗留租约不会自动清理后重跑。

主结果为固定40k raw，best-val仅作次要结果。val mIoU≥45.321831227388%后，锁定评估入口才会读取test227的实际图像/标签字节并逐文件验SHA，执行一次test。原尺寸FP32 logits先插值再sigmoid，严格>0.5，无TTA/EMA或阈值搜索，累计四类IoU后取均值。

```bash
python evaluate.py --lock /path/to/v3_seed42/endpoint_lock.json --data-root /path/to/DDR_split_613 --identity preparation_20261016/data_identity.json --pretrained /path/to/hrnetv2_w48-d2186c55.pth
```

test达标线为45.860383832668%。等于称达标，严格大于才称超越。评估绑定同一运行的40k检查点、内部验证记录及执行账本，`test_started.json`独占创建后不允许重复评测。所有成绩仍待未来执行。
