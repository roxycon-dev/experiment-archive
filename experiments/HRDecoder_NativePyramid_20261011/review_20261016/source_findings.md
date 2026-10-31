# 源码核查发现

审阅范围截至 2026-10-16。此项工作只读源码、协议身份、运行记录与公开原始论文/官方实现；未实例化模型、未执行 forward、未训练、未跑 SGD、未访问新的 validation/test 像素、未反序列化权重/检查点、未连接服务器。

## 覆盖边界与身份口径

全树冻结库存包含 37,487 个文件、1,940 个目录；准确字节数是 101,017,506,982。常规库存列出 1,323 条源码路径、1,146 个内容 SHA；第三方或归档源码 4,015 路径、1,541 个 SHA；另有 39 个归档容器。SHA 只用于认出相同内容，不能证明阅读完成。归档成员 68 路径/52 SHA 按成员读取、哈希核验后逐项标成全文、差异全文或 AST/索引；后两种状态不冒充人工全文阅读。逐路径状态见 [source_coverage.json](D:/dataset/experiment/HRDecoder_NativePyramid_20261011/review_20261016/source_coverage.json) 和 [archive_only_source_members.csv](D:/dataset/experiment/HRDecoder_NativePyramid_20261011/review_20261016/archive_only_source_members.csv)。

十一个正式或筛选运行协议记录了 43 个不同代码 SHA；这些精确代码版本全文审阅，逐个按 `protocol.json` 的 SHA 认定，未用邻近快照代替。协议覆盖的版本族包括 adapter、aligned、CET、Detail、expert、foundation、IoU、residual、SACD、semantic 和旧 v3。CET 是唯一完成正式 40k 的 ArchitectureDesign 运行；Detail 和旧 v3 是不同身份的运行，均在崩塌后由用户停止。比较表记录到 20k 时已严重退化；实际运行均未到 40k，停止节点与检查类型以 runs 审计及原始状态为准。其余对应筛选/短程或诊断臂，不把计划步数写成实际完成数。运行事实和逐协议身份另见 [runs_coverage.json](D:/dataset/experiment/HRDecoder_NativePyramid_20261011/review_20261016/runs_coverage.json) 与 [runs_findings.md](D:/dataset/experiment/HRDecoder_NativePyramid_20261011/review_20261016/runs_findings.md)。

Next 当前代码按独立 SHA 全文审阅；其中 model/data/evaluate/train 四个 SHA 是 Next 独有版本，adapter/loss/engine 与 Ready 字节相同，按 SHA 复用全文审阅。版本身份不能倒推：Next 的配置清单与后续 closed-node M 元数据绑定过不同版本；旧服务器 `tuple-dim any()` 异常属于较早 train SHA，不能归给修订版。Next 的某些源是执行候选，不因代码已读就推成正式成绩的运行源。

同理，H0 停止元数据、D_0913_retry 与 B0 缺少足够的完整逐文件 SHA 链，覆盖其候选源不等于证明该候选就是实际执行源。H0 元数据显示它是 1024 输入、effective batch 8、AdamW、6k 停止，与本方案 2048/batch2/40k-SGD 不是预算匹配。D_0913_retry 的 20k 结果来自 B@3k 初始化，并将局部学生和冻结 B 参考概率融合；不可称作纯单 W48 解码器结果。

## 前向、初始化与监督的源码事实

原 HRNet-W48 的 segmentation 接口产生四个尺度：通道 48/96/192/384，空间约为输入的 1/4、1/8、1/16、1/32；原始 HRDecoder segmentor 把低分辨率全图分支与局部高分辨率裁块分支结合。当前 W48 配置继承 `hrdecoder_fcn_hr18.py` 的 `compress=True`：全 720 通道先经 1×1 `ConvModule` 压至 64，再经 64→64 的 7×7 `ConvModule`，之后分类为四通道；当前配置的低分辨率损失权重为 0、高分辨率裁块损失权重为 0.1。`FCNHead` 的两个 `ConvModule` 都含配置的归一化/激活；它不是一个纯线性 720→4 映射。

H0 候选适配器 `HRDecoderStudent` 则把四尺度特征双线性对齐至 branch-0 后拼成 720 通道，调用 `FCNHead(in_channels=720, channels=64, kernel_size=7, num_convs=1)`，未启用 `compress`。所以其特征处理是 720→64 的 7×7 `ConvModule`，再分类到四 logits并上采样；它与原配置的 1×1压缩+64→64 7×7 是两个不同 head。候选实现的 `FCNHead` 使用 BN/激活；H0实际完整 source SHA 未被停止元数据绑定。候选训练目标在 FullCompetition 的 `trainer/loss.py` 是逐类有效像素 masked BCE+soft Dice；这仍不能证明停止运行的实际 loss SHA。原 HRDecoder 则按 `BinaryLoss(type='dice')` 与上述 HR crop 辅助项训练。H0 与原论文头结构不能混称。

当前 NativePyramid Python 是失败的 v1，不是待实现 v3。v1 的 W48 预训练权重用逐键严格载入指定 ImageNet 文件；分类头额外键被明确排除。主干设置训练模式后，将 BatchNorm2d 切回 eval，冻结 running mean/variance，但保留 affine 的梯度。v1 decoder 的特征通路为 F4 384→128，逐级拼接 F3/F2/F1 后依次压到 128/96/64，并各有局部/上下文细化；语义 aux 是随机初始化的 64→4。细节路径将 64 通道语义上采样、stem 投影64→32、裸 RGB PixelUnshuffle(2) 后 12→16 拼为112通道，压至48并两次细化，以48→16、PixelShuffle(2)产生四张2048 logits。每个细化块的 local/context depthwise 3×3与7×7、GN、1×1混合与残差明确写在 [model.py](D:/dataset/experiment/HRDecoder_NativePyramid_20261011/model.py)。

v2/v3架构本轮只在 Markdown 中，不在当前 Python 源码中。v3 的直接读出 `S_direct=b+Σ U_i(W_i F_i)` 中，每个 `W_i` 为无 bias、无 Norm/激活的 1×1 `C_i→4`。由于卷积与同一线性双线性 resize 可交换，精确算术下它等价于四尺度上采样拼接后一个720→4线性分类器；它不是原 HRDecoder 的非线性 FCN head。保留的非线性 `F4→128→128→96→64` 路径仍提供跨尺度非线性交互，`C(G)`形成辅助语义修正，stem+G 进入 48通道细节头，输出 `Z=U(S)+D`。结构上可保证每尺度有短监督路径、零末层令初始化 `C=D=0` 且 `Z=U(S_direct)`；不能保证所有原 feature 信息无损、随机读出已经学会 DDR 类别语义、或 mIoU 上升。

静态计数按 v1 源码各卷积/GN/细化块公式得到 decoder 244,244、W48主干 65,325,120；v3 删除 RGB 投影 224、112→48 融合块相对 96→48 多出的 768 参数、旧 aux bias 4，加四个 1×1 direct classifiers 的 2,880 权重及一份 4 类 bias，总增 1,888。故 v3 decoder 246,132、全模型 65,571,252。此为公式核数，不是模型实例化或运行报告。`S_direct` 只允许一份共享 bias；若为四个 Wi 各加 class bias，四份 bias 在 S 中仅以和出现并收到相同任务梯度，属于不可辨识冗余。C bias 与该共享 bias 同理，删去准确。D的16个 PixelShuffle相位bias则有逐相位能力；四相位相同的情形可在最终 logit 上模拟统一偏置，但因仅影响 Z、而 S 单独受辅助监督，训练梯度作用并不相同。

## 崩塌机制与浅层支路证据边界

Detail 与旧 v3 都曾采用 zero-last residual，随后仍严重退化；这反证的是“零初始化单独保证长期稳定”，不证明零初始化造成崩塌。Detail 的 4k 对照是在同一个模型/检查点中比较 private 与 base 两头，不是训练消融：MA TP 由25,245降至21,810，FN由44,039升至47,474、FP从1,326,940降至63,322；后续16k时SE已零检出。旧 v3 到20k时EX与SE TP为0。两次均没有查实源代码不一致、非有限 loss/optimizer state 或 BN 统计交叉是原因。训练探针在晚期 checkpoint 发现饱和 SE/EX 的正类 Dice logit 梯度小于平衡 BCE 约多个数量级，说明这两种损失在饱和区局部导数不同；探针没有 optimizer step，不能证明全局 SGD 因果方向，也不能证明改某种 loss/Norm 可以救回训练。

历史 adapter 与 Detail 私有支路都把多尺度特征及 base probability/logit detach 后送入 private head，且把 raw RGB 同概率/语义输入并用；adapter 4k有初期收益，Detail 4k收益也伴随 FP 大幅下降、TP下降，后续 Detail 仍退化。没有“只开裸 RGB PixelUnshuffle，其它相同”的正式对照。因此当前v1裸RGB支路既无独立有效性证据，也无证据证明它导致假阳性；v3删它是简化未经隔离验证的分支、并让预训练stem承担浅层输入的候选改动。预训练stem联合解码也没有单独消融证据。

GN↔channel-LN 没有任务内隔离训练结果。GN的统计跨样本独立，但标准NCHW GroupNorm仍跨组内通道及空间计算；通道LN若按 NHWC `LayerNorm(C)`，才是每个像素单独按C个值归一，排除batch和空间。它能移除“忽略区面积改变单次统计量”这一特定依赖，不保证卷积边界、插值、主干BN域差或整个网络都padding不变。每种Norm是否改善当前病灶和损失仍需正式实验。

## 对 v3 草案的独立数学审查

最新 v3 草案参数、通道、尺度、共享bias与两次 resize 约束相互一致。对 `W_i`，在所有尺度采用同一 `align_corners=False`线性插值、纯无bias点卷积、没有Norm/激活时，`U(W_i F_i)=W_i U(F_i)`；不同插值顺序 `Fi→512→2048` 与直接 `Fi→2048` 通常不等价。梯度审查应看 `W_i.weight.grad[c]` 的四尺度×四类范数，且 clip 前后分别记录；模块梯度范数无法判别共享主干中不同损失梯度的方向抵消，现稿已删除此暗示。

C、D的末层全零使首次 task backward 对其末层上游特征路径的梯度乘以零，故前置新块首步任务梯度为零；末层权重通常可获梯度但具体非零须依赖实际输入和上游 loss 梯度。零输出后仍会因 weight decay 改变的参数不能被误报为 task 梯度打开。直接 Wi 路径在初始化即连接 W48 特征至类别logits，所以主干仍有监督路径；但 W_i 是随机读出，不是 ImageNet 分类头、也不是已训练分割语义。bias `log(.01/.99)`只是偏置常数，feature值的分布与跨尺度求和未知，不能据此保证初始概率约为1%。

D没有幅度约束会允许它改变 `U(S)`，包括压过或反转语义 logit；S 的独立监督和D幅值监控只能提供反馈，不能构成硬保护。设计者放弃两种硬约束的数学反例成立但范围有限：若某点 `U(S)=-10` 且 `|D|≤2`，最终logit仍至多为-8；若每个2×2块 `mean(D)=0` 且同块 `mean(U(S))<0`，则该块 `mean(Z)<0`，因此不可能四点全为正。这只限制重负点或整块全阳性，不意味着无法修正较小的阳性子集，也不证明无界 residual 训练安全。

FP32 临时内存边界：B=2 时 48×1024² FP32特征单张约384 MiB、32×512²约64 MiB、stem64×1024²约512 MiB、4×2048²最终logits约128 MiB。它们是单tensor字节量，不是总激活/反传峰值；layernorm布局转换、保留的autograd中间值、checkpointing和卷积 workspace 会影响真实峰值。静态参数少不能代替实际显存 preflight。本轮不能运行所规定的preflight。

整体判定：静态数学没有发现必须推翻v3拓扑的矛盾；可确认的只是路径、张量映射、初始化恒等式与参数公式。随机 direct head 的初始输出尺度、零末层后支路是否开始学习、细节残差对SE语义的影响、Normalize 变化、2048显存/训练稳定和目标分数都仍未知。逐类/逐尺度 logit与梯度、TP/FP/FN、C/D幅值、clip前后范数、真实shape显存是合适诊断，但不能替代40k验证结果或因果消融。

## 额外自有源码族与决策脚本

全量候选清单中还包括旧版本的梯度探针、两种实验选择器和几套坐标/恢复辅助模块。它们均按内容SHA全文核读，并与对应报告/运行身份分开解释；这些源码不会被当成 NativePyramid v3 的实现或成功证据。

- `semantic_execution/loss_probe.py`仅将已训练模型的 logits 设为梯度叶节点，对Dice、balanced BCE、教师KL和活动集Lovasz逐类/按base TP/FN/FP/TN计算logit梯度绝对质量。模型处于eval；循环不调用optimizer或`step`。这能比较局部logit导数与区域分布，不能推出参数梯度、全局裁剪后方向、SGD因果或后续恢复。运行记录将该诊断标记为8个train batch、0次正式SGD、未访问val/test；源码本身未被执行。
- FinalDecision `execution/decision.py`规定C/S/D/SD、两个种子和single/fixed/half fusion选择次序；只有两个seed均优于参考且平均提升至少0.5pp才满足eligibility。其`qlib/decision.py`要求两端点齐备才触发严重失败，并要求双种子公共节点。`train_run.py`定义的是不同的32k多来源计划，D/SD前20k还加入TJDR，采用center768+halo128、EMA评估与注册节点。它不是旧v3 40k运行协议。
- FinalDecision `qlib/geometry.py`用768中心与128 halo构成1024输入窗，明确在归一化空间零填充并在整窗logits上裁中心；监督只落在真实中心区域。该实现解释那套tile数据几何，不能覆盖v3原图2048路径。
- Next 的 `common.py`、`decision.py`、`geometry.py`分别实现严格文件SHA/状态载入、配对停止选择、tile/全局特征坐标对齐。其几何代码区分native、square2048和long2048 canvas，并用`grid_sample(align_corners=False)`将global feature映到tile stride网格。该信息只限定Next自己的接口和坐标；不构成对v3插值路径的既往验证。
- NextPlan `execution/decision.py`为E_NEG/F4/F12登记各自终点、连续节点门和双seed候选规则；`historical/train_run.py`按Q_DDR/Q_MIX运行，依赖DINOv3预训练身份、冻结stream、microbatch×accumulation、梯度裁剪与检查点恢复。该计划/代码及其专属数据与预训练身份不转用于NativePyramid；其协议标为PROPOSED_NOT_STARTED，不能当成一项已完成的模型训练。

- FullCompetition Refiner源码单独修改冻结champion预测的MA/HE通道；EX/SE由原预测硬保留。它由固定参考概率、候选生成/裁块训练、tiled inference组成，训练目标是30倍Focal BCE加soft Dice。GN参与pad到批内最大crop形状的特征计算，而mask主要限制loss/writeback，因此不能假设pad对GN统计无影响。源码虽已全文核读，报告度量没有同SHA身份绑定；它只说明候选实现，不证明执行归因或效果。
- FullCompetition的ConvNeXt U-Net与Ready/Next数据栈是内容不同的版本。模型、增强、DDR/external loader和cache文件均读到EOF并按SHA去重；它们展示另一套骨干、局部crop与缓存/恢复机制，不是W48语义解码器的对照。repo_snapshot五个文件按SHA均能映射到Plus3/FusionAware/MA的已读源，未发现一个需要另立为独立历史版本的隐藏模型内容。
- DINO qlib的model/loss/data/evaluate/metrics/optimizer共7个独有SHA及NextPlan/Q训练身份被全文复核。NextPlan source identity绑定训练源码但其计划标为未启动；Q身份属于DINO路线。代码、预训练文件和性能都不转用为v3的结构证据，检查点未读取。

协议字段检查补充：5个不同`screen_gate.py` SHA中，adapter门控由adapter协议显式绑定；另一个共享SHA由aligned/Detail/foundation/IoU/semantic绑定；expert与residual各自绑定。SACD isolated gate未绑定任何11个正式/筛选协议，CET/SACD/v3身份也没有`screen_gate_code_SHA`。source_before_worker_RAM、failed-assertion和SACD isolated的4个新增模型/训练文件同样不在那43个protocol-named SHA内。它们虽已全文审阅，仍不能以相邻源码冒认运行身份。

上述来源的每个路径、SHA、行数、全文/差异/索引状态以及同SHA副本记录在`source_coverage.json`；正文读取与结构化AST扫描严格分开。最终未覆盖数量只以那份冻结的唯一SHA候选清单和代理回执合并结果为准。

## 最终源码覆盖结论

冻结源候选清单中的额外29个自有direct-core/support内容SHA现已逐份全文核读；Next的4个独有`nextlib` SHA同样全文核读，adapter/loss/engine三个Next路径通过与Ready版本SHA一致继承全文状态。历史代理独立清单复查没有新增未读核心。故本次明确界定的自有模型、损失、训练、评估、推理及直接导入核心范围未读数为0；可逐SHA复核的人员、路径、文件大小/行数和同内容副本均在[source_coverage.json](D:/dataset/experiment/HRDecoder_NativePyramid_20261011/review_20261016/source_coverage.json)。

这不是“全树所有代码均人工阅读”：库存仍有1,146个普通源码SHA；全树2,370个独立Python内容做了AST语法审查，其中2,368通过，2个失败都在第三方vendor镜像；归档独有52个SHA中，23全文、10版本差异核读、19仅AST/索引。第三方库、通用运维/部署工具与非架构辅助脚本保持这些较窄状态。语法解析和文件索引都没有被冒充为语义全文审查。
