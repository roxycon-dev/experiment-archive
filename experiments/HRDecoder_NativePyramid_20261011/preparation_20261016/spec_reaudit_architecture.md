# v3 逐条重新核对：网络、初始化、归一化与损失

2026-10-16。本轮重新完整读取 `02_v3_最终架构设计_20261016.md`、当前 `model.py`、当前 `losses.py`，逐条对照设计第 3、4、5 节，不用先前测试结论代替本次审查。不修改源码、不执行 SGD、不读取训练样本；仅对一个 `[1,8,4,5]` BF16 合成张量做 CPU LayerNorm 内存布局 profiler。

审查快照：

|文件|SHA256|
|---|---|
|model.py|9924206c00a818461b52693802b6e72e19c8b7e3d292bd3d275838686a2f081c|
|losses.py|7fbca98e62f6db79627217acab773065f157bfa36f41558278ab07038a657482|
|02_v3_最终架构设计_20261016.md|9247fd2d2a6b54fcc3467cd92a7c726d606c62da9739b50b704e9f7237e7865b|

## 结论的准确范围

在上述修复后快照中，没有发现网络拓扑、通道、分类偏置数量、C/D 初始化或损失数学定义偏离第 3—5 节的情形。不能把这个结论说成“每个浮点运算、每一位结果、每个未规定的实现细节都一模一样”；也不能由这两个文件证明数据、增强、SGD 预算、恢复状态机或服务器显存已验收。

本轮确实找出两项此前遗漏，已由主代理修复后重新读取源码核对：一项是 direct bias 的字面加法顺序；另一项是 LayerNorm 中可避免的整张 FP32 特征副本。

## 本轮发现及修复后状态

|项目|修复前事实与性质|修复后状态|
|---|---|---|
|直接语义 bias 加法顺序|旧实现从 bias 开始逐支相加；设计表格要求“四支 logits 插值相加，最后只加一份 bias”。bias 数量没有错，精确算术数学等价，但 FP32 舍入顺序可能不同，是字面执行顺序差异|`model.py:259–270` 现在先按 F1→F4 顺序累加四支，再加一次四类 bias；字面顺序已对齐|
|LayerNorm 额外 FP32 副本|旧实现对 NCHW 连续 BF16 输入先 `movedim(...).float()`；该转换保留非连续 NHWC stride，`F.layer_norm` 又做 FP32 contiguous clone。移动维度本身是 view，不能推出整个操作不复制。该多余副本与“避免无用整张特征副本”的要求不符，但不改变统计轴或数学函数|`model.py:160–163` 现在用单个 `.to(dtype=float32,memory_format=contiguous_format)` 合并精度转换与 NHWC 排布，再执行 LayerNorm，已消除这里的重复 FP32 clone|

tiny CPU profiler 实测：旧路径对 160 元素产生一次 640 字节的 FP32 `_to_copy`，随后 LayerNorm 内出现另一次 640 字节的 contiguous clone；`aten::contiguous` 和 `aten::clone` 是这同一份嵌套复制，不能重复计数。新路径只出现前一次 640 字节 `_to_copy`，LayerNorm 内没有 `contiguous/clone`。这证明该具体副本被消除；LayerNorm 的 FP32 输入、FP32 输出、均值/方差以及返回 BF16 的分配仍存在。未据此推断 CUDA 2048 峰值已经可行。

另只读检查了新加的 `test_shared_bias_is_added_after_scale_cancellation`：它把前两支设为常量 `1e8` 与 `-1e8`，其余两支零，bias设1，并独立要求S严格等于1；没有照抄旧实现作为expected。旧的bias先加法会因舍入丢掉1，而修复后的先求和方法保留1。这是针对字面顺序的有效回归。本代理没有再次运行全部测试；主代理报告的65项通过不冒充本代理重跑结果。

## 第 3 节：网络构成逐条对照

|设计要求|当前实现核对|判断|
|---|---|---|
|一个直接接收完整画布的 HRNet-W48|`NativePyramid.forward` 原样将 image 交给唯一 `W48Features`，未在主干前减半、未建第二个主干|一致；正式画布 2048 由训练入口保证，通用模型也接受 32 的倍数供测试|
|stem 64@1024；F1–F4 为 48@512、96@256、192@128、384@64|手动执行 timm 的 conv1 stem、conv2、layer1、transition1/stage2、transition2/stage3、transition3/stage4；读取对应 stem 与四分支。2048 尺寸遵循各层 stride|代码结构一致；本轮未重跑 2048|
|四个无 bias 的 1×1 分类器，48/96/192/384→4，不加 Norm/激活或前置新 Norm|`direct_classifiers` 各为 `Conv2d(...,4,1,bias=False)`，直接接原特征|一致|
|各支先抵达 F1/512 网格后相加，最后加唯一四类 bias|每支 classifier 后 resize 到 `f1.shape[-2:]`，四支和后只加 `direct_bias`|修复后字面顺序一致|
|F4 384→128+一次细化|`deep = ConvLN(384,128), Refine(128)`|一致|
|F3 192→128，拼接 256→128+细化|`laterals[0]`、`fusions[0]`，按 coarse/fine 拼接|一致|
|F2 96→96，拼接 224→96+细化|`laterals[1]`、`fusions[1]`|一致|
|F1 48→64，拼接 160→64+细化，得到 G|`laterals[2]`、`fusions[2]`，迭代顺序 F3/F2/F1|一致|
|细化块为 DW3×3 与 DW7×7，各 LN+SiLU；相加后 1×1+LN；与输入相加再 SiLU|`Refine.local/context` 分别 groups=C，padding=1/3；`mix` 无 bias 1×1+LN；最后 `silu(x+mix(...))`|一致|
|所有特征投影为无 bias Conv、通道 LN、SiLU|统一 `ConvLN`|一致|
|C 为 G 的 64→4 无 bias 纯线性卷积，替代旧 aux|只有 `auxiliary = Conv2d(64,4,1,bias=False)`；返回的 auxiliary_logits 是 `direct+C`，没有另建辅助头|一致|
|G 到 1024，与 stem 64→32 拼成 96，96→48 后两次细化；不再有裸 RGB PixelUnshuffle|`detail = ConvLN(96,48), Refine(48), Refine(48)`；拼接 G 与 `stem(shallow)`；没有 rgb 模块或 PixelUnshuffle|一致|
|D 为 48→16 有 bias 纯线性卷积，PixelShuffle2 得到四类完整网格|`readout` 与 `pixel_shuffle(...,2)`；无末端 Norm/激活|一致|
|S 先统一在 512，Z=U(S)+D；不能合并两次离散插值|`semantic=direct+C` 先在 F1 网格，之后单独 resize 至 image 网格再加 D|一致|
|所有双线性 align_corners=False|统一 resize helper；主干 31 个 nn.Upsample 替换为该 helper；损失辅助插值也显式 False。当前 timm hrnet.py 的融合插值仅这一处 nn.Upsample 构建来源|一致|
|不引入 dropout、总门控、概率裁剪；Z 直接进入 logit 损失|新头无这些模块，无 sigmoid；sigmoid 只在 Dice/推理所需位置使用|一致|
|各路径端到端，不冻结或 detach 主干|forward 训练路径没有 detach；仅诊断统计使用 detach/no_grad|一致|

## 第 4 节：初始化、归一化与梯度

|设计要求|当前实现核对|判断|
|---|---|---|
|四个直接卷积非零 N(0,0.01²)|`nn.init.normal_(layer.weight,std=.01)`，默认 mean=0|一致；“非零初始化”指层不是全零，不保证正态抽样每个元素都非零|
|共享 bias=log(.01/.99)|四元素 `direct_bias` 直接设此值|一致；并不声称最终初始概率处处 1%|
|C 无 bias、weight 全零；D weight/bias 全零|三个 `nn.init.zeros_`；C 构造 bias=False|一致|
|初始 C=D=0，Z=U(S_direct)|纯线性零末层，后面只有 PixelShuffle/加法，无其他非零偏置|结构上满足；不据此宣称分支后续始终受控|
|其余新卷积统一初始化；LN γ=1、β=0|其余 Conv2d 保留统一 PyTorch 默认 reset_parameters；LN 显式 ones/zeros|未发现冲突；卷积分布的完整参数是设计未写明的细节，见下节|
|首反传末层可得梯度、内部任务梯度为零，直接通路仍达主干|没有 detach；零 C/D 令通过它们到新内部层的任务梯度为零；直接 W_i 路径独立存在|一致；梯度具体非零仍取决于数据和上游梯度|
|每个 N,Y,X 独立沿 C 计算总体方差，FP32，eps=1e-5，γ/β 各 C|转换到连续 NHWC 后 `F.layer_norm(...,(C,),weight.float(),bias.float(),eps)`，autocast disabled；返回输入 dtype|一致；没有跨 batch/空间统计|
|固定主干预训练 BN running statistics，affine 可训练|`W48Features.train` 对所有 BN 执行 eval，不修改 affine.requires_grad|一致；单纯阅读不能代替服务器多步 buffers 身份检查|
|避免无用整图副本；真实 2048 记录峰值|已修复 LN 重复 FP32 clone；没有把网络 feature 保存到诊断结果|源码改进已核对，CUDA 峰值仍未验收|
|decoder246132、全模型65571252|构造末尾分别按实际 p.numel() 求和并与两个目标常量比较；模块尺寸与计数要求相符|一致；本轮为源码审计，没有把历史实例化结果冒充新实测|

## 第 5 节：本文件能核对的监督契约

|设计要求|当前实现核对|判断|
|---|---|---|
|仅锁定 SHA 的 ImageNet W48；1830 键逐项身份，排除126，31处双线性|文件 SHA 严格检查；1830 个 own key 检查存在/shape/dtype，strict load 后逐项 torch.equal；126 个排除键仅允许分类前缀；融合替换数强制31|一致；本轮未重新加载大权重|
|0背景、1EX、2HE、3SE、4MA、255忽略，独立四通道|`target_masks` 按 label==1..4 构造，mask=label!=255，拒绝其他标签；要求 logits 与完整 B4HW target 同形状|一致|
|仅 Z 与同一 S 两个损失位置，权重1/.2；只插值 S，不下采样标签|forward 对 logits 做 main，对 auxiliary_logits 做完整标签尺寸双线性插值后 aux；`total=main+.2*aux`|一致|
|k_ic=min(N_ic,max(1024,3P_ic))；按 logit 降序，平分值按有效阴性展平索引升序|计数按每图每类计算；从 mask 取 v 保留原展平次序，后从 v[~y] 取阴性；topk只取阈值，阈值处按nonzero的升序索引取成员，最终稳定排序|一致，未依赖 topk 的平分值成员次序|
|完整 batch 的 P_c/K_c/A_c/V 分母|`batch_denominators` 先按每图统计、再跨图求和；one_head接受共享分母|一致；调用者必须先用完整 batch 计算再传给微批，这一调用行为由 train.py 另审|
|仅阳性图算平方 Dice|仅 positives.numel()>0 时计算；分子2*sum(p[y])+eps，分母sum(p²)+sum(y)+eps|数学定义一致|
|阳性softplus(-z)/P，难阴性softplus(z)/K，denseBCE/V；权重.25/.25/.05；四类平均|逐图逐类累加对应项并除共享分母与4，最后施加三个权重|数学定义一致；浮点结合顺序并非逐字公式，见下节|
|FP32损失；255不入求和，缺组分子0且不转移系数|z.float、FP32分母与损失操作；所有分子来自mask有效元素；空图用关联图的空切片sum构造零；不存在缺组转移|一致|
|全ignore单图贡献0；整个batch全ignore拒绝更新|共享有效分母时空图各项0；batch_denominators对有效总数0抛错|一致|
|S与Z分别选难负例|两次独立调用one_head；不复用选择索引|一致|
|seed42/head seed20261011|NativePyramid默认种子相符，用fork_rng包裹骨干及头初始化|一致；正式运行是否传入其他值需由训练入口另审|

第 5 节其余事项——实际融合标签与454/76/227、40k×batch2、SGD参数、warmup/衰减、梯度裁剪、BF16硬件拒绝降级、增强、采样、恢复与异常预算——主要由 data.py/train.py/protocol.json/verify.py/evaluate.py 执行。本报告不从 model.py/losses.py 的正确性推出它们全部一致，也不授予服务器训练 PASS。

## 应公开说明的实现细节与浮点边界

这些项目未发现改变规定数学模型或监督配方，但不能以“字节级完全相同”概括：

1. **其余卷积的初始化族。** 设计只写“沿用统一的卷积初始化”，未指定每个参数名的完整张量快照，也未在本文规定 fan mode。代码采用现有原型同样的 PyTorch Conv2d 默认 Kaiming-uniform(a=sqrt(5))，对本头所有无 bias 投影/深度卷积/混合卷积一致使用。其权重范围相当于 ±1/sqrt(fan_in)。相同 seed 不构成与 v2 公共模块逐位同初始值的证据；若以后做匹配消融，仍须按方案另行固定共同模块快照。
2. **混合精度边界。** BF16 特征的双线性插值先转 FP32 计算，再转换回输入 dtype；直接分类器的卷积在调用者 BF16 autocast 下计算，之后四通道 logits 转 FP32，direct 求和、C/D logits 及最终加和均为 FP32。方案锁定 BF16 forward、FP32统计/损失/主权重，但没有逐算子限定这些转换。不能承诺与原生 BF16 插值核逐位相同。
3. **损失加法/除法的结合顺序。** 代码每图先除共享分母与4后累加，再统一施加组权重；文字公式先跨图求和、组内归一及施权，再对四类平均。两者精确算术相同，FP32舍入可能不同。微批梯度累加本来也不应要求逐位相同，应按事先锁定容差验收。
4. **内存布局及重算。** NHWC 连续的临时布局、回到 NCHW 的 channels-last view、非重入 checkpoint、按 head 重算损失是实现选择。修复消除了一份可避免 clone，不意味着没有临时张量，也不替代真实2048 CUDA峰值实测。
5. **通用接口与正式入口。** 模型允许测试用32倍数尺寸；loss构造允许指定测试用 min_negatives、可选valid掩码和外部分母；默认正式参数与方案相同。正式训练必须通过锁定协议入口，不能把通用接口的可配置性当成已执行了不同正式方案。
6. **仅诊断的统计定义。** direct 分位数在各原生尺度、插值前精确计算，使用.5/.9/.99；C/D均值/RMS/max用全部元素，附带明确标记的有界stride采样分位数。第3—5节未规定这些统计分位或采样点，它们不进入预测和损失，也不保存完整feature。

因此，修复后可准确表述为：“当前源码的网络结构、初始化约束和损失数学定义已逐条对齐给定 v3；本轮找到的两处执行细节已修正。浮点结合顺序及未逐算子规定的实现细节不宣称逐位同一，服务器实测仍待完成。”
