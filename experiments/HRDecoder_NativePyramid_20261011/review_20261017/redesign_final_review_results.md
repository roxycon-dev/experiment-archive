# DualScale 架构稿终审记录

审查对象：`05_HRDecoder_DualScale_架构定稿_20261017.md`。只读核验数字、模型结构、参数共享、训练预算与证据链接；没有实现或运行 DualScale，也没有把设计审查当作性能验证。

## 已核通过

- **目标及差距正确。** 本轮 Z@40k val mIoU 为 39.425551%，锁定融合目标 val 为 45.321831%，差 5.896280 pp。逐类 IoU 目标差为 EX 3.749077、HE 4.096487、SE 10.202743、MA 5.536814 pp；TP/FP 差值与结果表相符。按 `IoU=TP/(T+FP)` 推出的四类近似错误计数预算约为 15,301.45、69,205.19、25,905.48、6,067.57。SE 是最大的当前逐类缺口。
- **对照边界清楚。** 融合目标是异构系统，不是单一 HRDecoder；S 是本轮同一训练中的辅助头，未被写成独立训练基线。val 与历史 test 数值也未混用。B0 的 39.390%@32k 是历史报告级 best-val，不足以证明新设计已胜出。
- **前向/头结构与原代码吻合。** 1024 G、2048 N、同一 W48、完整图 logits 对齐后取固定 0.5 均值，以及 G 上 M4 辅助路径，与已核原实现的意图相符。完整 FCNHead 细节已补齐：720 通道输入、`in_index=0`、64通道、压缩 1×1 ConvModule、7×7 ConvModule（dilation 1、padding 3）、BN/ReLU、无 dropout、无 `conv_cat`/context、末端 1×1 分类、`align_corners=False`。配置证据见原仓库 `configs/_base_/models/hrdecoder_fcn_hr18.py:45–59`、`configs/_base_/models/hrdecoder_fcn_hr48.py:12–14`、`configs/lesion/hrdecoder_fcn_hr48_ddr_2048.py:24–27`，实现见 `mmseg/models/decode_heads/fcn_head.py:43–65, 93–125` 与 `decode_head.py:89–93`。
- **共享与统计银行定义自洽。** 可训练卷积、分类器和 BN affine 只有一份；backbone 按 G/N 隔离运行统计，head 按 G/N/M4 隔离运行统计，并纳入 `running_mean`、`running_var`、`num_batches_tracked`。稿件也说明了 bank 化不解决共享权重的梯度折衷，且目前不是已实现状态。
- **更新/视图计数正确。** 40k 更新 × 每更新 2 张原图 = 80k 原图出现次数；每次各有一份 1024 与 2048 编码，共 160k encoder 视图。`1024²+2048²` 是 B0 单个 1024 视图输入像素面积的 5 倍、v3 单个 2048 视图的 1.25 倍。稿件把它限定为面积估计，没有冒称精确 FLOPs、显存或时长。
- **未暗示新成绩。** 稿件明示当前 Python 仍为 v3、DualScale 尚未实现；40k 和目标数值仍是计划/门槛，没有声称性能通过。

## 已复核的收口修订

本次复查时，定稿已绑定当前 data identity `7c53b86bcb27ccd121042f667cc58e39b98716a8c477bfd1f1657fa126575719` 和 454/76/227 划分，并明确历史 B0 的标签字节身份未证明相同。无裁剪规格也已与日志定义对齐：`clipping_enabled=false, clip_coefficient=1`；历史 test 已注明多阶段复用、不能再称为独立盲测。

第 9 节列出的 04 复盘、目标证据、原实现核对和数学证据均已落盘，`redesign_reference_implementation.md` 当前存在且写明源码快照、核心调用路径及其与新设计的边界。`redesign_target_evidence.md` 已存在，无需重写。

## 结论

数字、结构、共享参数、预算口径和证据链接均通过静态复核。这不构成训练可行性或精度通过。2026-10-17追加更正：仍有两类独立缺口。第一，没有证据证明双尺度共享W48能补足到融合目标的性能差距；第二，实际实现/资源尚未验收，包括共享参数对象唯一性、BN bank 隔离与重算计数、BF16 数值检查、实际 batch=2 及双尺度 2048 输入的峰值显存和运行时间。实现验收通过不等于性能目标成立。最新结论以[主代理独立审查](D:/dataset/experiment/HRDecoder_NativePyramid_20261011/06_主代理独立审查_20261017.md)为准。

复核依据：`review_20261017/results_findings.md`、`redesign_target_evidence.md`、`redesign_math_evidence.md`、`implementation_findings.md`，以及 `autodl100/work/HRDecoder` 中列明的原 HRDecoder 配置与实现文件。未加载权重、未执行训练/前向或新 val/test 像素评估。
