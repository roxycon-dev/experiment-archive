# HRDecoder / 双尺度重设计参考实现证据

## 范围与运行证据

本备忘基于本地原始 HRDecoder 源码快照、已存 v3/d303 运行记录、历史审计台账以及 HRDecoder 与 HRDA 的论文/官方实现。没有运行模型、训练、前向/反向、像素测试、权重加载或新的 GPU 测试。HRDecoder 本地目录不是 Git 仓库，无法给源码快照附可信 commit；复核时应以文件内容及所列来源为准。

d303 的已授权远端只读回收证明 SGD 日志确实存在，远端为 /root/autodl-tmp/nativepyramid_sp_v3_20261016/run_seed42_launch02/train.jsonl，大小 3,473,259,370 bytes，SHA256 为 2b286aee99ed75d894c8d60a239c92018fed158bc98d0def7eed9e61fcf3a63f。远端读取前后 size 与 mtime 相同。日志有 40,000 行，update 恰为 1…40,000，重复、缺步或顺序异常为 0；这是对源日志的只读流式检查，不是把完整 3.47 GB 原文件下载回本地。

可复核材料在 review_20261017/recovered_logs/：

- train_metrics_all40000.jsonl：保留每步的顶层身份/学习率/SID 字段、主/辅损失、裁剪前总梯度范数、clip 系数、裁剪前后模块范数、4×4 direct_scale_class 矩阵、监督正像素数和正样本图数。40,000 行完整、连续。投影 JSONL SHA256 为 a72d77846ace86e429fd4120689ed814aa67668e6dd627c9274216835a5ac3a9。
- train_metrics_all40000.jsonl.gz：上述 JSONL 的 gzip 归档，31,749,245 bytes，SHA256 为 7e007ae350847ade6d46a8821731a71fd90d9de03d4492f18703aab9b45ec334。
- train_metrics_all40000.manifest.json：源路径、源大小与 SHA、首末 size/mtime、投影及 gzip 大小和 SHA、连续覆盖与投影字段清单。梯度对象有意省略庞大的 parameter_modules 逐参数映射，其余梯度汇总字段保留；manifest 记录了源字段和投影字段差别。
- train_metrics_boundaries_full.json：首尾各一条未经字段投影的完整原始 JSON 行，供 schema 核验。

原文件未完整回收到本地。早先中断的 SFTP 前缀保留为 recovered_logs/run_seed42_launch02_train.jsonl.partial，大小 177,209,344 bytes；它只是前缀，不能作为完整日志使用。日志投影由本地脚本 recover_metrics_projection.py 说明字段筛选。完整 raw 文件 SHA 由远端只读流式计算，投影 SHA 是本地文件 SHA，两者不可混淆。

## 原 HRDecoder 实际实现

本地源快照位于 autodl100/work/HRDecoder。关键文件是 mmseg/models/segmentors/HRDecoder.py、decode_heads/fcn_head.py、decode_heads/decode_head.py、losses/binary_loss.py 和 configs/lesion/hrdecoder_fcn_hr48_ddr_2048.py。可逐项对照 HRDecoder.py 第 63–86、137–233、235–276 行；FCNHead 第 46–65 行；DecodeHead 第 84–90、89 行附近。公开来源为 [HRDecoder 官方仓库](https://github.com/CVIU-CSU/HRDecoder) 及 [MICCAI 2024 原论文](https://papers.miccai.org/miccai-2024/paper/4020_paper.pdf)。本地 B0 的展开配置证据另存于 HRDecoder_FinalDecision_20261005/结题报告/evidence/original_HRDecoder_baseline_evidence.json。

B0 的输入是 batch=2、2048×2048 训练画布。模型先把 RGB 双线性缩到 1024×1024，再对这张低分辨率整图运行一次 HRNet-W48。四个 stage 输出为 F1 48×256×256、F2 96×128×128、F3 192×64×64、F4 384×32×32。将 F2–F4 双线性插值到 F1 网格并拼接，解码头输入为 720×256×256；FCNHead 用 1×1 压缩卷积 720→64、BN/ReLU，再以 7×7 卷积 64→64、BN/ReLU，分类 1×1 卷积输出4类。训练配置显式 dropout_ratio=-1，因此无 dropout。全局 logits 从 256 网格插值回 2048×2048。

所谓 HR 分支不再编码 RGB crop。它从同一张全图提取出的 720×256×256 融合特征上抽取随机 crop。基准 crop 边长为 1024，scale_ratio 随机在 0.75–1.25，因此 label crop 边长约 768–1280 且可整除8。按原图/特征的网格比例裁特征，将其双线性缩放回 256×256，再送入同一 FCNHead；GT 则直接取原 2048 标签上的相应 crop。M=4 个 crop 沿 batch 轴拼接，一次 decoder call 计算其局部 Dice loss，乘 0.1。论文也明确采用低分辨率特征 crop，以避免为高分辨率 crop 增加 encoder pass。

主监督不是单独的全局低分辨率结果。训练会把 1024×1024 全图的 global logits 还原到完整 2048 画布；同一 720×256 特征图上再按固定 1024×1024 窗口和同尺寸 stride 滑动解码局部 logits。2048 画布因此是 2×2 共4个窗口、无重叠。每块 logits 上采样回窗口大小，按覆盖次数归一化后铺回画布，再与全局 logits 做固定 0.5/0.5 平均。主 Dice 在完整融合 logits 与完整标签上计算，M4 局部 Dice 另加 0.1 权重。推理走同样的单次低分辨率 W48、full-map global logits、特征滑窗解码和固定均值融合。实现与论文中将低/高分辨率分支融合的固定均匀策略一致，没有可学习 attention head。

W48 与 head 的 BN 使用普通 BN、requires_grad=true、norm_eval=false；B0 只有单一组 running statistics。W48 backbone 从 ImageNet 预训练初始化。实际 B0 展开配置为 seed42、SGD lr=0.005、momentum=0.9、weight decay=0.0005、poly power=1、linear warmup 1500 steps、warmup ratio=1e-6、40,000 updates、无梯度裁剪、batch=2；类别顺序为 background/EX/HE/SE/MA。公开配置中的 seed14 不是该 B0 历史运行的 seed。该事实不能直接说明新架构表现会优于或逊于 d303。

## 与 05 双尺度草案的实际差异

05 草案将保留低分辨率全图路径 G：2048 RGB 缩至 1024、共享 W48 编码及 G 组 BN、256 网格 head；同时对原始 2048 RGB 运行第二次共享 W48 编码路径 N，输出 512 网格 logits。两路 logits 还原至 2048 后逐点均值，主 Dice 监督融合结果；M4 仍来自 G 路径的特征 crop、同一 head、独立 aux BN，避免再增加 RGB encoder pass。这不是原 B0：B0 没有 N 路径，也没有 N 组 BN。

按一次训练图像计，G 与 N 分别处理 1024² 与 2048² RGB，按像素规模为 B0 单次 1024² encoder 的 1+4=5 倍；相对 d303 单次 2048² 主干输入是 1.25 倍。40k updates、global batch2 对应 80k 个图像样本，每样本两种尺度，共 160k 次 scale-view 编码；这是新增计算成本。M4 继续复用 G 特征，不增加四次 RGB 编码。若把局部监督改成对 RGB 全覆盖裁块，每个样本的 2048² 画布按 1024² 无重叠需 4 次局部编码；与 G 全图路径合计 5 个 encoder views/样本（此处还未计是否保留独立 N 视图），在当前 05 的双路径基础上进一步增加4个局部 view。只训练一个随机 RGB crop 又不能等价于对完整高分辨率区域的全覆盖监督。

既有 HRDecoder/H4 台账中没有找到“W48 同时编码低分辨率全图和原图高分辨率 crop、共享同一组卷积权重”的历史实现或公平成功配方。原 HRDecoder 用低分辨率特征 crop；H4 有输入 RGB crop 作为冻结 D/B 集成后的校正器，但不是同一共享 W48 的双尺度主干；旧 HRDecoder 学生与 HRPlus 局部分支也不是 05 这一训练/推理图。因此应把 05 作为新假设，而非恢复了原算法。HRDA 官方实现与此处的权重共享概念相关：它在同一 backbone/head 上分别编码低分辨率 context 和原图高分辨率 detail crop；标准训练每样本一个高分辨率随机 crop，推理/伪标签可用重叠滑窗覆盖多个 crop，并有学习到的注意力融合。其证据见 [HRDA 官方仓库](https://github.com/lhoyer/HRDA)、[ECCV 2022 论文](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136900370.pdf) 与 hrda_encoder_decoder.py 中 extract_unscaled_feat/extract_slide_feat 的多尺度调用。HRDA 是不同任务与数据域，不能据此推断 DDR 病灶分割收益或显存可行性。

## 训练可行性的已知边界

已存的历史资源核查只覆盖一条 W48 路径的 2048 输入、batch2、BF16 forward/backward：RTX 4090、25,282,281,472 bytes 总显存，峰值 allocated 19,890,649,600 bytes、reserved 21,839,740,928 bytes；未创建 optimizer，SGD momentum 另估 287,821,328 bytes。它不是05双尺度图、双 BN bank、两个同时保留的反向图的实测。保留 G/N 两路反向图与 512 网格 7×7 head 可能显著提高峰值内存，现有证据不足以判定一定能放入单卡或一定 OOM。

本轮 d303 运行记录了 RTX 4090 型号、micro_batch=1 和逐图卷积执行方式；只读系统查询列出8张 4090，但 nvidia-smi 查询未返回显存总量/使用量，无法证明这8张卡对该容器或新实验都可用。故当前任务可用卡数、每卡空闲显存、双尺度峰值显存和两图 BN 的实际分布式执行能力均未知。不能将系统列出的8张卡当作实验预算，也不能将历史单路 forward/backward 检查当作双路可行证明。

目前可证伪的实现风险是：双路计算预算被低估；将 micro_batch=1 梯度累积误当作 batch2 BN；为保存 BN bank 而复制整个模块导致共享权重不再共享；checkpoint recompute 重复推进 running statistics；或遗漏 B0 的 dropout_ratio=-1。它们分别能由实际 encoder-call 计数、BN batch/统计计数、参数对象唯一性、重算统计对照和展开配置 diff 验证。此处不判断架构收益。
