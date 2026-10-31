# H4、HRDecoder 与互补分支代码审计

审计日期：2026-10-08。仅读源码、配置和已有审计 JSON；没有启动训练，也没有改动源代码。

下文简称：repo = D:\dataset\experiment\hrdecoder_plus3\repo；audit = D:\dataset\experiment\HRDecoder_审查轮_20260916；snapshot = D:\dataset\experiment\HRDecoder_审查轮_20260916\repo_snapshot。冒号后的 Lx-Ly 是对应文件行号。

## 结论

H4 是一个对既有模型输出做 MA 后处理的校准器：训练时冻结 D@2000 学生和其 B 参考模型，只用同一份训练图像上的 MA 标签训练一个 3,361 参数的小头；它依赖 D/B 两路概率及它们的固定 50/50 概率融合。它不是可直接插入 40,000 步新模型的通用分支。若把 D/B 替换为共同更新的内部动态分支，H4 的输入分布、融合概率语义和冻结前提都会变化。

## H4 代码事实

- 配置链为 hrplus_patch_dice.py → hr100_a.py → e4_detail_split613.py → baseline_split613.py。顶层配置覆盖数据根目录为 ./data/DDR_split_613，数据目录使用 images/{train,val,test} 与对应 labels/{train,val,test}；基线注释记载该划分为 train 454、val 76、test 227。证据：repo\configs\lesion\exp\hrplus_patch_dice.py:L1-L55、repo\configs\lesion\exp\hr100_a.py:L1-L19、repo\configs\lesion\exp\baseline_split613.py:L9-L50。
- H4 用 cfg.data.train 和 cfg.data.val 建集，逐步从训练集随机抽一个 ds_tr 样本，因此脚本训练 batch 实际为 1；配置中的 samples_per_gpu=2 不控制这个手写循环。训练管线先把图像拉伸到 2048×2048，旋转 ±45°，以 50% 概率均匀裁块、否则锚定当前存在的病灶类，裁 1024×1024，再随机翻转并 Normalize(mean=[81.205,50.636,21.216], std=[76.252,48.798,21.625], to_rgb=True)。H4 头接收的是归一化后的 RGB 张量；代码没有反归一化。证据：repo\configs\lesion\exp\hrplus_patch_dice.py:L7-L24、repo\hrplus_patch.py:L154-L201、audit\tools\h4_post_fusion_corrector.py:L104-L110,L170-L198。
- 验证/测试走无随机裁剪的 2048×2048 resize 流程，flip=False；H4 把模型输出恢复到 ori_shape，并把同一张已归一化图像插值到 ori_shape 后作为头的 RGB 输入。因此训练头看到 1024 crop 坐标，验证/测试头看到整图原尺寸坐标。证据：repo\configs\_base_\datasets\hr_ddr_2048.py:L39-L53、audit\tools\h4_post_fusion_corrector.py:L125-L149、audit\tools\h4_test_confirm.py:L110-L131。
- H4 的特征严格为 7 通道：[归一化 RGB(3), logit(P_fused_MA), logit(P_student_MA), logit(P_reference_MA), P_student_MA-P_reference_MA]。其中 P_fused=0.5*P_student+0.5*P_reference 是概率空间平均；头输出 delta=6*tanh(head)，最终 P_new=sigmoid(logit(clamp(P_fused))+delta)，不是对原始网络 logits 求平均。logit 概率先 clamp 到 [1e-4,1-1e-4]。证据：audit\tools\h4_post_fusion_corrector.py:L56-L79,L141-L152,L184-L202、audit\tools\h4_test_confirm.py:L122-L131、repo\hrplus_core.py:L67-L85。
- MACorrectorHead 为 Conv3×3(7→16)-ReLU-Conv3×3(16→16)-ReLU-Conv1×1(16→1)；最后一层权重与 bias 均零初始化。总参数量按卷积维度计算为 3,361（与最终报告一致）。训练目标仅 MA：0.5*BCEWithLogits + 0.5*soft Dice；优化器 Adam、初始 lr=1e-3、每步线性衰减、梯度裁剪 5。结果 JSON 记录实际 2,000 步、每 400 步验证，val MA IoU 最佳在 step 1,600；脚本默认值则是 4,000 步/每 500 步，需以 JSON 记录的实际运行值为准。证据：audit\tools\h4_post_fusion_corrector.py:L67-L79,L86-L93,L118-L123,L198-L225、audit\audit_runs\audit_20260916_A\experiments\h4_post_fusion_corrector.json:L2-L5,L18-L71、audit\audit_runs\audit_20260916_A\FINAL_本轮最终报告.md:L32-L45。
- H4 的 gt==255 被改成背景 0，再进入 BCE 和 Dice；这使 ignore 像素也作为负例参与训练。验证 IoU 则把 255 从 TP/FP 统计中排除。若新训练管线保留 255，复用 H4 目标时需明确是否要保持该差异。证据：audit\tools\h4_post_fusion_corrector.py:L139-L140,L182-L202、audit\tools\h4_test_confirm.py:L133-L144。
- D/B 模型在 H4 脚本中 .eval() 且所有参数 requires_grad=False；训练循环把 student logits 和 reference inference 放进 no_grad，优化器只接收头参数。BN 统计不会随头训练更新。证据：audit\tools\h4_post_fusion_corrector.py:L112-L120,L184-L210、repo\hrplus_patch.py:L43-L48,L75-L99,L139-L151。
- 零步delta=0时，旧头返回sigmoid(logit(clamp(P_fused)))，即截断后的融合概率；只在未触发clamp的区域与原融合数值等价，不能对极端概率称严格恒等。根据计算图，最后1×1层零权重会使第一次反传时前两层的梯度为零；输出层仍可得到梯度并先更新，因此这是首步梯度延迟，不是永久死梯度。E4的零初始化方式不同：detail_scale=0时detail分支第一步从融合损失收不到梯度，但标量detail_scale本身仍可学习。证据：audit\tools\h4_post_fusion_corrector.py:L56-L60,L74-L78,L193-L210、repo\mmseg\models\segmentors\HRDecoder.py:L43-L61,L228-L230,L261-L269。此梯度结论是由代码计算图推出的推断。主报告的新H4使用稳定混合logit作为输出基准，避免这项截断差异。

## 已有 H4 结果及权重可复用范围

h4_post_fusion_corrector.json 记录 val MA IoU：student 0.22784、fused 0.22617、corrected 最佳 0.25221（step 1600）。h4_test_confirmation.json 对 D@2000 的 227 张 test 记录 fused mIoU 0.425764、MA IoU 0.24729；corrected 为 0.434287、MA IoU 0.281383；非 MA 三类判定逐位不变。证据：audit\audit_runs\audit_20260916_A\experiments\h4_post_fusion_corrector.json:L6-L10,L69-L75、audit\audit_runs\audit_20260916_A\experiments\h4_test_confirmation.json:L3-L13,L38-L75。

工作区有两份相同 SHA-256 的 H4_best.pth（15,447 bytes）。只读加载确认文件只含 head 与 step=1600；输入卷积权重形状是 (16,7,3,3)，不含 D/B 网络权重。故它可在保持相同 7 通道顺序、类别、归一化和旧 D/B 概率接口时作为 H4 头载入；新单模型若没有这两路专家概率，输入通道就不等价，不能把该权重当作新模型的分支权重直接复用。可复核副本：D:\dataset\experiment\HRDecoder_Ready_20261002\best_pipeline\weights\H4_best.pth 和 D:\dataset\experiment\HRDecoder_Next_20261003\assets\H4_best.pth。R0 元数据也明确记录旧 H4 用 D@2000 的 student/reference 特征训练、之后才接到 D/E/B/E4 集成输出上；那是特定组合的迁移证据，不能证明对新 40k 模型也成立。证据：audit\audit_runs\audit_20260916_A\diagnostics\R0_counts_meta.json:L10-L44,L56-L70、audit\audit_runs\audit_20260916_A\STATE.json:L5-L25。

## 原 HRDecoder / W48 起点事实

- W48 配置继承 W18 基础模型，并把 backbone 通道改为 (48,96,192,384)；预训练声明为 open-mmlab://msra/hrnetv2_w48。解码器拼接四个尺度，输入通道共 720，4 个前景 sigmoid 通道。该 URI 是配置引用，不是仓库内固定的本地预训练文件路径。证据：repo\configs\_base_\models\hrdecoder_fcn_hr48.py:L1-L14、repo\configs\_base_\models\hrdecoder_fcn_hr18.py:L3-L63、repo\mmseg\models\segmentors\lesion_encoder_decoder.py:L38-L64。
- baseline_split613.py 设 hr_scale=(1024,1024)、scale_ratio=(0.75,1.25)、divisible=8、crop_num=4、lr_loss_weight=0、hr_loss_weight=0.1；数据注释为 454/76/227；训练 batch=2，seed=42，IterBasedRunner 40,000 步，SGD lr=.005、momentum=.9、weight_decay=.0005，继承 linear poly/warmup（base 1500 warmup steps）。证据：repo\configs\lesion\exp\baseline_split613.py:L9-L57、repo\configs\_base_\schedules\poly10warm.py:L7-L15。
- HRDecoder 把输入图缩到 1024 后跑 W48；多个 backbone 尺度上采样拼成 720 通道。所谓 HR crop 路径不是第二次对原图跑高分辨率 backbone：它从同一 lr_feat 按坐标裁特征并缩放，再调用同一个 decode_head。每张训练图随机取 4 个 crop（scale 为基准 1024 的 0.75–1.25，按 8 对齐）；HR crop loss 权重 0.1，融合全图 loss 正常计入，LR 单独 loss 为 0。LR 与 HR 在 logits 域简单平均。证据：repo\mmseg\models\segmentors\HRDecoder.py:L130-L186,L216-L276。
- baseline 数据输入 2048×2048；训练采用 0.5–2.0 Resize ratio、随机裁剪 2048、翻转、±45°旋转；灰度外像素在基础数据配置 pad 为 0。基线配置将 train/val/test 指向 DDR_split_613 对应目录。证据：repo\configs\_base_\datasets\hr_ddr_2048.py:L10-L53,L55-L81、repo\configs\lesion\exp\baseline_split613.py:L35-L50。

## E4、MA residual 与 local student

- E4 在原始 2048 输入上增加 stride-4 DetailBranch（浅层 3→32→32→32 卷积）和 4 通道 DetailHead；其 logits 插值回原尺寸后乘可训练标量 detail_scale 加到 LR/HR 融合 logits 上。E4 配置未设 detail_loss_weight，代码默认 0，因此 detail 仅受融合损失间接监督。证据：repo\configs\lesion\exp\e4_detail_split613.py:L1-L31、repo\mmseg\models\necks\detail_branch.py:L14-L68、repo\mmseg\models\segmentors\HRDecoder.py:L41-L61,L85-L87,L261-L274。
- 旧 HR100Refinement.SemanticMAResidual 用原图细节 + 720 通道语义特征预测 1 通道 MA logit 残差，零初始化输出层，并只把残差加到第 4（MA）通道；另加可选 hard-negative MA loss。hr100_a.py 是从 E4 checkpoint 起步、semantic_refinement=True、hard loss 权重 .02、冻结 BN 统计，但代码没有冻结 backbone 参数；该配置是 8,000 步微调。证据：repo\hr100_refinement.py:L13-L69、repo\configs\lesion\exp\hr100_a.py:L1-L19。
- 后续 SemanticMAResidualV2 额外输入基准 MA logit（经 1→8 特征映射），融合 detail、semantic、base 后零初始化预测 MA 残差。HRPlusMAResidual 明确冻结除 ma_refiner 外所有参数，只改 MA 通道；HRPlusMAOnBase 允许 student checkpoint=D、reference checkpoint=B 分开加载，但 B reference 仍是推理模型的一部分。快照配置 hrplus_mares.py 是 6,000 步头部训练配置（SGD .01）；R0 元数据记载历史最佳集成成员为 E@1500。证据：snapshot\ma_residual.py:L32-L65,L109-L163,L205-L263、snapshot\hrplus_mares.py:L1-L32、audit\audit_runs\audit_20260916_A\diagnostics\R0_counts_meta.json:L10-L27。
- HRPlus local student 的训练只对 1024 crop 的学生 encode_decode 输出计算 decode loss（可选 Lovasz）；冻结 reference 不参与训练。部署时学生则按 1024 tile、stride 768 重叠推理、平均 logits、sigmoid，再与整图 reference 的概率做 0.5/0.5 融合。H4 又在这次融合之后改 MA。也就是说旧训练监督、旧部署预测和 H4 输入处在不同阶段。证据：repo\hrplus_patch.py:L51-L61,L103-L151、audit\audit_runs\audit_20260916_A\diagnostics\train_inference_contract.md:L12-L29。

## 对新单模型方案的工程推断

1. 可保留 HRDecoder 的同一 W48 backbone、多尺度共享 decode head、LR/HR logit 融合；E4 原图细节与 MA 残差可作为单模型内部可训练分支。它们能融入单模型前向，不要求部署时保留额外的旧专家。
2. H4 的输入合同是固定的 D student、B reference 和概率空间 0.5 融合。共同更新的内部支路会改变两路分数及 P_fused 分布；若删除旧专家，H4 的 reference 通道直接消失。因而不应把 H4 后处理器作为新模型权重原样移植。可蒸馏的是旧最佳系统最终输出（含 H4 的 MA 输出）作为固定 teacher 目标；这是蒸馏监督，不是复用 H4 权重。
3. 新单模型应明确训练/推理的分辨率与融合点：H4 在训练时 head 输入是 1024 crop，测试时输入是整图原尺寸；HRPlus student 训练不见 reference，但推理会融合 reference 和 overlap-averaged 输出。将分支改成在线更新后，若想保留 H4 的后处理校准，应重新定义并训练其输入合同，否则其历史增益不能外推。
4. 加载历史 checkpoint 时按模块及形状做显式覆盖检查。H4 建模脚本对 D checkpoint 使用 strict=False；MA 残差代码提供了更稳妥的 student-only 严格加载方式，可忽略 reference.* 并校验其他缺失/多余键。证据：audit\tools\h4_post_fusion_corrector.py:L112-L120、snapshot\ma_residual.py:L87-L106。


## 对 40k 单模型草案的定点风险核查

- 原 H4 Dice 的精确实现是 D=1−2Σ(p·y)/(Σp²+Σy²+1)：平方分母，平滑常数 1 加在分母，分子未加平滑。它不是线性分母 Dice。原 HRDecoder BinaryLoss 的 binary_dice_loss 也默认 exponent=2，但 baseline 配置 smooth=1e-5，所以公式平滑项不同。证据：audit\tools\h4_post_fusion_corrector.py:L61-L64,L198-L202、repo\mmseg\models\losses\binary_loss.py:L46-L62、repo\configs\_base_\models\hrdecoder_fcn_hr18.py:L58-L60。
- H4 ignore 像素问题确认：gt==255 被替换为 0 后 y=(tgt==4)，再做全像素 BCE；因 y=0，ignore 全计为负例。Dice 公式中这些位置也进入 Σp² 分母，即使不进分子。相比之下，BinaryLoss 的 _make_one_hot 将标签 0 和 255 编到丢弃的额外通道，主分割 Dice 会忽略它们。对新 H4 分支，先定义 valid=(gt!=255)，然后 BCE 只在 valid 像素归一化；Dice 的交集、Σp² 和 Σy²都乘 valid。单纯把 target 置零仍不够。证据：audit\tools\h4_post_fusion_corrector.py:L198-L202、repo\mmseg\models\losses\binary_loss.py:L11-L37,L46-L62。
- 计划中的概率 mix 权重 .5625+.1875+.25=1，若 pA、pL、pC 均为 sigmoid 概率，则是凸组合且仍在 [0,1]。要保留 H4 输入语义，应先在概率域求 mix，再对 mix clamp 后取 logit；sigmoid(加权 logits) 与加权 sigmoid 概率不等价。H4 原实现 clamp=1e-4；若采用其他 epsilon，应在训练和部署保持一致。证据：audit\tools\h4_post_fusion_corrector.py:L56-L59,L141-L152、repo\hrplus_core.py:L67-L85。前述凸组合和不等价结论为直接数学推论。
- 新末层零初始化会让残差在起步时精确为 0；第一次反传的隐藏层梯度也为 0，但末层本身仍可学习。建议确认优化器确实包含该末层；避免再串联一个初始化为 0 的外部门控，否则分支输入梯度可能被两个零因子压住。原 E4 标量门控和 H4/MA residual 的零输出层是不同的零初始化位置。
- H4 输入 detach 能隔离 H4 loss 对主分支的梯度；但 pL/pC 仍会被 L_seg 与各自 KD 持续更新，所以 H4 看到的条件分布仍会变化。最后 1,600 个 global updates 与低 LR 可缓解漂移，不能据此声称等同于旧 H4 的冻结 D/B 校准。建议 H4 每次用与主损失相同的当前 batch、同一变换和同一输出网格，并核对末段 p_mix、pL、pC 的分布漂移。
- 训练输出网格与部署输出网格应一致。旧 H4 训练头使用 1024 crop，测试用整图 ori_shape；新 H4 若跟随 40k HRDecoder 的 2048 训练画布，RGB、logit(mix)、logit(L/C)、L−C 及标签都应在相同 2048 网格计算，再按同一部署 resize 规则恢复。不要把旧 H4 的 crop/ori 坐标约定无意带进新模型。证据：audit\tools\h4_post_fusion_corrector.py:L125-L149,L175-L202、repo\configs\lesion\exp\baseline_split613.py:L35-L50。

