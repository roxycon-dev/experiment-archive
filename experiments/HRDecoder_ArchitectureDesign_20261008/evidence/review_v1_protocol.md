# HRDecoder-CF v1 教师、预算与采样协议审阅

审查对象：`D:\dataset\experiment\HRDecoder_ArchitectureDesign_20261008\01_架构设计与研究论证.md` 与 `D:\dataset\experiment\HRDecoder_ArchitectureDesign_20261008\02_架构规格.json`，并回查原 HRDecoder 代码、历史教师/基线证据和 Ready、Next、NextPlan、Q 机器记录。下文 `01:`/`02:` 指上述绝对路径；其他相对目录均相对于 `D:\dataset\experiment\`，`evidence/...` 除外，指设计目录下的 `evidence`。只读审查；没有训练、GPU 推理或修改主方案。

## 结论

概率分解、分支融合顺序、40,000 个全局批次与 1,600 次并行 H4 更新在文档内基本自洽；设计也正确地区分了点估计、工程接近阈值和统计等效。当前真正需要实现前锁定的协议问题有三项：H4 训练/部署网格不同；增强后蒸馏目标是原图预测的几何搬运而非教师对增强图的预测，且规格未完全定义搬运/筛选顺序；教师训练概率缓存尚未证实存在。另有两个清楚披露、但不可在结果解读中忽略的限制：教师有学生不可见的原生窗口细节与 DINO 通用预训练，且历史 test 已反复用于研究反馈。

## 具体发现

### 1. H4 训练与部署的空间尺度不一致（主要未解项）

设计规定 H4 在增强后的 2048 画布上训练（`01:257-265,289`；`02 $.H4.training_grid`），部署则在各分支 logits 恢复到原图 H/W 后运行（`01:315-319`；`02 $.H4.inference_grid`）。H4 的两个 3×3 卷积没有空间归一化（`02 $.H4.convolutions`, `$.H4.normalization_in_head=false`），所以同一个邻域在训练和部署时覆盖的原图面积不同；原图尺寸不等于 2048 时，校准头看到的病灶尺度/上下文就变了。冻结 BN running statistics 不能消除此空间尺度差异。旧 H4 已有 1024 crop 训练、原图尺寸推理的不一致；其 1,600 步最佳点来自冻结 D/B 输入，不能证明新 H4 在动态 L/C 输入上同样最优（`evidence/h4_code_audit.md:14-17,56-57`；旧实现 `HRDecoder_审查轮_20260916/tools/h4_post_fusion_corrector.py:141-152,175-202`）。

这不是隐藏问题：新稿已如实列出训练/推理网格，但尚无同网格验证或量化证据。执行与报告时应将其标为未验证的 H4 尺度迁移；不要把旧 H4 的 1,600 步证据说成新头最佳点。若主结果在 40k，H4 才训练过；若 best-val 节点早于 40k，该节点的 H4 仍是恒等头（`01:287,291`）。

### 2. 教师拥有部分学生输入中没有的原生细节

学生输入契约是 2048 画布，主 HRNet 接收 1024，L 的 RGB stem 再以两次 stride-2 降至 512 网格（`01:120-149`；`02 $.input.canvas_hw`, `$.input.backbone_hw`, `$.detail_L.rgb_stem`）。锁定教师 S20 的推理却是 `native1024/center768/halo128`（`FinalDecision_20261005/execution/server_results/reports/final_selection_lock.json $.inference`）；旧 D 局部成员也有窗口/局部信息。故 S20/D 提供的部分软目标是在学生没有同等原生采样路径的细节上产生的，可能出现“输出要拟合的信息已在缩放/步幅中丢失”的容量上限，尤其影响小 MA。它是信息可达性风险，不是已证明的失败。注意原 HRDecoder 所谓 HR crop 也不是原图第二套 backbone：它先把图缩至 1024，并从同一 `lr_feat` 裁特征（`autodl100/work/HRDecoder/mmseg/models/segmentors/HRDecoder.py:216-233,244-253`；`evidence/h4_code_audit.md:32`），因此不应把整个教师描述成原生高分辨率模型。

### 3. 增强软目标合同不够精确，且目前没有 train 缓存

基线几何是 ratio 0.5–2.0、随机 2048 crop、flip、±45° rotate、再 pad（`autodl100/work/HRDecoder/configs/_base_/datasets/hr_ddr_2048.py:25-37`）。设计要求原图 teacher 概率双线性变换、GT 最近邻、valid 单独传播，并承认“搬运教师图”通常不等于重新预测增强图（`01:297-303`）。但 JSON 对 augmentation 只有一句字符串（`02 $.input.augmentation`），没有锁定 teacher warp 的 `align_corners`、旋转边界填充值、抗锯齿/插值次序，以及 `T>0.5` 是在 warp 前还是 warp 后计算。由于双线性插值与阈值化不交换，二者会改变 M 和 KD 像素集合；这是实现可重复性缺项，不只是措辞风险。

蒸馏掩码 `M=v AND ((T>0.5)==Y)` 是“最终融合教师与标签二值判断一致”的筛选，不是置信度估计：T=0.5001 与 T=0.99 在 M 中同等保留；教师与标签不一致的位置完全不收均值/差异蒸馏，只受 GT 主损失监督（`01:243-253`; `02 $.loss.distillation_support`）。设计已正确将其标成未验证，并承认会丢弃争议处软信息。又因教师在同一自定义 train 上训练，这些分数是 in-sample，不是 OOF；train 一致率不能作为泛化置信度。现有记录只确认 val 概率和训练图像/标签读缓存，没有证明 454 张 train 的 T_A/T_U/T_S 已导出；JSON 也标为 `not_verified_present; export required`（`02 $.teacher.train_cache_status`；`evidence/teacher_protocol_audit.md:46-49`）。不能把 val `.npz`、test 计数或历史汇总指标当作训练缓存。

### 4. 同步 40k 与模型选择说法需分开

规格内部计数清楚：40,000 个批次、40,000 次主网 SGD、有效 batch=2；H4 在同批次上并行更新 1,600 次，不另取 1,600 批或增加 backbone 更新（`01:277-291`；`02 $.training.global_batches`, `$.training.main_optimizer_updates`, `$.H4.optimizer_updates`）。因此这是相同的图像更新预算，不是相同 FLOPs/墙钟：新增解码、差异/KL、全分辨率 H4 和教师导出成本均不在 40k 计数内；设计已经明确披露这一点（`01:321-332`）。

比较时必须把固定终点与 val 选点分开。历史 B0 的 42.20% 是本地自定义 454/76/227 划分上的报告值，不是原论文的 32.25%；它来自验证选出的 32k checkpoint，虽完整训练跑到 40k。原 checkpoint 和逐图 test counts 在审计中未找到（`FinalDecision_20261005/结题报告/evidence/original_HRDecoder_baseline_evidence.json $.best_validation.completed_training_updates`, `$.historical_test.inference_reported`, `$.baseline_checkpoint_found_in_this_audit=false`；`01:26`）。所以新学生 40k 对 B0 42.20 是“固定 40k 对历史 best@32k”的预算比较，不是同一选点规则的复现比较；应并列报告 40k 主端点和每 4k scheduled best-val 次端点，且明确基线证据等级。新规格已要求这样报告，不能把两者合成同一数字（`01:291`; `02 $.training.scientific_endpoint`, `$.training.secondary_checkpoint_selection`）。

### 5. 缓存容量和起点哈希仍需落地

设计的 85.125 GiB 是 454 个固定 2048² 图、三教师×四类 FP32 的估算。它同时要求缓存原图 H/W（`01:297-305`），而冻结 `data_identity.json $.splits.train.samples[].hw` 合计 2,346,569,020 原始像素；按相同 48 bytes/像素算约 **104.90 GiB**，平均每图约 5.17M 像素、最大约17.92M。故 85.125 GiB 不是拟议原图坐标缓存的容量预算；压缩或分块可能降低峰值，但未有测量。教师导出时间/存储读写也不在 40k 学生步数里。

另，文中要求严格相同 ImageNet W48 文件与哈希（`01:118,131,400`），但规格 `02 $.student_initialization.sha256=null`，配置 URI 只是来源标识。跑前需要实际固定并记录权重文件哈希、baseline 所用文件身份及初始化张量核验；否则“同起点”仍是要求而不是已验证事实。教师学生起点不相同则是有意设计：教师含 DINOv3 ViT-B/16 的 LVD1689M 通用预训练以及多条历史任务训练谱系，学生仅 ImageNet HRNet。可以称“蒸馏压缩复合教师”，不能称同预训练、同累计计算或同监督预算（`evidence/teacher_protocol_audit.md:37-45`；`02 $.student_initialization`, `$.teacher`）。

## 已对齐的合同与不可混比证据

- **输出顺序无发现矛盾：** HRDecoder 内部 LR/HR 先做 logits 平均；学生各支路 logits 恢复原图尺寸后各自 sigmoid，再做固定概率混合；稳定混合 logit 只供 H4 输出基准；H4 最后只改 MA，阈值严格 `>0.5`（`01:129,192-205,313-319`; `02 $.anchor_A.internal_fusion`, `$.student_fusion`, `$.H4.correction`, `$.evaluation.threshold`）。需要将 H4 网格差异与这条正确的算子顺序分开看。
- **高分辨率负结果分开报告：** Ready 同一 val 配对节点，1k 融合 +0.3665pp，2k −0.0054pp（`Ready_20261002/round2_highres/reports/paired_validation_observation.json $.matched_nodes`）。Next G 的固定融合 test 44.6345% 对旧 P0 44.7809% 低0.1464pp，是另一评估身份，不能并入 Ready 配对差（`evidence/teacher_protocol_audit.md:55-56`）。
- **外部数据不能作因果归因：** Q_MIX 在24k内 DDR windows 144k，而 16k 仅80k，另混入 IDRiD/MAPLES；E_NEG24k单支37.0094%、固定融合44.1643%，F12追加12k单支40.2418%、固定融合44.5888%。DDR曝光、数据、训练阶段和结构未隔离，故这些结果不能单独证明外部数据有害（`HRDecoder_Q_20261003/server_results/reports/stream_Q_MIX_seed42.summary.json $.exposures`; `HRDecoder_NextPlan_20261004/execution/server_results/reports/最终实验报告.md:10-16,21-29`）。
- **DINO空间分支说明 single 与 fusion 不等价：** 同父权重/DDR曝光/日程的 matched val 中，S−C 单支 +1.2459pp@20k、+1.0451pp@32k，但固定25%融合仅 +0.1456/+0.0974pp；不能把单支增益说成最终融合增益（`FinalDecision_20261005/结题报告/HRDecoder项目结题报告.md:287,293,299-304`）。
- **统计边界表述恰当：** 45.8604 是历史固定 test 上的复合系统点估计，不是单模型分数；与 R 的配对图像 bootstrap 区间 −0.2626 至 +0.6519pp 且不含种子方差。该 test 已参与历史研究反馈；关闭本轮重选不使它成为新盲测。设计把 0.5pp 定义为工程容忍度而非等效界限，并否认单次训练可证明稳定性，未发现统计措辞越界（`evidence/teacher_protocol_audit.md:7,62`; `final_locked_test_evidence.json $.conditional_bootstrap_from_original_final_test`; `01:28,352-374`; `02 $.evaluation.close_margin_is_statistical_equivalence_bound=false`, `$.evaluation.historical_test_is_fresh_holdout=false`）。

## 执行前最小记录项

不改主方案也应先补齐可执行记录：教师三路 train-only 缓存的 454 ID、原图 H/W、EX/HE/SE/MA 顺序和 checkpoint/config/code SHA；缓存按实际 H/W 的磁盘和 I/O 预算；teacher warp、ignore 边界与阈值时点；学生 ImageNet 文件 SHA；H4 训练/部署网格差异的明确声明；以及 40k 主端点与 best-val 节点分开汇报。不得使用 val/test 概率训练，也不得把旧自定义 train teacher 用于官方 383/149/225 无泄漏声明：已核对的交集显示旧 train 454 与官方 test 有 142 张重叠（`evidence/teacher_protocol_audit.md:23-31`; `02 $.evaluation.old_teacher_allowed_for_official_split_claim=false`）。

