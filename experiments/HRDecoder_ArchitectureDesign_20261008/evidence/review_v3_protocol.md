# v3 反方复审：数据、预算、容量与教师边界

审查日期：2026-10-08。对象为 `05_v3_机制重构与反驳审查.md` 及其引用的既有协议/审计。没有训练、推理或扩大文献搜索；只做了源码/记录核查和按规格静态参数算术。未修改主方案。

## 结论

没有发现 v3 明确宣称“容量足够”“40k 等计算”“H4 已消除所有泄漏”或“结构首创”；文稿多数重要限制写得恰当。需收紧一处事实措辞、明确几项比较协议：历史 baseline 的具体 ImageNet W48 文件尚未哈希核实，不能说已证实与它同权重；40k 只可称同批次数/更新数；新教师目标缓存及其推理、存储仍是正式开训前的阻塞项；H4 虽不再进入学生前向，仍留在 in-sample 教师目标中。v3 的规格可算出新增约6.40M参数，但这只证实增量，不证实容量充分。

## 数据与 40k 公平口径

- `05_v3_机制重构与反驳审查.md:167` 将“原 ImageNet W48 文件及原随机解码头初始化顺序与基线相同”写成既成事实。现有审计只确认历史 baseline 用自定义 454/76/227 split、40k 训练；历史 checkpoint 和逐图 test counts 未找到，ImageNet W48 “同文件”SHA也待核验。应写为“拟按相同初始化规程；实现时需核对权重文件 SHA/状态”，不能宣称已复现同一初始张量。依据：[review_v2_protocol.md:17–19](D:/dataset/experiment/HRDecoder_ArchitectureDesign_20261008/evidence/review_v2_protocol.md)、[teacher_protocol_audit.md:21–33](D:/dataset/experiment/HRDecoder_ArchitectureDesign_20261008/evidence/teacher_protocol_audit.md)。
- batch=2、SGD、seed42、40k runner 与旧配置相符：[01_板块一_baseline实验配置.py:36–58](D:/dataset/lesion_segmentation/01_板块一_baseline实验配置.py)。但历史 42.20% 是 val 选出的 best@32k，不是40k checkpoint成绩；v3 第7节已正确说明不能把它当复现对照。公平对照应同样报告40k固定端点和预先定义的val选点，并把旧42.20%只作历史参照。
- 该 split 是自定义 454/76/227。既有核查显示旧自定义 train 与官方 DDR test 重叠142张、与官方 valid 重叠80张；若执行本方案，学生 train 也会把这些官方 test/valid 图像作为训练样本，旧教师同样已在该自定义 train 上训练。不能将结果包装成官方 383/149/225 协议成绩。v3 当前应明确限定在自定义 split；官方协议要另建隔离教师和学生。见 [teacher_protocol_audit.md:23–31](D:/dataset/experiment/HRDecoder_ArchitectureDesign_20261008/evidence/teacher_protocol_audit.md)。这不是对文稿当前自定义测试方案的反证。
- “同一两张图、同一增强参数”见 v3 第170行，但还需将 seed、逐步样本/裁块/旋转翻转/光度增强流纳入可审计身份，并对 B0、B1、v3 逐项配对。只有批次数相同不足以保证曝光序列相同。历史 baseline best@32k、固定42.20和新目标损失的有效域修正也不是完全相同训练配方；v2审计已要求以相同 `v_geo` 重跑匹配 B0，不能将新旧分数差直接归因于结构。[review_v2_protocol.md:17–19](D:/dataset/experiment/HRDecoder_ArchitectureDesign_20261008/evidence/review_v2_protocol.md)
- v3 已正确限定同40k并非等 FLOPs/速度/显存（第168、172行）。局部/全局新路径、辅助梯度反传和2048细节输入都会改变每步计算；此外离线生成教师缓存的推理和I/O不计入40k学生更新。应把这部分列为一次性数据准备成本，与学生训练预算分开报告。

## 新增参数能证明什么

按 v3 第79–112行的层定义静态计数：逐像素 `P/Down` 按无 bias 卷积加 affine Channel-LN 计，`B(c,k,e)` 按 depthwise `k×k`、affine LN 和两层无 bias 1×1 扩张/收缩计。新增局部路径约2,918,816参数，全局路径约3,444,128，分歧读出与最终分类器约40,960，总计 **6,403,904** 个新增参数；不含原 HRNet-W48 和 HRDecoder。此数是纸面规格算术，不是已实现模型的 `numel()` 或 checkpoint 统计。v2规格的新增参数记录为106,809（`02_架构规格.json:$.new_parameters_total`），所以 v3 把新增容量显著提高；但模型参数数目既不能推出“10万一定不够”，也不能推出“640万足以承载三教师知识”。文稿第9行的“容量风险成立”宜解释成待测的合理风险，不是由历史结果或参数比证明的结论；第47、49、204行的“不保证/不能声称容量充分”是正确边界。

对照边界：v3 第193行“同一 v3 网络、λD=0”可作为同容量、单一最终 T KD 对照，前提是保留相同 L/G 路径、分类器和初始化，只关闭有符号分歧辅助损失；这能测辅助监督的增量。B0/B1 是较小的原 HRDecoder 基线，比较完整 v3 与它只能评估整个系统包的收益，不能分开证明容量必要。第192–198行应明确这两个结论各由哪组对照支持。第195行 Hydra 式对照还缺可执行定义：成员目标如何分配、输出如何合并，以及参数量/训练 FLOPs如何匹配；不能只在结果中标为“Hydra式”就称为公平比较。

## 教师缓存与目标分布

- 当前454张自定义 train 的 A/U/S 三路概率仍未导出。已有记录中的893张训练缓存是 RGB/标签读缓存，76张概率是验证集 P0；没有 train soft target 或 spatial feature cache。[teacher_protocol_audit.md:46–50](D:/dataset/experiment/HRDecoder_ArchitectureDesign_20261008/evidence/teacher_protocol_audit.md)
- 104.90 GiB 是三路×四类×FP32在454张原图像素上的未压缩 tensor payload，先前逐尺寸复算通过；不是已有文件，也不是实测落盘大小（[review_v2_protocol.md:27–31](D:/dataset/experiment/HRDecoder_ArchitectureDesign_20261008/evidence/review_v2_protocol.md)）。训练前仍需一次只针对454个 train IDs 的教师前向/导出，并校验 ID、尺寸、类序、权重/代码/推理配置 SHA；val/test概率不能补位。该导出计算、空间及压缩方案要单列。
- v3第170行明确使用原图坐标教师 soft map，再套学生几何。v2已明确说明“变换原图教师图不等于在增强图上重新推理”，并指出学生视图/教师输出存在偏移风险（[04_v2_修订架构.md:181–189](D:/dataset/experiment/HRDecoder_ArchitectureDesign_20261008/04_v2_修订架构.md)）。v3说数据合同沿用 v2，但建议保留这条明文限制，尤其若原图光度增强只作用于学生输入时。
- 这些教师在同一自定义 train 上训练；A 还包含旧 H4 的成员内预测。故目标是 in-sample 软预测，不是 OOF 泛化标签。其软目标可能过拟合训练图，不代表它们泄漏到本轮自定义 val/test；但也不能称为样本独立。若换用官方 test，则上一节的图像重叠构成直接协议污染。

## H4 与“无泄漏”用语

没有发现 v3 虚称“取消 H4 后无泄漏”。第11行把 H4 风险限定为训练内 stacking 过拟合/分布偏移，而非验证/测试标签泄漏；第178–180行又说明旧 H4 保留在教师 A/T、不是 OOF，取消在线 H4 不给整个学生 OOF 保证。这与既有审计一致：旧头用同训练 split 的 D@2000/B reference 成员内预测，没有留图/折外预测（[round3_h4_independence.md:7–15,23](D:/dataset/experiment/HRDecoder_ArchitectureDesign_20261008/evidence/round3_h4_independence.md)）。该审计也提醒，没有逐样本训练访问计数，不能说每张图在D@2000前都必然被采到；“不是OOF”是正确结论。

第176行的13.96G是旧 H4 在2048×2048输出网格上的单次卷积前向 MAC，适用条件写清楚，数值与 [round3_h4_independence.md:27](D:/dataset/experiment/HRDecoder_ArchitectureDesign_20261008/evidence/round3_h4_independence.md) 一致。它只表示学生不再每步执行在线 H4；教师 A/T 的 train cache 导出仍会调用含 H4 的教师推理，新路径成本也要另计。故“节省14G”不能扩展为整个蒸馏项目总计算下降，v3第176行已作此限制。

## 创新性表述边界

就文内已有 Hydra/Nam/Allen-Zhu 论证而言，v3没有主张多头集成蒸馏或局部/全局模块首创；Hydra 用于承认近邻方法并要求匹配对照，Nam 只作 train 预测相似度/多头多样性风险提示，且明确不把其分类输入扰动直接迁移到病灶数据；Allen-Zhu & Li 被用来说明“可压缩存在性依赖数据条件”，未推导 DDR 容量结论。这样的引用边界较审慎。第188行应继续称为**待验证的经验假设**，只有同预算、同容量及明确定义的 Hydra/普通 KD 对照通过后，才讨论经验贡献；这轮没有新文献搜索，也不能据此确认论文层面的新颖性。

## 执行前必须闭合

1. 将初始化说法改为规程要求，取得并归档 HRNet 与 decoder 起点的身份/哈希；固定 split、seed 和逐步采样/增强流。
2. 先生成并验证454张 train-only 教师目标；保留104.90 GiB为未压缩估算，并单列教师推理、I/O与存储开销。不得用验证/测试资产替代。
3. 将 `λD=0` 明确为完整保留 v3 架构的单一最终 T KD 对照；明确 B0/B1 的有效域、40k选点规则；给 Hydra 式成员监督写可运行的参数/计算匹配定义。
4. 将报告范围限定在既有自定义 split，并声明历史 test反馈和教师 in-sample 目标；若要求官方 383/149/225 或患者级独立结论，必须更换隔离数据和教师。
5. 当前 v3 等待算子检查（05第3行），v2 CPU probe 不是 v3 的实现验收；在 v3 模块真实计数、梯度路由、mask几何、tile行为与资源成本通过检查前，不把纸面参数数或算子自由度称为已验证模型能力。

## 参数计数更正（2026-10-08）

复核新增的 [v3_module_checks.json](D:/dataset/experiment/HRDecoder_ArchitectureDesign_20261008/evidence/v3_module_checks.json) 后，更正上文静态计数：CPU 模块实例的 `$.parameters.actual` 与 `$.parameters.analytic_expected` 一致，新增参数合计 **6,419,648**，分组为 L 2,918,816、G 3,459,872、fusion 36,736、contrast 4,224。旧静态算法漏计 G 路径的 `P(96,128)` 与 `P(48,64)` 两个 F2/F1 投影，合计15,744；把两组 contrast 读出归入 fusion 造成了组别差异，但不影响原fusion+contrast小计。这一实例结果取代上文6,403,904总数及3,444,128/40,960分组数。

该检查为 CPU、128×128小画布，特征提供器是随机 shape stub；未加载真实 HRNet 权重、未运行2048前向、未创建 optimizer 或更新参数（`$.scope`）。因此它确认新增模块定义的参数量与小画布接口，不是完整 HRNet 集成后的训练验证，也不改变“参数增量不能证明容量足够”的结论。
