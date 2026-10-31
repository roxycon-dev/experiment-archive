# 实际方法实现审计：HRDecoder 起点至锁定空间 EMA 完整系统

审计日期：2026-10-23。只读本地源代码、保存配置与原始结果；没有连接服务器、训练、模型推理或测试集调参。下列路径均相对 `D:/dataset/experiment/`。行号为当前本地文本的一基索引；`method_source_manifest.json` 保存被审计文件 SHA256 与功能选择器，便于复核。论文名称应描述机制，实验缩写仅用于来源定位。

## 1. 身份边界和可用命名

最终交付不是一个单一 HRDecoder 或单一 DINO 网络，而是先训练后固定的多模型系统：

\[
 P_{\mathrm{final}}=0.75P_0+0.25P_{\mathrm{spatial\ DINO,EMA}},\quad
 P_0=0.75P_{\mathrm{HR,corr}}+0.25P_{\mathrm{ConvNeXt}}.
\]

展开为 `0.5625 corrected HRDecoder experts + 0.1875 ConvNeXt + 0.25 spatial DINO EMA`。新空间 DINO 分支由 F12 父模型初始化并追加 20,000 步，采用其 EMA。其父链是 DINOv3 通用预训练 → DDR 16k（前2k仅解码器，之后末4块）→ 12k全主干追加训练 → 当前空间分支。最终分支无 TJDR 监督，不继承 Q_MIX，也不是后来的 NativePyramid、JointContext 或未实施的新设计。

最终测试45.8603838327%属于完整概率系统；42.2703219049%是20k空间分支的**验证** mIoU。两者不能互换。原设计 `configs/protocol.json` 写两种子，但实际执行由 `reports/single_seed_parallel_amendment.json:4-30` 改为仅seed42、四臂、24个端点/公式候选。不能依原设计声称两种子完成。

锁定 `execution/server_results/reports/final_selection_lock.json`：S，20k，fixed_fusion；checkpoint SHA256 `c933256b07556689a7fd199dd80e50d4d0c27eb6b9ea782713d95ec51e6d8d62`；lock SHA256 `3146ca1edafdffdae7bf4987524bddc502209929e5d6d5b0e7ba49f12b6bd66e`。此处锁定与数值选择审计由并行 evidence 材料进一步记录。

## 2. HRDecoder 基础和原图细节改造

源 `HRDecoder_Next_20261003/vendor/champion_repo/mmseg/models/segmentors/HRDecoder.py`：

- `:145-169`：末级HRNet多尺度特征对齐后拼接；高分辨率路径在**同一份较低分辨率主干特征**裁区再放大解码，不是再次读取原始图像。
- `:216-231`：先把输入画布缩放到 `hr_scale` 提取主干特征，分别解码整图和特征裁区，再平均两路logit，之后可加入细节残差与MA细化。
- `:41-61`：可选直接读取当前图像画布的 DetailBranch / DetailHead；学习系数 `detail_scale` 初始化为0。

\[
 z_{\mathrm{detail}}=z_{\mathrm{HRDecoder}}+\alpha\,\operatorname{Up}(D(x)),\qquad\alpha_0=0.
\]

`mmseg/models/necks/detail_branch.py:34-45,65-74`：3×3卷积两次stride2，再3×3 stride1，32通道stride4；两层3×3细节头后1×1四类输出。`configs/lesion/exp/e4_detail_split613.py:16-31` 设置BN、ReLU。所谓“原图”是相对于送入HRNet前再缩小的图像画布；该专家的画布仍为2048×2048，不能解释为完全不改变采集图像原始像素。

固定参考模型是从细节专家继续训练得到的 HR100Refinement。`configs/lesion/exp/hr100_b.py:1-2` 继承 `hr100_a.py` 但明确将 `semantic_refinement=False`：**保留参考B不含早期语义MA细化器**。`hr100_refinement.py:13-29` 的早期细化器属于未选择的A配置，不是B结构；不能因共同类名而自动归入最终模型。B仍继承 `hard_weight=.02`：`:32-43` 的额外MA hard loss 保留阳性及至多65,536个hard negative（数量 `min(Nneg,max(256,3Npos),65536)`）；`hr100_a.py` 确认冻结BN、SGD5e-4、8k追加日程。不要把参考模型误写为未经改动的原HRDecoder。

## 3. 原尺寸局部学习的准确几何

源 `hrplus_patch.py:52-155` 和 `configs/lesion/exp/hrplus_patch_dice.py:7-61`：

- 图像先强制缩放2048×2048（`keep_ratio=False`），随机旋转±45°，以0.5混合均匀与前景定位1024裁块，随机水平翻转0.5。
- 学生和冻结参考均从成熟参考权重初始化。学生训练采用裁块分割损失；不经过最终集成，也不含参考融合损失。`hrplus_patch.py:103-113` 明确将ignore255位置的logit置-1e4以从平方Dice中移除该位置；`lovasz_weight=0`。
- SGD lr5e-4、momentum.9、weight_decay5e-4；200步warmup、线性poly、min_lr5e-5；batch2、clip5；采用2k保存最佳权重。保存配置max_iters20k并不等于被选模型训练20k。
- `hrplus_core.py:24-57` 与 `hrplus_patch.py:116-138`：推理在该画布上1024窗口、768步幅，重叠区域**平均logit**，恢复原图尺寸后sigmoid，并按metadata撤销flip。
- 学生概率与冻结参考整图概率各半：
  \(P_{\mathrm{local}}=.5P_{\mathrm{student}}+.5P_{\mathrm{reference}}\)。

因此不能将HR局部专家称为“native-resolution tiling”。DINO的原生图上下文窗口与它有不同几何协议。自然表述为“image-space local learning on the 2048-pixel canvas”，以区分原HRDecoder的feature-space crops。

## 4. 两种MA残差机制必须区分

### 4a. 专家内部语义条件MA残差

`ma_residual.py:32-67` 的 SemanticMAResidualV2：RGB经16/32通道stride4细节图；主干720→32语义图；基础MA logit→8通道图。对齐拼成72通道，经两层32通道卷积后1通道输出，末层零初始化。仅MA相加，另外三类直通。`HRPlusMAResidual` 是从成熟参考复制冻结底座，另保留冻结参考用于各半概率组合；`configs/lesion/exp/hrplus_mares.py` 确认仅训头、SGD.01、warmup300、batch2、clip5、无额外hard loss。保留1,500步结果。

这与后续使用7个输入通道、3,361参数的MA后融合校正头不是同一个结构。

### 4b. 小型有界后融合校正头

原训练源 `HRDecoder_审查轮_20260916/tools/h4_post_fusion_corrector.py:62-87,183-232`：

\[
 \phi=[x,\ell(P_{b,MA}),\ell(P_{s,MA}),\ell(P_{r,MA}),P_{s,MA}-P_{r,MA}],
\]
\[
 \delta=6\tanh h_\theta(\phi),\quad
 P'_{MA}=\sigma[\ell(P_{b,MA})+\delta],\quad
 P'_c=P_{b,c}\ (c\ne MA).
\]

\(\ell(p)=\log(\tilde p/(1-\tilde p))\)，\(\tilde p=\operatorname{clip}(p,10^{-4},1-10^{-4})\)。输入RGB是归一化画布按需要对齐到原图坐标，不是裸RGB整数。头为 Conv3×3(7→16) / ReLU / Conv3×3(16→16) / ReLU / Conv1×1(16→1)，合计3,361参数，末层零初始化（严格零残差作用于**clamped**基础概率）。只更新头，Adam.001，clip5，线性poly；保留1600步头。

**训练损失不是DINO损失。** 原实现 `:67-70`：
\[
 L_{H}=.5\operatorname{BCEWithLogits}(z',y)+.5\left[1-\frac{2\sum p'y}{\sum(p')^2+\sum y^2+1}\right].
\]

训练 `:216-219` 将ignore255标签改为背景0，BCE与Dice没有valid mask；不能声称所有阶段统一严格忽略padding。验证计数则排除了255。

**已确认的训练/应用迁移。** 训练 `:205-214` 中 \(P_b=.5P_s+.5P_r\)。而最终应用源 `HRDecoder_Next_20261003/legacy/runtime.py:735-791` 先构造四成员翻转增强概率集成，再以该较大集成的MA概率作为 \(P_b\) 及feature输入；\(P_s,P_r\) 仍来自局部成员未flip的student/reference。它是保留头在不同基础分布上的应用，不是按最终集成输入重新训练。`h5_ensemble_corrector.py:140-163` 亦直接确认这一过程。正文/补充须披露该差异，不将它写成端到端融合训练或严格匹配的纠错学习。

## 5. 固定专家概率集成和翻转增强

`legacy/runtime.py:660-662,735-753`：四个成员是局部student/reference专家、语义MA残差专家、成熟整图参考、早期细节专家。每成员原图/水平flip预测后复原坐标并平均，最终权重3:2:1:1：

\[
 P_{\mathrm{HR,mix}}=(3P_{\mathrm{local}}+2P_{\mathrm{MAres}}+P_{\mathrm{reference}}+P_{\mathrm{detail}})/7.
\]

然后按第4b节仅修改MA，得到 \(P_{\mathrm{HR,corr}}\)。局部与MA残差专家内部又各含固定参考，不能把这四成员等同为四次等代价的forward，也不能把旧系统称为一个单网络。`member_prob:629-656` flip经mmseg撤销，不二次翻转输出。

配对h7来源43.2289242453→44.2115035614%，MA24.7254712788→28.6557885432，非MA计数相同。后续最终系统消费的是历史冻结概率导出版本44.2110%，并非把h7的44.2115直接替换进去。此差异虽小，来源必须保留，不能混算精确相邻增量。

## 6. 独立ConvNeXt伙伴

`HRDecoder_Next_20261003/trainer/models/convnext_unet.py:36-38,125-168,227-264`：ConvNeXt-Small depths=(3,3,27,3)，channels=(96,192,384,768)，stride=(4,8,16,32)；编码器为原生LayerNorm。UNet式解码从stride32起256通道，逐级双线性上采样与skip拼接，经两层3×3/GN/GELU，128/64/32/16/8通道恢复到stride1，1×1输出4类；并没有该分支的额外detail branch。保存伙伴是历史 whole-image 分支，不是其后高分辨率续训版本。其概率占固定伙伴的25%，旧纠正专家占75%；它不是训练损失或类别自适应权重。

## 7. DINO父网络和四尺度解码

`HRDecoder_FinalDecision_20261005/execution/qlib/model.py:25-77`：DINOv3 ViT-B/16，12块、768维、16patch。取0-based块(2,5,8,11)，每个patch map仍为stride16；分别1×1投影128，再×4、×2、不变、stride2卷积重构stride(4,8,16,32)。不能将这四路称为ViT天然四尺度。

每尺度3×3/GN/GELU，最粗层起用残差细化；逐级上采样相加。RGB支路3→32(stride2)→64(stride2)→64，拼接128通道解码图得到192，3×3压128、1×1输出4类、双线性恢复窗口尺寸。`set_update:41-49` 是早期Q训练冻结方案；当前FinalModel从已训练F12继承并全12块微调，不能将Q的末4块冻结规则套到当前S。

记录identity中的88,263,748是构造父Q网络时的total_parameters，**不包含新增空间模块**，不能据此报告S参数量。未对完整多模型系统做统一参数/FLOPs/延时核算，不应作轻量或高效主张。

## 8. 当前空间交互模块

`execution/spatial.py:134-166`：5个stride2 Conv3×3，channels3→32→64→128→256→256，均GN(8组)/GELU。stride4细节图64通道保留；stride8/16/32的128/256/256图经1×1到768，flatten并加3个learned level embedding。

`DeformAttention:59-120`：维度768，12头，每头64维，每层4采样点；offset和weight由query线性预测，softmax覆盖同头全部level×point，使用normalized reference cell centers，offset除每value层(width,height)，双线性取值；周围AMP但采样kernel强制FP32。其几何与作者ViT-Adapter/Deformable-DETR实现相关，是借鉴式适配，不是首创双向可变形注意力。

\[
 A(q,F)=W_o\operatorname{Concat}_{m=1}^{12}\left[\sum_{l=1}^{L}\sum_{k=1}^{4}a_{mlk}(q)W_v^mF_l(r+\Delta_{mlk}(q))\right].
\]

`execution/final_model.py:78-108`：第0/3/6/9块前，以patch token为query，从3层spatial tokens注入；经过3个原DINO块后以spatial tokens为query，从单层stride16 patch grid提取。LayerNorm分别作用于查询/值。CLS和4 storage/register tokens不参加空间交互，保留它们在原ViT块中的作用与官方RoPE。

\[
 T\leftarrow T+A(\operatorname{LN}T,\operatorname{LN}C),\qquad
 C\leftarrow C+A(\operatorname{LN}C,\operatorname{LN}T_{\mathrm{after\ 3\ blocks}}).
\]

注入output projection为0，提取output projection非零；4个spatial-to-decoder1×1全为0，故起始附加残差为0。`final_model.py:14-35` 四路spatial maps分别64/768/768/768→128，在尺度对齐**之后**、原scale_refine**之前**相加。stride4来自浅层卷积图；stride8/16/32来自第4次提取后的tokens。保持原解码器和RGB支路。

**顺序细节**：每一阶段的ViT特征是在该阶段提取器更新C之前保存；最后的spatial maps取最后一次C更新。概念图不得反画为“提取更新了当前ViT已保存特征”。

## 9. 当前分支训练、损失、EMA

源 `engine.py:8-34`、`schedule.py:7-82`、`state.py:10-28`、`qlib/loss.py:5-25`、`qlib/geometry.py:17-30,63-104`：

- 原生图1024窗口；只对central768×768监督，halo各128；ImageNet mean/std，归一化空间零padding；每更新DDR8窗口=4个foreground+4个uniform，microbatch2×4累计。
- BF16 forward，float32 BCE/Dice；分母与归约以每类batch+spatial有效像素计算，再对具有有效像素的类别平均。\(v\) 屏蔽ignore/padding，全部无效microbatch报错。
\[
 L_B^c={\sum v\operatorname{BCEWithLogits}(z,y)\over\sum v},\quad
 L_D^c=1-{2\sum vpy+1\over\sum vp+\sum vy+1},\quad
 L={1\over|\mathcal C_v|}\sum_{c\in\mathcal C_v}(.5L_B^c+.5L_D^c).
\]
- 一次更新为4个DDR micro-loss均值；当前空间S无source loss，D/SD对照前20k额外2个TJDR micro-loss均值×.25。因此不能将source混合损失描述为最终S的训练。
- 全12块微调；mask token、cls_norm/local_cls_norm冻结。AdamW betas(.9,.999),eps1e-8,wd.01，bias/norm no decay，clip1。block b峰值1e-5×.8^(11-b)，embedding/cls/register峰值1e-5×.8^12；final norm1e-5；原decoder5e-5；spatial及residual1e-4。500步线性warmup，此后以固定32k horizon余弦下降到峰值1%。20k权重是32k日程中的注册节点，不是另设20k cosine日程。
- EMA在u0复制完整模型；每次成功optimizer.step及scheduler.step后 `ema=.999ema+.001model`；浮点state平均，非浮点buffer直接复制。所有臂均只用EMA进行候选验证/选择；未做raw/EMA竞赛。
- 当前F12严格加载model，不继承父optimizer；新增模块独立初始化RNG；C/S共享保存DDR stream、数据/增强与原backbone随机流，额外模块不任意改变paired RNG。相同DDR曝光不等同相同算力，S增加参数与计算。

## 10. 推理、阈值和计数

`execution/validation.py:48-91` 与 `qlib/geometry.py:17-30,99-104`：DINO以768栅格遍历原图，读取1024上下文，只放回central768输出；边界按真实尺寸裁掉padding；不平均邻窗中心、不对DINO加flip TTA。cast logits FP32后sigmoid，将输出直接写回original image坐标，与已冻结且对齐的P0 float32概率融合。最终严格 `p > .5`，不是 `>=.5`、argmax或softmax；四类独立，background不进入mIoU。

`qlib/metrics.py:6-25`：按有效像素累计每类所有图的TP/FP/FN后算IoU，再对4类macro平均；不是图像IoU先平均。

\[
 \operatorname{IoU}_c={TP_c\over TP_c+FP_c+FN_c},\quad
 \operatorname{mIoU}={1\over4}\sum_c\operatorname{IoU}_c.
\]

原实现 `validation.py:20-30` 的 `new_false_positive_regions` 实际计算 `prediction & ~GT` 的8连通域，不是“与另一模型逐域配对后新增的FP”，甚至不要求整个预测连通域完全不接触GT；一个与GT相交的预测物体的外延亦能贡献FP连通域。GT漏域则为GT8连通域完全零像素重叠。不得称临床实例敏感性或相对模型新增误检数。

## 11. 可支持贡献与禁用说法

可以将故事组织为HRDecoder基础上的图像细节、image-space局部学习、定向MA修正及跨架构/空间互补的累计系统设计。局部机制有明确实现依据，但各阶段联合改变训练/模型时不能拆成纯单因素因果增量。

仅当前matched spatial C/S对照支持同父/同种子/同DDR stream/同日程下的空间适配验证差异（仍混合更大参数与计算）。MA头在固定集成输出上的配对前后计数支持目标通道干预，不能消除其训练/应用分布变化与历史test反馈。细节专家独立分数低于历史基线，应将其保留价值描述为所选集成互补，不声称已证明单网提升。

禁止：S单网test45.8604；顶刊级首创保证；NativePyramid/新设计继承旧成绩；两种子S已完成；统一各阶段loss；所有专家native输入；完整系统只有一个backbone；原始HRDecoder是当前固定B参考；将平滑/EMA/全主干各自归因；测试全新盲测；48k父链与基线40k等预算；完整系统参数量=父Q的88M。
