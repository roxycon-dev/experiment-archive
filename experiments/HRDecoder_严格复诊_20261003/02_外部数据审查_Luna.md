# HRDecoder 2026-10-02/03 外部数据与预训练证据复诊

复诊范围：只读旧实验、本机用户数据源、DDR train/val 图像头和标签掩膜；未访问 SSH/服务器、未训练、未读取 test 预测。统计使用 Python 3.13、Pillow、NumPy 及现有 OpenCV，仅处理 IDRiD 54 张图/216 张mask与 DDR train/val 454+76 张图及融合标签。

**主要结论：外部预训练确实执行完成，但 IDRiD 监督来自已预处理方形图上的用户多边形栅格，不是官方原图/官方逐像素GT。** 本机源文件 SHA 与实验拷贝逐一吻合，代码映射与坐标栅格化能复现现有mask；这只能证明输入链路一致，不能证明多边形与官方像素标注等价。全54张都有MA多边形，因此没有“整类MA漏标”的证据；MA像素占比高于DDR，但未找到官方像素GT配对，不能据此认定多边形面积偏大。另一个已确认的比较混杂是 U1/H1 的 DDR val 被拉伸成2048×2048，而训练增强保留图像宽高比、U0按原尺寸整图推理。

## 1. IDRiD来源、几何和标签生成

当前源图位于 `D:\dataset\best_segmentation\images\training\IDRiD_01.jpg`…`IDRiD_54.jpg`，源多边形位于 `D:\dataset\best_segmentation\labels\training\IDRiD_01.txt`…`IDRiD_54.txt`。逐一核对54张后，源JPEG SHA与 `assets\external\IDRiD\images\train\` 拷贝及 manifest `image.sha256` 一致；54份源 `.txt` SHA均与对应 manifest `notes` 中的 `polygon_label_sha256` 一致。目录内每个ID只有一张图及一份标签，没有 `rot/aug` 后缀或同ID多版本。以上确认用户处理后文件就是预训练输入，但 SHA 对应的是用户源文件，不是官方原图SHA。

源JPEG为近方形处理图，不是4:3/3:2官方原始视野。例：`IDRiD_01` 4194×4194、`IDRiD_28` 3416×3416、`IDRiD_54` 3418×3418；54张宽高均在3413–4194之间。实验manifest的 `notes` 明确写 `original_dimensions_means_user_supplied_processed_image_dimensions`。在本机 `D:\dataset` 当前副本中未找到相应官方原尺寸IDRiD图或原始分割GT用于SHA/像素对照；因此可证明来源是 `best_segmentation` 用户处理目录，不能证明此前是否做过裁剪、拉伸或旋转，也不能恢复其到官方坐标的变换矩阵。`IDRiD_28_training_metrics.txt` 的14像素差异比较对象是旧 `best_segmentation` 与现在已不存在的 `best_segmentation_two` 用户标签，不是官方GT。

实际转换实现（`tools\prepare_external.py:21-28,30-50,54-73`）：类映射 0=MA、1=HE、2=EX、3=SE；读取归一化顶点后按 `round(xy*[w,h])` 转像素并clip到图像边界；每类独立建立空白mask、用 `cv2.fillPoly` 填充，输出0/255 PNG，类别之间保留重叠。`prepare_external.py:68-73` 备注直言 `polygon_rasterized_not_original_official_pixel_masks`。我在 `IDRiD_01/28/54` 上以同一代码重栅格化12个类别掩膜，全部与当前资产逐像素一致；这排除了这三例的坐标/类映射落盘偏差，**没有**验证绘制边界是否等于官方像素GT。

实验资产统计为54张JPEG、216张四类PNG。全54图中，EX/HE/SE/MA都写入manifest并统一记为 `annotated=true, valid_mode=full, valid_mask_path=null`；`external_dataset.py:179-200` 因而把整幅矩形都作为有效监督，`valid_mode=none/file` 才会屏蔽类别/像素。源多边形中HE有1张无多边形、SE有28张无多边形；转换代码将这类mask保留为全零，训练时即视为全图负样本。MA和EX均54/54有多边形。**“合同中四类齐全”只证明文件字段齐全；多边形覆盖是否穷尽所有真病灶仍未验证。**

## 2. IDRiD 与 DDR 的小型密度对照

统计分母是原始矩形图像面积；不同摄像机分辨率、处理尺度、标签编码、图像选择和视野边缘都影响像素比例，因此这不是临床患病率或标注误差估计。IDRiD源mask允许多类像素重叠；DDR训练读取单张0..4融合PNG并转成互斥的one-vs-rest通道，逐类TIF不参与训练接口。IDRiD有22/54张至少一对类别mask发生重叠（两两重叠像素之和6410），DDR融合格式则不可能表示重叠。这是来源间已确认的监督语义差异，不自动等同代码错误。

| 类别 | IDRiD 54张 阳性图/54；阳性像素；像素占比 | DDR train 454张 阳性图/454；阳性像素；占比 | DDR val 76张 阳性图/76；阳性像素；占比 |
|---|---:|---:|---:|
| EX | 54；5,208,568；0.7921% | 303；5,740,603；0.2446% | 46；290,803；0.0725% |
| HE | 53；6,253,548；0.9510% | 366；8,863,815；0.3777% | 63；1,205,924；0.3006% |
| SE | 26；1,219,796；0.1855% | 143；1,327,424；0.0566% | 24；201,404；0.0502% |
| MA | 54；703,313；0.1070% | 338；527,407；0.0225% | 59；69,284；0.0173% |

IDRiD的MA矩形像素占比约为DDR train的4.8倍、DDR val的6.2倍；EX/HE/SE也都更高。这个发现值得在下一方案里控制来源分布和缩放，但不足以说MA标签面积偏差：IDRiD是经挑选的用户处理图及独立多边形mask，DDR是互斥融合标签；本次没有IDRiD官方像素掩膜配对、逐实例边界误差或相同病例对照。源多边形统计为EX 7,526、HE 1,365、SE 112、MA 2,412个；**没有MA整类空白图，未发现MA映射错类或生成mask落盘错位。**

## 3. 划分、训练执行证据及解释边界

`assets\external\PROVENANCE.json` 的 `$.n_train=54`、`$.n_test_used=0`、`$.curated_idrid=true`、`$.maples_available=false`；`external_train.jsonl` 与 `external_train_dedup.jsonl` 都是54行且SHA相同。`reports\external_partition.json` 的 `$.method` 为 source-stratified stable-hash rank、ceil 10%、seed42；`$.train_ids` 有48项、`$.dev_ids` 有6项：IDRiD_22、43、47、37、17、44。记录 `patient_id=null`，所以是图片级划分，没有可审计的患者级分割证明。

本机镜像配置 `server_results\configs\resolved\U1_convnext_unet_ext_pretrain.yaml:50-78` 与 `H1_hrdecoder_ext_pretrain.yaml:45-74` 均明确使用 `external_train_dedup.jsonl`、只允许 `train`、禁止 `test`，以外部train派生dev选择预训练节点。`trainer\data\external_dataset.py:71-78,101-118,237-270` 对test作硬拒绝，并以稳定哈希将54张按48/6拆分。`server_results\work_dirs\U1_external_pretrain\train.log:2,39` 和 H1 对应 `train.log:2,42` 均记录正式 external_pretrain 完成2400 updates；`completion_summary.json` 的 `$.updates_done=2400`、`$.best.mIoU_lesion4` 分别为 U1 0.271605（27.1605%）和 H1 0.429998（42.9998%），最佳节点均为2400。外部dev只有6张用户多边形图，且U1为1024/768滑窗、H1为整图HRDecoder推理；这两个分数不能横向判断架构优劣，也不能视为官方GT泛化结果。

因此有**已执行外部监督优化**的证据，不能说预训练没有运行或因工程故障失败；但没有证据证明该路线优于公平匹配的DDR-only控制。完整报告 `HRDecoder_完整实验报告_20261003.md:19-21,148-154` 明确指出主臂未匹配、旧控制有差异，且官方像素IDRiD/MAPLES/FGADR未检验。外部dev分数不等于DDR质量，DDR续训后果也同时受网络/解码器/训练预算/评估几何影响，不能将变化单归因于外部预训练。

## 4. DDR val几何差异

本机 `D:\HRDecoder\data\DDR_split_613\images\val` 76张的尺寸头统计：17种分辨率；宽1380–4496、高1382–3000；长边1382–4496（中位2592）、短边1380–3000；73张横图、2张竖图、1张近正方形；W/H中位数1.4993，42/76大于1.4。例：`007-1774-100.jpg` 2592×1728（1.5）、`007-1829-100.jpg` 1956×1934（1.0114）、`007-1900-100.jpg` 2048×1536（1.3333）。

训练增强 `trainer\data\augment.py:72-83` 先用同一factor缩放高宽（`min(side/h,side/w)`），保留宽高比；但U1配置 `server_results\configs\resolved\U1_convnext_unet_ext_pretrain.yaml:158-163` 和H1对应 `:155-160` 设置 `resize_to=[2048,2048]`。`trainer\eval.py:134-137` 直接 `cv2.resize` 到精确方形，所以原始W/H为1.5的图被各向异性变形约1.5倍；U1随后用1024窗口/768步长，H1整图推理。完整报告第2节说明U0沿用原生整图几何。该输入/推理几何差异是**已证实的评估混杂因素**，但没有当前证据能量化它对IoU的因果影响；它与训练日程因素应分开设计消融。

## 5. MAPLES-DR / FGADR 状态

MAPLES并非“毫无资产”：有 `assets\MAPLES-DR.zip`、`assets\maples_labels\train` 与 `test` 标注文件夹（病灶标签inventory为train 138、test 60；这里只数文件，没有用test标签做方案选择）。但 `reports\maples_asset.json` 的 `$.images_available=false`，没有匹配的MESSIDOR原图；`tools\prepare_external.py:77-88,134-136` 要求138张train标签与原图逐ID配对，没有图会报错。`PROVENANCE.json:$.maples_available=false`，最终54条manifest只有IDRiD；无MAPLES训练配置、记录或运行日志，故是“部分标签资产存在、不可训练、未执行”。若下一轮取得兼容MESSIDOR原图，现有prep代码可将Bright/Red uncertainty栅格做逐像素valid mask（`prepare_external.py:104-113`）。

未发现FGADR图像、标签或manifest条目；源白名单仅包含IDRiD与MAPLES-DR（`tools\contract_common.py:57-58`），manifest工具也只提供idrid/maples两种dataset选择（`tools\build_external_manifest.py:465`）。因此FGADR既无可执行资产也无本轮执行证据。

## 6. 审计记录和已证实事项

- 当前合同通过：`reports\external_contract.md:3-10` / `external_contract.json` 的 `$.errors=[]`、`$.counts.records=54`；同报告的1170个阈值≤6近似哈希对是IDRiD内部候选，不是1170个真实重复。
- `reports\external_audit.md:3-16` 是09:22生成的旧快照，把54条 `notes` list 误判为schema错误；`prepare_external.py:141-144` 后续将备注join成字符串再写manifest，09:26的external_contract PASS。因此这是旧审计工件过期，不是当前训练时掩膜损坏的证据。
- 独立图像级比对 `reports\duplicates\duplicates.json` 有 `$.ddr_images_compared=757`、`$.external_images=54`、`$.excluded=[]`、`$.high_similarity_review=[]`；这只是有限图像相似性审计，不能证明patient-level去重，也未覆盖未知旋转/形变。`external_contract.json` 的 `ddr_images_compared=0`是合同审计未执行跨数据集比对，不与上述单独审计冲突。
- DDR代码 `trainer\data\ddr_dataset.py:6-15,105-114` 将融合标签值0..4变成互斥4通道并全图有效；`trainer\data\common.py:102-111` 说明逐类TIF不参与训练接口。外部数据 `external_dataset.py:179-200` 则保留独立多类mask并按valid_mode执行屏蔽。

### 已确认事实与待验证假设

**已确认**：IDRiD输入是54张 `best_segmentation` 用户处理图及其归一化多边形，不是实验包内的官方像素GT；48/6外部划分和U1/H1两段2400步预训练确实执行；MA在54图均有polygon；当前栅格坐标与类别映射在抽查3图逐像素复现；IDRiD全类full-valid导致无polygon类别成为全图负例；DDR与IDRiD编码语义不同；U1/H1 val方形拉伸与U0原生几何不同；MAPLES只有标签无配套照片，FGADR无可用资产。

**待验证而非已证实bug**：MA单实例过大、标注漏病灶、旋转增强导致几何错位、近似哈希对为同一病例、数据密度差异导致外部预训练变差、方形val拉伸足以解释某个分数差。由于无官方像素GT和原图对应SHA，这些都不能从现有聚合统计下结论。

## 7. 对下一方案的数据控制建议

先把监督域说清楚：若继续用当前IDRiD，按“用户处理图上的弱/派生多边形监督”记录，并补做分层人工边界抽检与每类覆盖审计；确有未知区域时要用 `valid_mode=file/none` 表示，而不是将未标注区域默认为背景。若要声称官方IDRiD监督，应先取得配套原始图、官方像素mask及可审计的坐标变换。

下一轮若要判别外部预训练，应固定同一模型/初始化、同一训练和验证变换，在DDR-only与IDRiD预训练间只改预训练数据；验证始终保留长宽比并使用同一整图/滑窗几何。把“改变训练数据/增强”与“改变推理缩放/TTA”分开成独立因素，至少报告同一节点的逐类像素TP/FP/FN。6张图片的外部dev不足以支持泛化结论；如不再有更大独立训练来源，至少把它仅作为节点/标签诊断，避免用它调很多方案。MAPLES需先配齐原始MESSIDOR图并使用不确定区域valid mask；继续关闭test。

本报告机器可读摘要见 `02_外部数据审查_Luna.json`。
