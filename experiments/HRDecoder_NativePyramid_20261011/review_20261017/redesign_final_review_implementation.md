# 05 双尺度方案实现可行性终审（静态）

## 结论

05 的数据流与共享权重意图表达清楚，设计内部没有发现确定的拓扑矛盾。其可执行性仍有一个未解决的资源门槛：没有双分辨率图、三套 head BN bank、两路保留反向图及激活重算的实测内存/吞吐证据。不要把它标成已验证可训练的配方。这里仅做静态审阅；没有执行模型、训练、forward/backward 或新像素测试。

05 的 FCNHead 配置已显式列出 dropout_ratio=-1、concat_input=False、dilation=1、padding=3，且写明无 dropout；这与 B0 一致。此前关于草稿可能漏写 dropout 的担心已由当前05稿消除。

## 路径与张量检查

对 batch=2 的2048×2048画布，G 路径 resize 到1024，W48输出 F1–F4 分别为48×256²、96×128²、192×64²、384×32²；插值到256网格后拼接720通道，再经同一FCNHead输出4×256²。N 路径对原2048图编码，输出 F1–F4 分别为48×512²、96×256²、192×128²、384×64²，对齐到512网格后也是720通道，再由相同head输出4×512²。两路各自logits先用align_corners=False上采样到2048，再做Z=(G+N)/2，最后对完整标签算Dice。数据流没有混合logits/概率，也没有在不匹配网格相加。

G路径的M4辅损失继续使用原算法：由G的720×256²融合特征抽取四个随机特征crop，缩放到head输入尺寸后共用head，GT从2048标签取对应原图crop；它不额外编码四张RGB crop。G主head与aux head必须选用不同running-stat bank，避免aux crop分布改写G整图统计。

## 必须保持的BN与权重语义

BN bank 应只复制运行状态，不复制模块。主干每个BN保留两套独立 running_mean、running_var、num_batches_tracked（G/N）；FCNHead每个BN保留三套（G/N/aux）；每个BN只有一份共享gamma/beta，所有卷积和分类层也只有一份。若实现用整模块复制来分开统计，会复制可训练affine/卷积并破坏“共享W48”。初始化器、预训练加载与优化器参数组必须最终指向同一套Parameter对象。

建议把“优化一次”作为实现断言：对 trainable Parameter 对象去重后构造一次SGD；检查优化器参数ID无重复；一次更新只执行一次optimizer.step。G、N全图主loss和M4 aux梯度在同一反传/累积窗口内相加后，经过一次裁剪（如配方规定）与一次step。不能对G/N分别各step一次，否则就不是每个样本共享一次训练更新。

BN的每个真实主路径前向应覆盖同一步两张原图。单卡物理batch=2或每域两卡各1图且确实使用SyncBN可满足；普通micro_batch=1逐图前向再梯度累积不会产生两图batch统计。05已把这个限制写明。原v3进度记录micro_batch=1，并不构成新设计batch=2 BN已可行的证据。

激活检查点重算有可预见的状态风险：PyTorch checkpoint重算可能再次执行训练态BN并推进EMA buffer/num_batches_tracked。重算函数应捕获具体G/N/aux bank，而非从后来被其他分支改变的全局开关取域；重算时仍用当前batch统计完成归一化，但需抑制第二次写running buffer及计数。先后两路前向不能交叉改写银行。验收应检查真实前向每次只增加目标bank计数一次，其他bank不变，并比较不重算/重算的输出、loss、梯度数值；当前还没有该验证。

## M4辅助归约语义

原HRDecoder把M4个crop沿batch维拼接后调用一次head与Dice loss。每个样本、每类分别算sigmoid平方分母Dice，再对类、crop实例按loss reduction取均值，最后整体乘0.1。05写明“四个crop贡献取原均值”，方向正确；实现必须避免将4个crop的loss相加而变成4倍辅助权重，也不能额外把已取均值再除4。主full-map Dice与aux Dice分别做各自归约，再以1.0和0.1相加。全图与crop损失形状/标签ignore语义仍需按05的255显式mask方案落实，不能直接照搬旧 BinaryLoss 对ignore的处理。

## 配置、成本与尚未验证项

FCNHead已固定为 in_channels=720、channels=64、num_convs=1、kernel_size=7、compress=True、concat_input=False、dropout_ratio=-1、num_classes=4；BN/ReLU、dilation=1、padding=3，输出分类层有bias。以上与B0相符。不得因使用mmseg BaseDecodeHead而依赖其dropout_ratio默认值0.1；必须把05中的-1落到实际构建配置。W48预训练加载之后再创建BN banks，优化器最后仅创建一次；严格核验预训练键集合，不允许 strict=False 静默漏载主干张量。Head权重按同一初始化规则共享，不是为G/N/aux各初始化一份卷积。

每个训练图像需要一次1024² G编码和一次2048² N编码，按输入像素量为单路B0的5倍、d303单次2048²视图的1.25倍。M4仍是特征裁块，不额外贡献RGB encoder views。40k×batch2对应80k个图像出现和160k个尺度视图；优化步数相同不代表计算成本相同。

历史资源证据只测过单路W48、2048、batch2 BF16 forward/backward，峰值allocated约19.89 GB、reserved约21.84 GB（总量25.28 GB），且没有创建优化器；双路同时保留激活明显超出该证据覆盖范围。当前运行环境只可确认4090型号及microbatch=1；系统查询显示8张卡，但没有本任务可用显存总量/余量或容器可用卡数的可靠结果。不能将单路历史内存数据外推为05一定可行或一定OOM。

正式实现前需以纯静态或专门小规模实现门验证：参数对象唯一性与一次step；预训练逐键核验；每个bank的更新隔离和计数；物理BN batch；重算前后buffer/输出/梯度一致；各路径真实tensor shape与crop mask；主/aux损失独立归约；双路训练峰值显存与每步耗时。此清单是未来验收条件，不是当前已执行测试。

## 参考实现与历史先例

原始结构、M4 reduction及训练参数见同目录 [redesign_reference_implementation.md](redesign_reference_implementation.md)。本地HRDecoder.py的hr_forward_train将4 crop拼为一个batch；HRDecoder.forward_train从同一低分辨率整图特征构造global与滑窗局部分支；该模型没有05的原RGB N encoder。历史审计中没有发现相同W48共享encoder的双尺度HRDecoder先例。HRDA提供另一任务下的共享encoder多尺度训练参考，但不能作为本数据集效果或资源保证。

