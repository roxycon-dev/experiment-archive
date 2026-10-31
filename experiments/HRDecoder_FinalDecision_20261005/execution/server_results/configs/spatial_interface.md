# 空间交互与随机流接口

2026-10-05。随主方案冻结的实现约定。此文件定义待实现网络，不代表已经通过单元测试或GPU验收；不是ViT-Adapter原文逐项复现。

## 1. 输入与空间金字塔

输入就是传给旧F12的同一RGB张量：`B×3×1024×1024`，使用ImageNet mean/std归一化，padding为归一化坐标零。不得额外CLAHE、BGR换序或采用单独裁剪。监督仍取原几何中心768。

SPM固定五个`Conv3×3(stride2,padding1)→GN(8 groups)→GELU`块，输出通道依次32、64、128、256、256；卷积bias=False。输出第2–5块得到下表。暂不添加其它残差块、dilation、FFN或注意力类型扫描。

|张量|形状，1024输入|用途|
|---|---|---|
|c1|B×64×256×256|stride4，最终投影到DPT细层|
|c2|B×128×128×128|stride8空间token来源|
|c3|B×256×64×64|stride16空间token来源|
|c4|B×256×32×32|stride32空间token来源|
|spatial tokens c|B×21504×768|c2/c3/c4各用1×1投影到768，按行优先flatten并顺序拼接|
|ViT patch tokens x|B×4096×768|64×64 patch网格；CLS/register不在此片段中|
|完整ViT序列|B×4101×768|1 CLS+4 register+4096 patch；以实际已严格载入官方模型再assert|

投影后给c2/c3/c4分别加可学习level embedding，初始化normal(std=0.02)。不另加ViT绝对位置编码。不改变旧CLS/register排序或RoPE。

## 2. 四段交互顺序

四段独立参数，不共享attention权重。每段：

1. `delta_x = Injector(LN_patch(x), LN_spatial(c))`；仅对patch片段做`x += delta_x`，拼回完整序列。
2. 运行原官方ViT的三个blocks，分别为0–2、3–5、6–8、9–11。保留官方RoPE、register和旧随机层配置。
3. 保存该段输出，按旧F12方式使用共享backbone final norm取得DPT所需patch特征。
4. `c += Extractor(LN_spatial(c), LN_patch(x_after_blocks))`。Extractor查询为三个空间尺度的token，value为当前64×64 patch网格。

这里的attention pre-LayerNorm独立属于新模块，不替换旧backbone norm。Injector输出`B×4096×768`；Extractor输出`B×21504×768`。两者均12heads，每head64维，每value层采样4点。Injector有3个value层，Extractor只有1个。attention dropout为0，不增加额外随机深度或FFN。

第四段后把c按原长度切回128²、64²、32²三层，得到更新后的stride8/16/32特征；c1仍取SPM输出。四层分别用1×1投影到128通道，加入旧DPT四级投影后的、同shape的特征（stride4/8/16/32），之后完全沿用旧decoder。DPT的ViT四级输入仍来自blocks2/5/8/11，不能换成四个同一最终层输出。

## 3. 可变形注意力的坐标约定

- 所有二维坐标按`(x,y)`；flatten按`y*W+x`。`spatial_shapes`按`(H,W)`，并显式写每层start index。
- 查询参考点是查询自身网格中心：`((j+0.5)/Wq,(i+0.5)/Hq)`，范围(0,1)。Injector的查询网格为64×64；Extractor的查询按其所属128/64/32网格分别生成。
- 每个value层的offset以该层像素为单位，sampling location=`reference+(dx/W_level,dy/H_level)`。Extractor的value层固定64×64。
- PyTorch参考实现使用`grid=2*location-1`，bilinear `grid_sample`、`align_corners=False`、`padding_mode=zeros`；边界外采样为0。attention权重对每query/head的全部`levels×points`做softmax。
- Value线性投影后按head切分；采样加权拼接后再output projection。Offset linear weight初始化0、bias采用原MSDeformAttn的各head方向/采样半径初始化；attention-weight linear weight/bias初始化0。优化kernel必须与这份参考前向/反向一致，不能使用坐标含义不同的实现。
- 建议验收误差：小张量FP32前向`atol=1e-5,rtol=1e-4`，输入/value/offset梯度`atol=1e-4,rtol=1e-3`；同时做有意偏移一个像素的合成位置测试。误差不通过就定位实现，不能先放宽容差继续。

## 4. 初始化、域头与梯度

Injector最终output projection weight/bias置0；四个spatial-to-DPT投影weight/bias置0。SPM、其输入投影、Extractor和内部投影按正常初始化，不再加零门控。所有旧DDR参数从父F12严格复用。

D的TJDR decoder/RGB独立复制父DDR decoder；SD同样复制，并拥有自己一组新增spatial-to-DPT投影。共享的只有ViT和SPM/四段交互；源域head是输出映射隔离，不是backbone梯度隔离。

对C/S/D/SD的DDR路径都做零步恒等检查。前两个训练更新仅用于验收复制模型，正式训练回到同一u0和固定随机状态，不把验收更新混入正式日程。第0次backward检查最终零投影梯度非零；第1次更新后检查SPM/Injector/Extractor上游梯度有传播。旧参数及新增参数变化必须能按名称列出；clip前检查，避免极小数被显示为0。

## 5. 随机流隔离与恢复

窗口/增强预生成，DDR stream在同种子四臂完全一致。TJDR独立stream。新模块初始化用单独RNG上下文，不能改变旧权重或后续DDR抽样。

每个训练microbatch前进入保存/恢复CPU与当前CUDA RNG的上下文（如`torch.random.fork_rng`），内部用以下稳定种子设置torch CPU/CUDA RNG：

`int.from_bytes(SHA256("campaign20261005|seed|source|update|microindex")[:8], "little") mod 2^63`

字符串中字段以实际整数十进制、source为固定`DDR`或`TJDR`替换；update从1开始，DDR microindex0–3、TJDR0–1。不包含臂名，因此同种子、同更新、同DDR microbatch的原骨干随机抽样配对；源域额外前向不会推进下一个DDR前向的RNG。新增attention/SPM不使用dropout，避免新增层消耗旧骨干的随机数序列。父F12原随机层配置须导出记录，不静默改成另一配置。

若用activation checkpointing，必须保留重算的RNG状态。恢复记录global CPU/CUDA、NumPy、Python RNG和stream cursor；即使前向种子可由update重建，也不能漏掉数据准备/保存等路径的随机状态。

验收：小样本连续跑2更新与中断后恢复第2更新，比较下一窗口、增强、学习率、EMA及模型变化；同时验证C与D在关闭源域loss的专用验收模式下DDR-only前向/梯度一致。该验收模式不能用于正式D训练或成为额外候选。
