# HRDecoder_Q 实现与结果审查（2026-10-04）

## 审查范围与结论

只读检查了新实验报告、原方案与验收规范、`qlib/`、`configs/`、训练及预检代码、固定采样清单、运行日志和服务器回收的验证/数据验收记录。没有 SSH、训练、改源码或读取大权重。服务器运行所用源码身份与本机副本的固定 SHA 相符。

**核心结论：核心模型与训练执行没有发现能解释低单模分数的已确认实现错误；但 IDRiD 的 SE valid 语义有一项实质性负监督漏用。** DINO 层索引、解冻切点、优化器参数组、有效更新和调度、loss mask、训练/推理裁块几何均与锁定方案一致；预检还实际验证了未解冻层不变、解冻层及四类 head 有梯度并发生参数更新。另一方面，accepted manifest 将 28 张无 SE polygon 的 IDRiD 图像都标成 unknown，整类 valid=0。依据最新用户确认：标注来自完整官方像素 GT 的无损转换，这 28 个空类应是已审阴性，旧转换守卫把它们从 loss 中整体屏蔽了。应修正新的数据 manifest 并做单因素配对复核；这是数据有效性状态漏用，不是 DINO/optimizer/geometry bug，也尚不能单独证明它导致了 FP 增量。

已确认的主要研究边界是：Q_MIX 与 Q_DDR 的 24k 对照同时替换了训练来源和 25% 的 DDR 窗口；它测的是等更新预算下的来源分配，而非在相同 DDR 曝光上附加外部数据。Q_MIX 的 SE 验证差值在两种子均约 −5.6 点，且 FP 大增；这与丢掉 SE 已知阴性监督相符，但来源/曝光也同时改变，仍不是单独的因果证明。最新用户说明确认 IDRiD 多边形来自官方像素标签的无损转换、效果基本趋同；因此不应把 polygon 格式本身说成粗糙或不真实监督。

## 按实现问题核对

### DINOv3 层和预训练身份

规范要求取 ViT-B/16 的 `[2,5,8,11]`，这是 Python 从 0 开始的第 3、6、9、12 个 block。`qlib/model.py` 对 12 blocks、768 维、patch16 作了结构断言，然后调用官方 `get_intermediate_layers(x, n=[2,5,8,11], reshape=True, norm=True)`；官方 vendored 实现按 block 索引收集输出、用最终 `norm` 归一化、去掉 class/storage tokens 后重排 patch 网格。输出断言要求每层为 `[B,768,64,64]`（1024 输入）。索引与去 token 的方式正确。

加载路径先核权重 SHA，再以 `strict=True` 装载。四个正式运行的初始化记录均为 `strict_pretrained`、missing/unexpected keys 为空；DINO 官方代码 commit 为 `6876159a11b4df116f30f667f8c9888617df0751`。这支持“预训练实际装载”的判断，不支持怀疑随机初始化。

### 冻结、解冻、AdamW 与学习率

模型创建时冻结整个 backbone。训练循环每次以已完成更新数调用 `set_update(completed)`：完成数 0–1999 时更新 1–2000 的 backbone 冻结；完成 2000 后，第 2001 次更新前启用 `blocks[8:]`（正好第 9–12 层）和最终 `backbone.norm`。patch embed 与前八个 block 始终冻结。`get_intermediate_layers(norm=True)` 在所有四个特征层共用这个最终 norm，因此末 norm 的梯度能影响全部 decoder 输入。

优化器在解冻前创建，参数组已包含末四 block 和最终 norm；它们在前 2000 步的 LR 为 0，解冻时没有重建模型或优化器状态。head/RGB LR 3e-4，末四层 LR 5e-6；weight decay 0.01，bias 与一维 norm 参数不衰减；累积结束后全局裁剪 norm 1.0，再每次有效更新执行一次 optimizer step 和 scheduler step。调度预先锁为 24k：head 前 200 步 warmup，backbone 更新 2001–2500 warmup，二者余弦降至各自初始 LR 的 1%。

这不只是源码推断：GPU 预检明确执行了 100 个末四层真实反传更新，并断言 head 四通道与每个末四 block 均有梯度、head/末四层参数变化、早期层保持不变；预检状态为 PASS。正式日志在更新 2000 之后记录的 backbone 下一步 LR 为 `1e-8`，更新 2020 为 `2.1e-7`，终点达到 `5e-8`；head 同步按计划退火。这与更新 2001 开始解冻的日程吻合。

### 有效 batch、loss 与采样

正式配方是 micro-batch 2、累积 4、BF16，即每个有效更新处理 8 个窗口。每个 micro 的 loss 在 logits 转为 FP32 后计算：逐类 masked BCE 和 masked Dice 各占 0.5；invalid 像素先被中和并从分子/分母排除，完全无 valid 像素的类在该 micro 中不参与类平均。每 micro 的 loss 除以 4 累积，更新边界只裁剪/step/schedule 一次。

Dice 在每个 2 样本 micro 上计算，**不是**把 8 个窗口合并后算一次 Dice。规范已明确记录了这种差异；这是实验目标的一部分与潜在目标权重变化，不是代码偏离。每更新 4 个均匀窗、4 个按 EX/HE/SE/MA 各一个的前景窗，GT 只用于训练样本抽取。它会提高小病灶/阳性区域在训练窗口中的比例，可能改变输出校准与固定阈值 0.5 下的 precision/recall；当前没有校准曲线证明存在错误，也没有理由把固定阈值结果改写。任何阈值研究都应在新方案中预注册，并与旧 test 选择隔离。

### 训练与推理几何

训练和验证都使用 1024×1024 窗口，中央 768×768 为监督/写回区，四边 halo 为 128。RGB 在归一化后以零 padding；模型先把 logits 插值回完整 1024，再裁中央 768。推理以 stride 768 铺满原图，裁取末块有效高度/宽度，之后 sigmoid 一次并写回原坐标；没有二次缩放或重复 sigmoid。旋转/翻转同时变换图像、target、valid，尺寸变化会重算窗口边界。审查代码未发现训练标签和验证输出错位、缩放次序颠倒或 halo 裁切不一致的确认错误。

## 外部数据的逐类有效性与 SE 结果

接受清单共 54 张 IDRiD train、138 张 MAPLES train；外部测试图未参与训练。`External.load` 对 `pixel_mask` 类加载与 RGB 相同坐标/尺寸的独立通道；`unknown` 类整张 `valid=0`，不会被伪装成全零阴性。采样正样本目录从 `target & valid` 建立，因此 unknown 的 SE 通道也不能作为 SE 前景正样本。MAPLES 的 RedUncertains 对 HE/MA、BrightUncertains 对 EX/SE 置 invalid。

最终接受 manifest 的逐类状态为：

|来源|EX 已知/未知|HE 已知/未知|SE 已知/未知|MA 已知/未知|
|---|---:|---:|---:|---:|
|IDRiD 54 张|54/0|53/1|26/28|54/0|
|MAPLES 138 张|138/0|138/0|138/0|138/0|

这里“已知”只表示当前 accepted manifest 给该类分配了 mask/valid；MAPLES valid 仍会排除 uncertain 像素。IDRiD 的 SE 仅 26 张有类级有效监督，且这 26 张均有正像素；另 28 张 SE 都没有 polygon，derived manifest 记为 0 positive pixels，`official_completeness_verified=False`，accepted manifest 把它们全标为 `unknown`。转换记录显示 54 张 train 的每个源标签文件都被读取并导出四类 mask；结合最新用户确认“这些 polygons 是官方像素 GT 的无损转换”，这 28 个空 SE mask 对应已审阴性。当前 valid 策略于是把 28 张已知阴性整体丢出了监督：`External.load` 对 unknown 整个通道设 `valid=0`，整张图的 SE 负像素都不贡献 loss。此处是旧守卫造成的 valid 语义/清单漏用，不是 polygon 栅格精度或 loss 计算 bug。新清单应把这 28 条 SE 从 `unknown` 改为 `confirmed_negative`，保留其他尚未证完整的 unknown；`External.load` 已支持该状态：目标全零、valid 全一，无需改模型代码。

影响规模不仅是 28 个图像标签。accepted manifest 对这 28 张累计记了 342,108,024 个 SE `ignore_pixels`（原图像素数，不等同训练实际见到的像素数）。实际 Q_MIX 采样流中，seed42 有 10,761 个 IDRiD 窗口来自这 28 张，seed2024 有 10,762 个；每臂一共 24,000 个 IDRiD 窗口，约 45% 的 IDRiD 窗口因此没有 SE loss。未知样本没有进入 SE 前景采样池，但会进入均匀窗口以及 EX/HE/MA 前景窗口。

按最新用户澄清，IDRiD polygon 是官方像素标注无损转换、效果基本趋同；不把格式本身列为 SE 负迁移原因。旧转换脚本对每类都写 `official_completeness_verified=False`，并把无 polygon 写成“不能证明阴性”；这正是 28 个 SE 类被接受为 unknown 的来源。脚本采用归一化坐标乘当前存储图像宽高、`rint`、裁边和 `cv2.fillPoly`；旧收据只自动核验 mask PNG 编解码相同与源 TXT SHA 未变，没有独立官方像素图逐像素比较。该证据边界不反驳用户的无损来源确认，但说明后续应把完整性确认同步进 manifest，而不是继续沿用“无 polygon 即 unknown”的旧守卫。数据加载会逐图检查 mask/RGB 尺寸；20 张叠图审查也通过存储坐标对齐。

## 结果应怎样归因

24k、相同种子的 DDR/MIX 单模验证比较：seed42 为 39.0123% vs 37.5168%（MIX−DDR −1.4955 点），seed2024 为 37.8438% vs 36.6333%（−1.2105 点），均值 −1.3530 点。逐类验证差值：

|种子|EX|HE|SE|MA|
|---|---:|---:|---:|---:|
|42|+0.1144|−0.9621|−5.6270|+0.4926|
|2024|−1.4763|+1.9497|−5.6089|+0.2936|

SE 错误组成显示 FP 增量更大：seed42 的 DDR→MIX TP/FP/FN 为 `140240/75845/61164` → `138887/107538/62517`，即 FP +31,693、TP −1,353、FN +1,353；seed2024 为 `146626/87884/54778` → `140682/110694/60722`，即 FP +22,810、TP −5,944、FN +5,944。固定阈值下 SE precision 从 64.90%→56.36% 和 62.52%→55.96%，recall 从 69.63%→68.96% 和 72.80%→69.85%。FP 主导的变化与已知 IDRiD SE 阴性被全部 ignore、负监督不足相符；它仍不是因果证明，因为 MIX 还少了 48k DDR 窗口，且包含 MAPLES 与来源/时间顺序变化。

因此负面信号集中且重复在 SE，但 EX/HE 方向并不一致，MA 小幅正向。24k 的 DDR 臂曝光 192k DDR 窗口；MIX 是 144k DDR +24k IDRiD +24k MAPLES。前 12k 更新混合、后 12k 更新全 DDR。因此它同时改变来源、DDR 样本数及前半训练的域顺序。结果足以说“这个冻结的来源替换配方没有提升 DDR val，SE 尤其退步”，不足以说“外部监督必然有害”或“IDRiD 导致了 SE 退步”。

报告最终 test 的 45.6993% 是 `P0 + Q_DDR@16k` 固定融合的两种子均值（相对 P0 +0.9184 点），并非 Q 单模或 MIX 的 test 成绩。MIX/DDR 的 24k 比较是验证集对照，没有对它们做独立 test 比较。47.2115% 目标差 1.5122 点。不得把融合收益归为 DINO 单模或单个解码部件贡献。

## GPU 数字是否只测了冻结阶段

不是。`preflight_gpu.py` 先单独记录了 10 个冻结更新；之后设置完成数 2000、启用末四层，并在清零 peak 统计后实际做 100 次末四层反传更新。写入的主性能数值是这 100 次：allocated 3,816,224,256 bytes（3.5541 GiB），reserved 4,200,595,456 bytes（3.9121 GiB），每有效更新均值 0.460821 秒、P90 0.625807 秒、数据等待 0.101728 秒。全量 76 张 val 预检用时 207.50 秒。报告中的 3.55GiB/0.46 秒是未冻结、micro2×accum4、BF16 的实测，不是冻结头阶段测得。正式训练日志也覆盖到 24k 更新；整轮 14.56 小时包含验证、保存和恢复调用，不能解释为纯 GPU kernel 时间或账单。

## 下一方案建议

优先做数据语义修复，不先改源码、损失或阈值。将 54 张 IDRiD train 的来源 TXT 清单、四类转换状态和 SHA 锁定；最新用户已确认数据来自官方像素标注的无损转换，转换代码也成功读取全部 54 份训练标签。把无 SE polygon 的 28 条在新 accepted manifest 标成 `confirmed_negative`，重新生成数据身份 SHA，保留旧 Q_MIX 证据不变。随后只做一个 `Q_MIX_valid_negative` 配对复核：采样流、模型、loss、微批、更新数、LR、几何和其他数据逐项与原两臂相同，唯一变化是这 28 张 SE 通道由全 ignore 改为全图有效负类；同样跑 seed42/2024，只用 val 比较 SE FP/precision 与 recall及四类 mIoU，不碰 test。因为空 SE 图不进入前景正样本池，这项改动不会改变采样流；它能直接检验漏掉已知阴性是否解释 SE 假阳性增加。若负类恢复令 FP下降且 recall可接受，结论是旧 valid 守卫在原 MIX 下过度保守；若不变/恶化，再单独设计来源替换或 DDR 曝光实验。DDR val 有历史使用，所以仍属探索性验证。

若目标是继续追 test +1.51 点而非解释 MIX，可先要求新架构给出在双种子 val 上的单模净增益，再为一个预注册候选留独立确认集；当前结果不支持把再加一个 MIX 种子或对旧 test 做阈值搜索当成确认。未来若要谈“增加外部数据的收益”，应另用相同更新计划并明确对照估计的是外部窗口替代 DDR，或外部监督额外追加；二者不能混为一个因果问题。

## 关键源文件与证据位置

- 原方案/规范：`D:\dataset\experiment\HRDecoder_Breakthrough_Plan_20261003\01_成熟突破方案.md`（架构、采样、训练和判据）；`...\02_执行与验收规范.md`（层索引、mask、优化、预检和报告要求）。
- 最新完整报告：`D:\dataset\experiment\HRDecoder_Q_20261003\reports\HRDecoder_原方案完整实验报告_20261004.md`。
- 锁定实现：`qlib/model.py:31-76`；`qlib/optim.py:6-57`；`train_run.py:98-147`；`qlib/loss.py:5-23`；`qlib/geometry.py:59-88`；`qlib/evaluate.py:39-55`；`qlib/data.py:56-98,101-124,127-207`；`configs/protocol.json:14-19`。
- 预检测速/梯度断言：`preflight_gpu.py:123-167`；服务器 `server_results/reports/gpu_preflight.json`。训练日志：`server_results/runs/{Q_DDR,Q_MIX}_seed{42,2024}/train.jsonl`。逐样本来源标签与窗口计数：`assets/external_derived/IDRiD_best_segmentation_pixel_masks/manifest.json`、`server_results/reports/external_acceptance/manifest.server.accepted.json`、`server_results/reports/stream_Q_MIX_seed{42,2024}.jsonl`。
- 同终点类级比较：`reports/supervision_comparison_common24k.json`；完整节点：`reports/paired_validation_all_nodes_24k.json`。
- 数据类有效性：`server_results/reports/external_acceptance/manifest.server.accepted.json`。IDRiD 转换记录与新用户澄清应一起阅读：`reports/convert_user_idrid_polygons.py`、`reports/idrid_polygon_conversion_receipt.json`、`reports/external_source_user_override.json`。
- 源身份：`reports/source_identity.json` 与 `server_results/reports/source_identity.json` 聚合 SHA 相同（`d0fc4e0478835131cf3b731302174395d6827260bdae78bfa276b8e9851c9a3f`）；训练/模型/优化器/loss/data/geometry/evaluate/protocol 的本机 SHA 均匹配服务器 pin。
