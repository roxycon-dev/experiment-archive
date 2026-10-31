# 本轮补读的委派文本：证据、提案缺陷与适用边界

本文件记录`delegation_split_20261016.csv`中分配给`remaining_materials_luna`的SHA排序前30项。逐项状态见`delegation_coverage_remaining.csv`：30项共593,652 bytes，29项完整全文读完；`HRDecoder_FusionAware_20260917/delegation/opsfix_response.txt`的所有现存内容读到原始文件EOF，但源文件本身只到未闭合代码行`stem = os.path.splitext(name)[`，所以缺失尾部无法阅读。该项状态为“现存全文已读、原件不完整”，绝不记成“完整原件已读”或“未读完的输出”。文件实测32,563 bytes，SHA-256 `600e56629d2da1bd268e09cef91f48712c4946b6962d1f743c81419eb7a8a3d1`，与清单一致。其余文件都逐份读全文；大JSON/TXT按字符范围完整分段，没有运行嵌入代码、测试或远程命令。没有从旧MD继承此组的正文阅读；historical agent按SHA确认这60份delegation与已扫描Markdown没有exact-content重复。

## 经既有运行结果交叉核验的事实

- **plus3不等于“只做了启动失败”。** `hrdecoder_plus3/delegation/report_prompt.txt`列出的D_0913_retry验证序列为10个val节点：best 41.615%@2,000 iter，20,000 iter末次40.476%；该提示明确current_test为空。`hrdecoder_plus3/results/D_0913_retry/D_0913_retry_summary.json`及状态记录由历史运行审阅者独立核对：正式阶段抵达20k、best在2k、末次40.48、test未运行。准确表述应为“一个较早启动尝试被persistent_workers拦住，修复后D_0913_retry完成20k正式训练；test未跑”。它不是v3成绩，也不是test增益证据。
- **smoke与正式训练需分开。** `hrdecoder_plus3/delegation/smoke_response.txt`源码包含3个SGD更新和一次val图像前向；配套既有smoke记录的成功只证明短时预检走通，不代表正式训练完成。它的代码只检查有非零student梯度、reference不变和输出形状/范围，未证明仅decoder/stage4参数更新。
- **已计划但尚无运行结果的方案边界。** `protocolfix_prompt.txt`/`protocolfix_response.txt`仍属于旧FusionAware实现提案，提出成对A/B训练、1024 crop从2048图像、train-only reference概率缓存、val门槛和测试前锁定。不得把提示中的目标值、预算或合成test断言写成结果。

## 已读提案中的实际代码/测试缺陷

下列问题说明代码草案或其测试不可靠，不能拿来解释真实实验崩塌。对应文件是委派回复/拟议实现，不是已运行实现。

- **核心损失/标签：** `HRDecoder_FusionAware_20260917/delegation/core_response.txt`的Dice shape assert将3维标签与完整4维概率形状比较；其one-hot直接以原始label为类别索引，和DDR 0背景、1..4病灶、255忽略及4个前景通道不匹配。其reference-mismatch测试使用`fill_(tensor)`。`corefix_response.txt`后续稿尝试将1..4映射至0..3并补齐验证，但本组看到的是书写出的源文和合成测试，未执行。
- **运行时缓存/证据：** `runtime_response.txt`第一稿的cache `complete()`仅检查标记文件；identity probe对同一小切片计算后只`pass`；strict载入采用`strict=False`；BN affine被冻结；first-update evidence未验证必要非零梯度；eval缺少严格输出形状/值域校验。`runtimefix_response.txt`后稿强化了这些点，但`ReferenceCache.complete()`实际只核对文件路径/SHA/扩展名；每项数组shape、dtype、有限值及概率范围在`get()`时懒检查，和它的docstring“complete验证每项shape/range”不一致。`audit_split`还检查test标签目录中是否存在PNG文件，应确保任何实际隔离政策明确说明允许的元数据检查边界。
- **评估协议：** `protocolfix_response.txt`确实由`build_student()`写入`settings['_cfg']`；此前我曾误报缺少该赋值，已撤回。仍可确认`evaluate_candidates(..., dump_prefix=...)`不使用`dump_prefix`；测试`test_rng_restore`最后将长度3的`first3`与长度1的列表作相等比较，按字面必失败。提案包含final-test读取GT的路径，但没有因此运行任何测试集评估。
- **guard：** `hrdecoder_plus3/delegation/operations_fix_response.txt`请求和提议清理子进程组，但现存稿只在leader仍运行时清理；正常退出后finally只在leader仍运行时再清理，不能覆盖leader退出而同组子进程留存的场景。该测试稿还用`setsid`创建独立进程组，却期待原PGID能将其杀死，测试假设不成立。shutdown返回非零时只在write状态文件后改动内存对象，故该状态不会被持久化。
- **ops/split审计：** `opsfix_response.txt`现存代码的`choose_checkpoint`只在分数严格更高时替换当前项，遇到反序同分输入会保留较晚step，与测试声称“任何顺序取最早”矛盾。`asset_inventory.audit_split`对train、val、test一律要求有label PNG，和项目test无标签布局矛盾，会把无test标签的合法输入报为BLOCKED。`extract_metrics`取JSON遍历遇到的首个mIoU和首个per_class，若一个文件含多个结果节点可能把不同节点拼成比较项。此回复文件本身在split审计测试中途截断，因此只对EOF前已有文本作上述结论。
- **空间预检：** `HRDecoder_FusionAware_20260917/delegation/storage_response.txt`遍历data_root下图像与mask文件并读尺寸头；因此“从不读labels”的说明过强。跨文件系统分支的ok布尔只比较裸cache和checkpoint字节数，没有把其报告的20%/2GiB headroom用于通过条件。测试是文本中的拟议合成测试，未运行。
- **裁块测试：** `hrdecoder_plus3/delegation/pipeline_tests_response.txt`用AST抽取生产`ForegroundMixedCrop`类，但自行注入了pad/crop/foreground_bbox helpers；因此覆盖类调用路径的几何一致性，不等价于验证真正helper或mmcv pipeline的端到端行为。测试未运行。

## 对当前架构设计的影响

这些材料没有提供新的胜出单一W48结果，没有提供v3性能结果，也没有推翻现有结构选择。它们提供的是实现约束：必须显式做DDR标签到4通道的正确映射；缓存须完整绑定配置、reference权重和训练图像身份；严格初始化不能先写模型再校验；首次更新/BN统计/可训练参数要有可复查证据；val评估需严格检查输出形状、值域和ignore处理；所有选型锁在val上，test不参与训练或checkpoint选择。以上属于实现审查结论，不是对过去真实崩塌成因的因果解释。

## 阅读方式声明

这30项的`status`、SHA和字节数在CSV中逐文件记录。完整文本阅读的意思是读取并理解原文件现存正文；对不完整的opsfix文件仅能读完已存部分。所有提案代码及测试均未执行；所有引用的历史指标来自文本或既有运行产物核对，并且本轮没有读取任何模型/图像/预测数组作为性能评估，也没有训练、SGD或新的test像素评估。

## 补充数据来源身份差异

`HRDecoder_Q_20261003/assets/external_derived/IDRiD_best_segmentation_pixel_masks/README.md`（SHA `a463e528cc986623e2287cb0ebf5da44fb79582288847605122a90f2d7826bf46`）已全文阅读：其54张训练图、27张测试图标签由用户多边形栅格化到方图坐标；文档不称其为官方像素级GT，也没有确认阴性/未标注图是官方负样本。它与其它材料中关于官方GT多边形转换的描述是不同来源，不应合并。v3使用接受的454/76/227融合标签划分，没有引用该派生目录；这项来源差异仅用于防止证据身份混淆，不新增设计或实验。
