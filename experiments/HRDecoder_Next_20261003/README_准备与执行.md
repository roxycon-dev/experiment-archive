# 10月3日严格方案：准备交付与服务器执行

对应原方案：`D:\dataset\experiment\HRDecoder_严格复诊_20261003\01_严格诊断与下一轮主方案.md`。
本目录是新的实验工作区。历史结果、配置、部署锁与权重没有改写。准备已完成，服务器执行已于10月3日启动。实时状态及执行中的资源/工程调整见 [服务器执行记录](reports/server_execution.md)。下文保留最初准备交付记录；最终结论以完整val、锁定test和最终报告为准。

## 已完成的准备

- G：U0与U1的 native/等比例长边2048 × whole/1024滑窗stride768；U1旧方形滑窗参照；H1@9k普通forward/原eval_forward对照。whole OOM会记为不可评价，停止该项，不改小图。矩阵结束锁定画布与协议，U0高分辨率@1k只补一次锁定评估。
- M0/M1：严格用冠军原加载器还原D学生，导出FCN分类层前64通道；冻结参数和BN。完整复用U0；stride4零初始化语义adapter；M1新增真实RGB stride1/2跳连，投影末层为零，stem为逐像素通道归一化。输出四类独立自由logits。
- 共同几何：全局视图等比例长边1024；1024局部输入、halo128、中心768。中心全图覆盖一次；真实边界之外的监督屏蔽；窗口外归一化后补0。完整RGB与GT先同步翻转/90度旋转，全局网络真正重新看增强后的整图。
- 共同训练：固定样本/窗口/增强清单；micro4×2，必要时在正式训练前两臂共同改micro2×4；后两stage与解码器1e-5，新支路1e-4，AdamW/WD0.01，裁剪1，原FP32 BCE+Dice。warmup400，8000步共同预算；0/1k/2k/4k/8k完整76图val，单模与固定50%完整旧部署融合两种输出。恢复同时保存optimizer、scaler、RNG和数据游标；AMP跳步重放同一批。
- 止损：4k前不单臂停。两臂最佳都不超各自0步/完整旧部署起点且2k→4k无上升，才共同停；否则保持共同预算到8k。相同分数优先早节点，再优先单模。首组部署增益约0.5点才触发第二种子门槛；不加逐类权重搜索。
- 诊断：固定0.5严格大于的四类全局TP/FP/FN、逐图计数、病灶面积分桶、整颗漏检/边界FN、共同错误与新FP、PR/IoU阈值曲线；可额外运行三折折外常数校准、25%专家GT上界、配对图像bootstrap、按失败计数选图的人工错误图库。
- 条件分支：M1容量完全相同的低通RGB支路已可构造，但只在M1稳定胜出后投入机制消融。E的官方原图/独立像素GT/ignore资产审计入口已准备；既有用户polygon不会通过官方GT入口。
- 最终输出：val锁定脚本和一次可恢复的完整管线test脚本已准备；旧test不会参与G、M选型。既有227图仍注明探索性延续。

## 本机验收与剩余服务器验收

`reports/local_graph_preflight.json`：真实U0高分辨率@1k权重严格完整加载；原U0、M0、M1的0步最大输出误差为0；两次合成更新后新增stem获得非零梯度，解码器、后两stage与新增投影接通；冻结参数不变。全局输入为明确标记的合成冻结分支，此项没有代替真实D验收。

`reports/resume_acceptance.json`：使用真实U0和真实新训练器，在小型合成图像上比较中断恢复与直接训练；最终所有模型张量最大误差为0，数据游标相同，完成缓存不会重新推理。合成验收权重已随临时目录回收。

`tests/test_protocol.py`覆盖不规则图像与边界的唯一回填、原标签复原、非整数缩放对齐、逐项样本重放、前景确在监督中心、共同止损/继续、选点平分规则、严格阈值、病灶漏检与评估中断续跑。

服务器必须再验：真实D/B逐键加载及张量一致；导出特征与原FCN conv_seg之前的真实tensor一致；真实D+U0两臂0步一致；D无梯度/BN不变；1024真实窗口microbatch两次前后向；固定454/76划分与76张原GT SHA；显存、磁盘与环境。未通过时不会自动开正式训练。本机没有mmcv/mmengine，因此这部分明确待服务器完成。

## 资产与空间

九份必需权重见 `configs/assets.json`。D/B/E/E4/H4保持历史文件SHA；U0/U1/H1/U0@1k仅剥离不用的optimizer等状态，每个模型张量与源文件逐项相等。H1确认为9k，U1为12k，高分辨率U0为1k。新资产合计约2.07GiB，完整旧部署仍独立保留。

由历史76图尺寸计算，共401,111,888个像素，一份四通道FP32概率原始体积约5.98GiB。G需要保留所有诊断概率，13份实际缓存、14份作为空间估计余量。M保留全部节点指标/逐图计数，长期只保留每臂最佳节点的概率，清理范围限本目录`work/M_seed*/M*/val_u*/`新产生的npz，保留每节点报告和best/last权重。不会清理任何旧实验。

建议服务器提供24GB或更多显存、约120GiB可写空间，或安排G缓存及时镜像到本机。压缩为无损FP32，不预设压缩比。50GB数据盘若仍保留旧数据/权重，通常不能直接容纳全部缓存；实际资源会先实测，不以缩图或降低概率精度解决。

## 上传和启动

代码包：`D:\dataset\experiment\HRDecoder_Next_20261003_code.tar.gz`。
资产包：`D:\dataset\experiment\HRDecoder_Next_20261003_assets.tar.gz`。
若还是旧服务器，可复用SHA完全相同的D/B/E/E4/H4，只上传四份新的U0/U1/H1/U0@1k，不必重复传整套冠军。若是新服务器，两个包自包含代码和权重，DDR数据仍需固定划分。

先复用服务器能正常运行旧冠军的PyTorch/mmcv/mmengine环境；旧mmseg源码已带入。Python 3.8或更高版本；torch需具备AMP与meshgrid indexing接口。新服务器环境先探测后按真实版本配套，不盲目升级mmcv。不会用本机的合成分支替代缺失的HRNet依赖。

```bash
cd /root/autodl-tmp
tar -xzf HRDecoder_Next_20261003_code.tar.gz
tar -xzf HRDecoder_Next_20261003_assets.tar.gz
cd HRDecoder_Next_20261003
python configure.py --ddr-root /root/HRDecoder/data/DDR_split_613
python preflight.py --server
python run_campaign.py --stage G
```

G完成后读取`work/G/matrix.json`、`work/geometry_lock.json`、错误图库及校准证据再决定投入M；默认入口只跑G，防止来不及审阅就把所有训练排满。

```bash
python analyze_cache.py --cache work/G/current_deployment --baseline work/G/current_deployment --out work/G/current_calibration_OOF.json
# 如几何诊断确实指向新表征，再启动首组及有收益才触发的第二种子：
python run_campaign.py --stage M
```

若micro4×2在服务器验收OOM，应先共同改成2×4并重新验收，之后固定：

```bash
python configure.py --ddr-root /root/HRDecoder/data/DDR_split_613 --micro-batch 2 --grad-accum 4
python preflight.py --server
```

出现校准为主、已有权重足够、收益集中于少数图、主模型仍稳步上升或官方外部资产到位时，依据证据写入新的阶段决定，再自适应推进。不会机械重复历史baseline、延长U1或重开H1/A0。

`freeze_final.py --seed <所选seed> --arm M0|M1`仅从该臂val记录锁定最佳节点和单模/固定融合；`final_test.py`只接受这份不可变锁。没有赢家时保留现有44.7809管线，不为了方案完整再测test。外部E资产齐备后用`audit_external.py --manifest <官方资产清单> --out <审计报告>`验收，再单独制定最佳结构的DDR-only/外部迁移配对阶段。

本次准备未登录服务器、未启动付费计算、未创建自动监控或关机任务。服务器提供后，先完成真实环境与资源验收再估计时长。
