# NativePyramid-SP v3 实际执行源码与协议一致性审计

审计范围只覆盖本轮实际运行身份、模型拓扑、损失与更新协议、初始化权重、数据几何及验证评估。按 D:\dataset\AGENTS.md 操作。未运行模型、训练、forward/backward 或像素测试，也未下载或加载权重。经用户明确授权，本次仅对目标远端运行目录做了 SSH 只读检查并流式回收训练日志字段投影；未修改远端文件或环境。连接端点 connect.nmb2.seetacloud.com:27443 在本地 known_hosts 中原无条目；本次首次观测的 ED25519 host-key 指纹为 SHA256:yswdgtshuc+hy/m0jR576i7Nu0Es7lZ71jIrFKXnkE0，未遇到已知指纹冲突。

## 实际运行身份与结果

本轮实际运行是 run_seed42_launch02，source SHA256 为 d3032fc8b20f30a058adbcf3e6bc5a3485b83adafe9b7cdce0682a2d55d7f627，data identity 为 7c53b86bcb27ccd121042f667cc58e39b98716a8c477bfd1f1657fa126575719，ImageNet W48 SHA256 为 d2186c55782308392ae14cf179ac4840143b09abdbba81aa3f9274c795ebc960。completion_report.json 第3行、preflight.json 第3–4行、progress.json 第12–14行和 endpoint_lock.json 第3、6行相互印证；progress.json 第2–5行记录正式完成并持久化40000步，第45–46行记录micro_batch=1及8个worker。

版本不是按目录名称推断的。我将 preflight.json 中全部15个 source_files SHA256 与项目根目录文件逐个比对，15项全部吻合。train.py 第30–33行定义这15个源码身份文件，第75–77行由它们计算source SHA；train.py 第385–400行要求服务器预检的protocol v3、source SHA、完整文件清单及逐项测试门全部一致。根目录这份源码就是实际运行的d303版本。completion报告记录的主结果是固定40000步 raw 权重，val mIoU 39.425551%，比45.321831%门槛低5.896280个百分点；test_permitted=false、test_accessed=false（completion_report.json 第4–6、14、30、35–36行）。

performance04_candidate当前delivery_manifest使用另一source SHA 14e47c3749db5dd28c2938ddbb227aad66b07c40b4fb9d049fdb0564b91c95d0；其model.py、train.py、evaluate.py、verify.py均与d303不同。completion记录“未切换performance04”，故该候选包不是本轮代码。historical_manifest03保存d303文件哈希。observations/job_result.json第4行属于旧source ddb4，不可用作当前运行证据；当前运行身份应以d303预检、progress、endpoint_lock及completion报告为准。completion记录最终检查点只在远端，checkpoint SHA256为1ecea4834b77aa87281eabad869ec5309464a08979418564154c98d2630a69ef；本审计没有检查其权重内容。

## 与v3设计及继承协议的核对

- **拓扑、bias及零残差：符合。** model.py 第238–240行创建F1/F2/F3/F4各自无bias的1×1分类器和唯一4类共享bias；第235–237行的C为64→4无bias，D为48→16有bias；第243–247行按标准差0.01初始化W，C权重清零，并将D权重与bias清零；第261–287行计算四尺度分类、在F1网格合并、叠加C，再以PixelShuffle2的D形成Z=U(S)+D。没有发现额外直接头bias或最终分类bias。
- **归一化与BN：符合。** ChannelLayerNorm在model.py第143–164行只对每个像素的C通道调用LayerNorm，输入、affine及统计使用FP32，epsilon为1e-5；主干train覆盖在第111–117行只把BN置于eval模式，没有冻结affine参数。预检backbone_integration记录BN buffer未变；full_resolution_smoke的梯度记录含非零bn1 affine梯度（preflight.json第101–128、2016–2043行）。
- **尺度、预训练和2048路径：符合且有实际门证据。** W48初始化代码要求锁定文件SHA、逐键校验1830个backbone张量并排除126个分类头张量（model.py第63–107行）；预检记录architecture=NativePyramid-SP-v3、相同W48 SHA、逐键身份PASS（preflight.json第101–129行）。主干融合插值替换数为31。预检的2048画布、global batch 2、micro batch 1全分辨率门PASS，共8个丢弃权重更新，且BN buffer不变（preflight.json第2016–2027行）。模型代码将stem投影64→32、G与stem融合，并用D输出对齐输入网格（model.py第228–240、279–287行）。不过预检没有持久化stem及F1–F4的实际张量空间尺寸；因此64×1024、48×512、96×256、192×128、384×64仅有设计和结构代码支持，缺少独立记录的逐层尺寸表。
- **损失、全batch分母、微批与clip：符合。** losses.py第63–75行在完整标签batch计算各类P、K、A及V；第77–114行按图Dice、全batch正例/难负例/有效像素分母计算FP32四类损失；第116–133行对Z和上采样S分别监督，auxiliary系数0.2。train.py第205–237行先计算global-batch分母，用两个microbatch各自累计分子、不乘1/2，之后统一clip一次并step一次。
- **优化预算和学习率：符合。** train.py第138–152行实现基础lr .005、1500步warmup、40k线性衰减、SGD momentum .9和weight decay .0005；protocol.json第27–48行锁定优化器、clip=5及损失常量，train.py第352–383行启动前验证这些锁。progress和endpoint lock确认主运行恰好40000次，固定终点raw checkpoint为主结果。
- **标签、几何和评估：符合。** data.py第14–18、29–47、94–100行锁定EX/HE/SE/MA映射、0背景、255 ignore、454/76/227划分及identity；第59–84行默认核验train和val实际图像/标签文件哈希。增强采用2048画布、0.5–2.0等比例缩放、最近邻标签、旋转合成边界ignore及填充ignore（第108–140行）；val输入固定resize到2048方形（第165–168行）。train.py第241–294行先将S按512→2048→原尺寸对齐、logits回原尺寸后sigmoid并严格大于0.5；metrics.py第5–28行排除ignore、累计四类全图TP/FP/FN后宏平均。test入口要求val目标通过才解锁（evaluate.py第14–25行），与本轮未达标、未访问test一致。

## 结论分类

**设计缺陷：** 本次静态核对未发现会令网络或损失定义自相矛盾的确定性设计错误。已确认的事实是：v3正式固定终点没有达到预先锁定val目标；这说明该配置未通过验收，不能由此确定架构哪一组件导致差距，也不能把性能失败改写为“源码实现不符”。

**实现偏离：** 在本任务要求核对的拓扑、初始化、归一化、主干BN、损失分母、microbatch累积、clip、warmup/衰减、权重身份、类映射和评价几何中，没有发现已被证实的d303运行实现偏离。

**证据缺口：** (1) checkpoint未下载到本地，只有远端路径和SHA；本审计未独立重算其验证分数。(2) preflight记录了全W48及2048 smoke通过，但没有保存stem/F1–F4实际空间尺寸。(3) completion的job_result附件是旧ddb4身份；虽然其余当前记录足以绑定d303运行，报告应明确排除此附件。d303部署包压缩实体未见于completion目录，但服务器预检的15项源码哈希与本地运行源码一致，故没有因此产生运行版本不明的问题。



## SGD 运行遥测记录与回收状态

d303 的 `train.py` 明确在每次成功 SGD 更新后将一行 JSONL 追加到 `out/train.jsonl`（第603–619行；预计40000行），并非源码未实现记录。条目顶层含 `update`、`executed_updates`、`checkpoint_persisted_updates`、`lr`、`seconds_this_process`、`metrics`、`training_sids`。`metrics` 包含 `positive_bce`、`negative_bce`、`squared_dice`、`dense_bce`、`main_loss`、`aux_loss`、`loss`、`grad_norm_before_clip`、`clip_coefficient`、`gradients_before_clip`、`gradients_after_clip`、`supervision_positive_pixels` 和 `supervision_positive_images`。梯度结构由 `gradient_norms()` 定义（第166–202行）：`modules`、`parameter_modules`、`missing_gradient_parameters`、`missing_gradient_parameter_modules`、`direct_scale_class`（四尺度×四类别）、`class_order`、`scale_order`。每次更新均记录裁剪前后梯度，`clip_coefficient` 为实际裁剪乘数。组件损失值对应主 Z 输出，`aux_loss` 单独记录。

已通过授权SSH只读确认远端 `run_seed42_launch02/train.jsonl` 存在且完整，路径、大小、mtime与SHA见 `recovered_logs/train_metrics_all40000.manifest.json`。远端源文件为3,473,259,370 bytes，SHA256 `2b286aee99ed75d894c8d60a239c92018fed158bc98d0def7eed9e61fcf3a63f`；完整原文件没有下载到本地。读取流式核验40,000行，update 1–40,000连续且无重复、缺步或顺序异常。字段投影写入 `recovered_logs/train_metrics_all40000.jsonl`，保留每步loss、lr、clip系数、clip前总梯度范数、clip前后模块范数、四尺度×四类direct_scale_class、正像素/正样本图计数；仅省略两组梯度中的逐参数 `parameter_modules`。投影JSONL SHA256为`a72d77846ace86e429fd4120689ed814aa67668e6dd627c9274216835a5ac3a9`，gzip为 `recovered_logs/train_metrics_all40000.jsonl.gz`；首末原始完整行在 `recovered_logs/train_metrics_boundaries_full.json`。原文件未完整下载；早先中断的177,209,344 bytes前缀仅保留为 `.partial`。因此日志记录实现存在且该远端日志已完整可读；completion未带回原始日志，但诊断需要的逐步字段已通过可审计投影回收，不能归为训练代码“未记录”。

## 运行身份文件清单中的文档

completion observations 下的 d303 `preflight.json` 的 `source_files` 15项中包含 `00_最终架构方案.md` 与 `02_v3_最终架构设计_20261016.md`（后者记录SHA256 `9247fd2d2a6b54fcc3467cd92a7c726d606c62da9739b50b704e9f7237e7865b`）。因此运行时哈希身份对应运行当时的这份设计文件内容；运行后编辑设计稿头部不会改变已记录运行身份，但会使当前文件树无法按15项原始哈希复算出d303。README与新增的04复盘不在该清单。


