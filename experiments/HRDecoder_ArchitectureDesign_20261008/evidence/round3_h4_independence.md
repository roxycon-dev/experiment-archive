# 第三轮核查：H4 stacking 独立性与空间 MAC

审查日期：2026-10-08。只读检查源码、配置、日志和已有 JSON；没有训练、推理或改动模型/方案文件。

## 判定

旧 H4 用的是 **train split 上的成员内预测，不是成员 OOF 预测**。H4 脚本从 `cfg.data.train` 取图，再用加载的 D@2000 学生和其冻结 B reference 在线产生输入；D 本身按同一 DDR split 的 train 集训练，且由 B@3000 初始化。代码没有按图像留折、重训折外成员或读取折外预测缓存。D@2000 前每个训练样本被访问的精确次数没有保存在当前材料中，因此不能断言某张具体图在该 checkpoint 的前 2,000 步里一定被直接采到；但没有按样本排除训练成员，不能把此机制称作 OOF。【`hrdecoder_plus3/本轮实验报告_20260913.md`:L7-L18；`hrdecoder_plus3/repo/configs/lesion/exp/hrplus_patch_dice.py`:L37-L57；`hrdecoder_plus3/repo/hrplus_patch.py`:L64-L88；`HRDecoder_审查轮_20260916/tools/h4_post_fusion_corrector.py`:L104-L116,L170-L202；`HRDecoder_审查轮_20260916/audit_runs/audit_20260916_A/experiments/h4_post_fusion_corrector.json`:L1-L5】

H4 运行时将 D 模型设为 eval 并冻结全部参数，预测放在 `no_grad` 内；这只说明 H4 头优化期间模型不更新，不改变该 checkpoint 在同一训练划分上拟合过标签的事实。R0 元数据也确认 H4 的输入来源是 D@2000 的 student/reference，而不是部署的强系统集成；特征中的基准、student、reference 概率均来自这个固定组合。【`h4_post_fusion_corrector.py`:L112-L120,L184-L202；`R0_counts_meta.json`:L33-L45】

## v2 detach 与 OOF 的区别

v2 输入为归一化 RGB、`clip(z0_MA)`、`clip(z1_MA)`、`clip(z2_MA)`、`p2_MA−p0_MA`；H4 输出是 `sg(z2_MA)+6*tanh(h(sg(X_H)))`。文档明确 `sg` 是 H4 损失梯度说明；主网络仍用同一批样本的 GT/KD 训练未经过 H4 的 `z2`。所以 detach 阻止 H4 loss 回传主网，却没有让 `z2` 来自未见该样本的模型，也没有重建折外预测。v2 H4 因而仍是以训练成员的在线输出作输入；输出分布会随共享主网训练变化，不能把它描述成经典 OOF stacking。【`04_v2_修订架构.md`:L117-L131,L168-L175；`02_架构规格.json`:L96-L110,L143-L158；其中 `train_predictions_are_OOF` 也显式为 `false`，见 L25-L33】

**推断及局限：**与以 OOF 预测拟合二层学习器相比，成员内预测通常会低估训练样本上的残差，存在对训练输出分布适配、验证输入分布变化的风险；detach 不消除此风险。它仍可作为联合模型内的训练头，验证/测试效果必须由完整模型的未参与训练数据评估。这里的判断不等于声称 v2 必然过拟合。

## 可用 OOF / 不同 seed / 未见样本资产

在可见工作区中，没有找到可直接用于 H4 训练的 **train split 成员 OOF 概率缓存**、按样本留出的成员预测集或对应于旧 D/B 接口的多 seed 训练预测资产。发现的 OOF 命名资产属于验证集融合/阈值选择：`make_fusion_folds.py` 明确只给 76 张 val 标签分折，写入 `split="val"`；`crossfit_fusion.py` 在固定 val 概率缓存上交叉拟合融合选项；`fusion_folds.json` 记录 25/25/26 张 val 图并注明分折只用 GT、checkpoint 已由 val 选择。这些是验证折的融合/校准 OOF 统计，不是“成员未见过某训练图”的 OOF 预测。【`HRDecoder_Ready_20261002/tools/make_fusion_folds.py`:L1-L1,L9-L41；`HRDecoder_Ready_20261002/tools/crossfit_fusion.py`:L1-L7,L31-L42,L78-L90；`HRDecoder_Ready_20261002/reports/fusion_folds.json`：`split=val`, `sizes=[25,25,26]`, `selection`, `limitation` 字段】

另有 `HRDecoder_FullCompetition_20260920/refiner_v1/audit/smoke_train_cache_index.json`，但它标记 `split=train`、`n_images=4`、`complete=false`，只是 smoke 缓存。缓存构建代码在 train split 上直接运行冻结 champion，并把其候选 patch 概率存下来；没有折外成员或留样本训练。该 champion 是 D×3、E×2、B×1、E4×1 加 H4，因而这既不是旧 H4 的 D/B 专用输入，也不是 OOF 资产。【`smoke_train_cache_index.json`:L2-L6；`refiner_v1/build_patch_cache.py`:L3-L8,L115-L144,L231-L240】

结论限于当前可见文件及清单：上述 val OOF、四图 smoke cache 与历史 H4 运行代码，都不能充当新 H4 的 train OOF 数据。不能据此证明机器/远端历史目录绝无未登记缓存，但当前没有其路径、索引或来源证据；不应把它们当作现成可用的折外资产。

## 旧 H4 的 MAC 网格和权重复用

旧头是 `Conv(7→16,3×3) + Conv(16→16,3×3) + Conv(16→1,1×1)`，每个输出位置为 `7×16×9 + 16×16×9 + 16 = 3,328 MAC`。因此 2048×2048 网格是 **13,958,643,712 MAC（约 13.96G）**；1024×1024 crop 是 **3,489,660,928 MAC（约 3.49G）**。14G 是 2048² 网格计数，不是 H4 训练 crop 的单次头前向成本。旧训练循环的 student logits 明确为 1024×1024；验证函数将 H4 的 RGB 和预测映射到 `ori_shape`，所以旧验证/部署头的实际网格是原图尺寸：若它为 2048² 才对应 13.96G，通式是 `3328×H_ori×W_ori`。【`h4_post_fusion_corrector.py`:L67-L78,L125-L151,L175-L195；训练 crop 配置见 `hrplus_patch_dice.py`:L7-L24，验证缩放见 `hr_ddr_2048.py`:L39-L53】

空间尺寸本身不改变卷积权重形状：旧头没有归一化层、位置编码或固定尺寸全连接层；2048 与 1024 之间不是 checkpoint 加载障碍。但这不支持直接复用旧 H4：旧七通道为 `[RGB, logit(P_fused), logit(P_student), logit(P_reference), P_student−P_reference]`，其中两路是 D/B、fused 是 50/50 概率均值；v2 七通道改成 `[RGB, clip(z0), clip(z1), clip(z2), p2−p0]`，而且 v2 明确不加载旧权重。故**直接权重不兼容的主要原因是输入语义/成员分布变化，不是 14G 或空间张量尺寸**。旧头在 1024 crop 上拟合、v2 在 2048 网格上训练与部署，可能改变裁块边缘频率和训练/推理上下文分布；这是迁移效果风险，不是卷积形状不匹配。【`h4_post_fusion_corrector.py`:L141-L152,L184-L195；`R0_counts_meta.json`:L33-L45；`04_v2_修订架构.md`:L119-L131；`02_架构规格.json`:L96-L110】
