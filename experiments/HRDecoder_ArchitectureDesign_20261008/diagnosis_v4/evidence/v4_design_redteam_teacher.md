# v4 教师蒸馏方案红队复审

只读复审；未训练、推理或搜索文献。

## 初审结论

v4以有界概率融合代替v3无界logit残差，分支目标更直观；但变成教师专属多头KD，近Hydra，不能保证新颖。P=.5625pA+.1875pU+.25pS仅在类序、坐标、概率和权重一致时等于T；若pA=0、pU=pS=1，则P=.4375，过不了严格>.5。应报告各类GT+阈值曲线。

三头/最终P都做GT，分支KD外又有最终KD；须定全系数与detach边界，尤其pA-KD是否更新原decoder。新GT损失偏离基线“无class weighting”，需同损失B0/B1。40k仅同更新数；6.38M只计L/G，另报头参数/MAC。核W48源SHA，不说初始预测等同B0。A含训练内H4、缓存OOF=false；旧train与官方test/valid重叠142/80张，不能称官方独立测试。v4失去v3有符号分歧假设；对照普通KD、真实Hydra及文献审查前不主张创新。

## 更新复审

取消fusion ramp使固定学生P从首步代数对齐最终T；same ImageNet W48且不声称预测逐位同B0，口径更准确。λKD端点及component KD系数/时程仍须定义。

末层零初始化可解随机头阈值风险，但仅保证二值决策：若两头末层仿射、sigmoid紧随且W/b=0，则pU=pS=.5，P0=.5625pA+.21875=.5+.5625(pA-.5)，故严格>.5的决策与当前pA相同（pA=.5仍不检）。P0压缩到[.21875,.78125]，不等于B0概率，也非性能/校准保证。零W使首轮tower body无损失梯度（weight decay仍作用）；head先更新，W非零后下一步才有body任务梯度。这只类似v3最终融合主路径；v3 L/G另有非零contrast读出从首步受辅助梯度。

`g`只是GT二值决策一致，不保证KL梯度同向：GT0、q=.49、p=.05时KL仍推高p；GT1、q=.6、p=.9时仍压低p。全域CE会竞争，不能称消除冲突。`sum(m)`分母避免稀少一致区放大，但KD有效强度随掩码覆盖率变化，须记各师/类阳阴覆盖和梯度。GT系数和为1；融合Jacobian下A/U/S名义乘数约.531/.219/.250，仅是假设导数相同的权重，不是实测梯度比例。若删空类Dice，须与保留项配对。

参数6,382,912=6,419,648−36,736，账目一致；44.697−9.546=35.151G，与35.150G差0.001，可能舍入，给未舍入MAC表。最小对照：无component KD、空类Dice、v3、匹配Hydra；同40k采样/val，多seed并报实测成本。

证据根：`D:\dataset\experiment\HRDecoder_ArchitectureDesign_20261008`；`05_v3_机制重构与反驳审查.md:111-127,171-211`；`evidence\review_v3_protocol.md:11-15,28,38,48-52`；基线`D:\dataset\lesion_segmentation\01_板块一_baseline实验配置.py:11-58`。
