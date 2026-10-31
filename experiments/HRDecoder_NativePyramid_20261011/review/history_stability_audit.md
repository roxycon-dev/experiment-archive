# 历史稳定性审查：实际阅读覆盖与方案约束

审查日期：2026-10-11。用户要求只写方案，不再训练；随后要求立即输出。本审查因此停止扩大阅读。没有启动训练、SGD、新推理或读取新 test 像素，没有改动原型和历史证据。

## 覆盖边界

分配清单为 102 份 SHA 去重 Markdown；实际完整阅读 **73 份**，其余 **29 份仅登记清单，未完整阅读**。逐篇路径、SHA、状态及一句发现见 `coverage_stability_audit.json`。没有把搜索命中当成完整阅读；全文过长时分段读全后才计入 full_read。本次完整阅读不是对全部历史代码、全部原始日志或全部实验目录的穷尽复核。受立即输出要求，本轮新增的报告数值未逐项重新核验对应原始 JSON；下述出处标明报告层证据。上一轮对当前原型的源码与已有探针检查单独列示。

## 会改变设计与验收的历史证据

1. **早期 PASS 不能承担长期稳定承诺。** Detail 真实 2048/batch2 的 12 步 smoke 与 4k 全部门槛均通过；4k/8k/12k 的 mIoU 为 32.033820/34.923222/36.735849，16k 跌至 13.074770，20k EX/SE 均为零。同期 gt_base loss 下降且参数与 BN 有限。见 `detail_execution/半分辨率细节纠错服务器训练启动记录.md`、`detail_execution/趋势分析记录.md`，以及 `final_report/HRDecoder完整实验报告.md:270–299,398,467`。应分开记录数值通过、四类学习通过、固定 40k 结果；严重类别失活保存并停止，不能因 loss 下降继续等下一个 4k。

2. **正组缺失时不能把负组系数翻倍，且分层归一化从来不是精度保证。** `diagnosis_v4/evidence/real_gradient_review.md:27–45` 已要求缺组贡献零、系数不重新分配；也警告逐阳性图等权会放大小标注/噪声，普通全背景均值会稀释稀少 FP。CET 的方案已采用有阳性才启用 Dice、固定 .25(B+ + B−)，实际完整 40k 仍只有 38.313759%，低于基线端点 38.835%，见 `research_v5/00_HRDecoder_CET_架构与训练方案.md:164–202` 和 `final_report/HRDecoder完整实验报告.md:120–174,396`。v2 修复归一化应称实现层修复，不能称已验证达标。

3. **更强 FP 梯度和更高 precision 可能伴随严重漏检。** Semantic 后的 IoUCorrection 在训练无 SGD 探针中，SE FP 处 Lovasz logit 梯度约为旧损失 60.95 倍；19 项 CPU 和非零辅助 2048 smoke 均 PASS。然而 4k final−base 为 −6.249143pp，SE FP 少 256646、FN 多 102833，recall 降 51.058072pp。见 `final_report/HRDecoder完整实验报告.md:406–440`、`iou_execution/趋势分析记录.md:5–18`。v2 的 OHEM 必须以 TP/FP/FN 共同验收，不能以负梯度大、precision 高判好。

4. **detach 不隔离全局梯度裁剪。** 历史 Semantic 起改为 base/private 各自 L2 clip5，CPU 实际 SGD 配对确认放大 private 梯度不再改变 base 更新；其泛化收益未被单因素隔离。见 `final_report/HRDecoder完整实验报告.md:400,404`。新架构若是联合训练，可保留统一 clip5，但不能声称辅助分支不影响主干优化尺度。

5. **饱和恢复只能陈述 logit 梯度性质。** v3/Detail 的 GT 阳性概率饱和与旧 Dice 救援梯度弱有真实冻结状态证据；修复 BCE 的 logit/分类偏置方向可逆转，BN 交叉未救回，不等于唯一触发原因已知，也不等于含动量、共享参数的后续训练必恢复。见 `diagnosis_v4/00_诊断结论与v4方案.md`、`diagnosis_v4/evidence/real_gradient_review.md`、`detail_execution/diagnosis_16000/诊断结论与停止记录.md`。

6. **不能把旧配对诊断写成干净因果成功。** 归一化暖启动 500 点实验组 final 高 1.632108pp，但 base 高 2.011336pp，MA FP 多 29144，两组终点都低于父状态。修复配对一致性后只完成 128 更新，无 250/500 结果。见 `normalization_execution/配对诊断指标与分析.md`、`research_v10/00_配对一致性修复与下一轮验证计划.md`、`research_v10/02_GPU配对修复实测与未完成记录.md`，汇总见 `final_report/HRDecoder完整实验报告.md:374–392`。相同种子和终态 RNG 不证明每步输入/状态一致。

7. **短期零检出是未通过证据，不能单凭它锁定根因或宣布永久失败。** PrivateExpert 的 HE 在 2k 为零，但 4k 恢复至 IoU 40.556773%；该轮仍因别的门槛拒绝。见 `expert_execution/验证指标与趋势.md:7–8`。当前 NativePyramid 的 SE 短探针应保持 NOT_PASSED，既不能包装成功，也不能仅凭零检出证明某一损失是唯一根因。

8. **归档早期拒绝不等于架构无希望。** Residual 8k 较 4k 上升 7.736538pp，仅因 final−base 为 .278182pp 低于 .5pp 拒绝；Adapter 4k 仅差绝对 floor .031606pp。见 `residual_execution/趋势分析记录.md:21–24`、`adapter_execution/趋势分析记录.md`。原筛查是资源规则，未证明与最终性能必然对应。

9. **评价身份必须固定。** 当前目标 val=45.32183122738841%、test=45.860383832668% 来自异构概率融合；B0 的 test42.20%属于40k预算内32k最佳权重。历史 test 已被多轮研究使用，不能称新盲测。旧 H4 是成员内 train 预测，非 OOF，旧静态 H4 的实际收益也不能迁移为单 W48 预期收益。见 `evidence/teacher_protocol_audit.md:7–33,60`、`evidence/round3_h4_independence.md:7–29`、`evidence/round3_history_mechanism.md`。

## 当前原型与 v2 方案的阻断要求

上一轮已直接审查当前原型 `losses.py/metrics.py/train.py/verify.py/model.py` 和 attempt1/2 原始短探针。两次探针分别 64、128 discarded updates，均为 NOT_PASSED。attempt2 最终 SE TP=0、FP=5、FN=6004；2048 两步 smoke 的真实首批只有 MA，不能称已证明四类 2048 学习。原型保留原样，方案修改不冒充重新实测。

- v1 按图、类先平均，缺阳性时余下负组由二分之一变成一；SE 原生阳性图率143/454，汇总名义负/正组权重比为 (2−q)/q≈5.35，短探针 q=2/8 时比值7。该数学偏置值得修复，但不是 SE 零检出的唯一根因证明。
- v2 候选应明确：B=2 全局正像素总分母 P_c；各图 top-k 再汇总 K_c（不是全batch top-k）；每类 Dice 只在有阳性图上平均；空组 numerator=0、分母 max(count,1)，固定 .25/.25 系数；.05 dense BCE 按全局 valid 总数。主头和辅助头都用各自 logits、相同 GT 全局计数，aux 权重 .2。
- 微批贡献必须按预先计算的全 B=2 计数直接相加，**不能再乘 micro_batch/2**。各图 top-k 规则、Dice 定义和 tie 处理固定；一次 clip、一次 SGD；部分微批失败不作半步更新。若更换为非独立图归一化或随机算子，必须重审与真实 batch2 的等价边界。
- 固定 BN 仅冻结运行统计，仿射参数和前层特征仍会改变；GN 和 clip 不能保证长期不崩塌。
- 区分 never-learned class 的持续零 TP 与已学会后饱和丢失；仅检查 logit<-12 会遗漏零召回但未深饱和状态。
- 恢复必须在完整 optimizer update 边界保存 PAUSED 及模型/optimizer/scheduler/RNG/sampler/guard/history，拒绝半步状态。当前代码 CtrlC 保存 FAILED 且拒绝恢复、OOM可能留旧 last.pt；仅修文档不得写成已实现恢复。

上述为静态设计可接受条件，不是训练通过回执。最终 v2 文档若完整落实，可以通过“方案一致性”审核；真实四类学习、40k稳定性以及是否超过融合仍未实测。


## Stability Luna 补充复核（2026-10-11）

按既有 SHA 去重阅读表仅新增全文阅读科学正文15篇；累计88/102全文读，余14篇为运维启动/资源/归档说明或字段来源元数据，标为 index_only，不据其推断实验结论。逐篇状态与发现更新在 `coverage_stability_audit.json`，审阅队列同步于 `reading_stability_audit.json`。

静态复核主方案第4节：固定0.25正/负组权重、空组零贡献不重分配；正例CE按完整 batch `P_c`，负例top-k按每图汇总 `K_c`，Dice按阳性图数 `A_c`，dense BCE按全局有效像素 `V`；微批1只累加局部分子且不乘1/2；ignore像素不进统计，整批全ignore拒绝更新。极负阳性logit有softplus局部上行梯度，但共享参数更新、召回与最终mIoU不受此代数保证。阳性图Dice等权是有意混合目标，也会给极小阳性图相对更大影响，不能声称无偏或性能保证。静态公式无代数阻断缺陷。

仍阻断方案定稿的两点：top-k平分值没有确定性索引tie-break；按logit降序、有效阴性展平索引升序稳定排序后取k即可。健康规则第6项“已学类别零检出”未规定连续次数，现有v1 `metrics.py` 是有过IoU≥5%后单次val recall≤0.1%即停；历史PrivateExpert HE在2k零检出、4k恢复，说明此规则可能重复短期误停。建议full-val须连续两个节点≤0.1%，且此前有稳定学习证据；单次只记录/诊断。第5项的“历史探针recall≥5%”也应要求不早于4k且连续两探针达到，避免早期一次偶然过线后触发失学。触发状态须明确是可诊断、可恢复PAUSED。

15pp mIoU大跌阈值能覆盖Detail 12k→16k与v3 16k→20k的大幅崩塌，且不把约5–6pp短期回落立即淘汰；应保持暂停诊断语义。预检末两点四类IoU≥20%是可学习性筛查，明确非泛化判断；未授权任何训练或新test像素读取。
