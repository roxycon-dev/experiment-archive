# HRDecoder-IoUCorrection

2026-10-10；版本16，2048/batch2/40000，fresh ImageNet W48 seed42/private20261008，初始输出exact base，私有284496/总65856916。semantic4k拒绝后不恢复旧训练，保留checkpoint及RAM。

本轮只把私有分支的原 squared Dice 替换为二值Lovasz hinge。私有GT=Lovasz hinge+.25 fullGT正负平衡BCE，teacherKL仍fullGT正负固定.5/.5、既有lambda_K；base仍Dice+.25平衡BCE+.1HRcrop+.5lambda_K平衡固定GT一致teacherKL。网络、teacher、LR4x、单SGD、sampler、BN、分别clip5、collapse-stop同Semantic，没有重加旧actionKL。

Lovasz hinge按每张、每类有效像素的margin1误差排序计算Jaccard增量。只排序严格正误差，但分母的GT阳性总数含所有有效GT阳性；不是top-k或负采样近似。CPU与dense原式loss和logitgradient逐元素比较；ignore/空类/饱和正负/排列不变性/实际private routing/resume共19项。hinge最大错误损失用于空但有标注的类，全部无效返回可微0；边界tied-errors允许有效子梯度，随机无tie比较精确。实现采用FP32/测试FP64。参考Berman等CVPR2018论文与作者公式https://github.com/bermanmaxim/LovaszSoftmax，不能据论文保证本数据集达标。

证据：前一full76val private提高HE/MA却降低EX/SE，8个trainbatch无SGDlogit梯度诊断支持旧误检梯度质量偏弱；不同损失标度/参数空间传播仍须真实GPU与freshval验证，不作唯一因果。

新4k gate完全保持既有数值：76val>=30.475812328119%、final-base>=1pp、各类非零/相对base不跌超2pp、HE/MA FN/FP tradeoff全通过后，同一fresh继续40k。CPU pinned19+真实2048/b2非零辅助12步GPU gate缺一不可。严重完整val崩塌保存并停止；失败自主证据改造，不放宽已见结果gate。

服务器保持开机、外部诊断不控制，96GiB RAM39434复用；专用key、小型日志回执、无权重下载/全cache哈希。最终只固定40k76val>45.32183122738841%后锁定40k单次custom227test>45.860383832668%才达目标。test不选择结构/阈值/权重，推理无GT/teacher/TTA/H4。不保证结果。
