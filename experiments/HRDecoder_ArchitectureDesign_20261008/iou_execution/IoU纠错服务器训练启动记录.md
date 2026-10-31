# IoU纠错服务器训练启动记录

核验时间2026-10-10T15:56:49.618882+08:00。正式train64343/controller63572，identity49ee0537d2802450284844a32405a6da0116ee9b77c2ab519aa9f985d15d47d4。已核验日志100正式更新/性能120正式更新，loss有限，GPU100%/322.77W，datawait0.00004738秒；自己的96GiB RAM39434复用。

原torch2.0.1+cu117 CPU19项PASS，真实2048/batch2非零辅助12步GPU gate PASS，all6private parts非零有限梯度；private284496/total65856916。新fresh ImageNetW48 seed42/private20261008，无旧task权重，smoke12次优化丢弃后正式重新初始化。程序、规格及screen source identity与本地小型核心文件相同。model.py与Semantic逐字节相同，唯一数学目标改动是私有squaredDice→Lovasz hinge；平衡BCE/KD/base/objective routing/分别clip5/sampler/LR4x不变。

Lovasz exact active-margin实现不做top-k，完整GT正数保留；CPU与dense原式value/gradient、ignore/empty/permutation/saturation及实际SGD隔离/恢复验证。新目标未有val，不能宣称已达标或保证结果；不同损失标度logit梯度探针不能证明唯一因果。

GPU非零辅助warm smoke平均1.160543秒/步，allocated18.28GiB；预计4k筛查2026-10-10T17:15:05.474192+08:00，实际1500后100稳态需更新ETA，前1500不用于完整外推。PASS才同一fresh继续固定40k，含最终验证/test预留预计2026-10-11T05:34:25.034769+08:00。

筛查与原数值一样：完整76mIoU>=30.475812328119%、private+1pp、非零/类差不跌2pp、HE/MA FN/FP规则。REJECTED自主证据改造，不问选择；严重完整val崩塌保存即停。服务器一直开机，不控制外部诊断。主hrdecoder-4每45分钟，iou4k专门交接一次后删；旧semantic4k已删。日志/回执保存server_receipts/iou，不下载checkpoint或重扫cache。

最终只固定40k76val>45.32183122738841%后锁定同一40k单次custom227test>45.860383832668%才达目标；test不选结构/阈值/权重，推理无GT/teacher/TTA/H4。详情00_IoU纠错方案.md与LAUNCH_VERIFIED.json。

## 最终轮与收尾授权更新

2026-10-10T16:07:34.934548+08:00：用户指定当前IoUCorrection为最后一版，原4k预登记阈值保持；未过则停止、不再改造或重开，通过才同一fresh继续固定40k。终态先归档日志/指标及完整报告所需证据，再执行整台服务器关机。此授权取代此前保持开机/失败自主改造的要求，不主动控制外部诊断进程。不下载checkpoint，不重复数据/cache哈希。
