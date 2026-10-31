# 当前唯一执行方案

自 2026-09-19 起，后续实验以此文件为唯一主方案：

`C:\Users\GuanXingRen\Documents\Codex\2026-09-19\d-dataset-experiment-hrdecoder-fusionaware-20260917\outputs\HRDecoder_训练优化方案_20260919_深度修订.md`

执行优先级：

1. 核清初始化来源、学生/参考/旧融合/完整冠军的真实调用路径。
2. 完成训练集尺度、实例采样、外部数据排除和实测速率检查。
3. 运行匹配的 C/S/P 受控比较；外部数据或伪标签未过门时删除 P。
4. 对胜出方向进行成对第二 seed 复核。
5. HACDR、分类预训练仅在前序证据与预算允许时进入。
6. 最终只锁定一个候选运行当前 test 一次。

旧版 `NEXT_ROUND_PLAN_20260919.md` 与 `NEXT_ROUND_PLAN_20260919_REVISED.md` 均为历史材料，不再直接执行。
