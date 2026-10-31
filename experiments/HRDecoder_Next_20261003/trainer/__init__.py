# -*- coding: utf-8 -*-
r"""trainer — H0/H1/U0/U1 共享训练核心（可上传服务器的独立训练入口依赖本包）。

四臂共享同一套：数据管线（DDR 固定 split + 外部 JSONL manifest）、
masked BCE + soft Dice 损失、整图累计指标、checkpoint 格式与训练循环。

设计约束（见 docs/TRAINING_IMPLEMENTATION_STATUS.md）：
- 旧仓库（D:\HRDecoder，只读）的 frozen reference、多裁剪融合包装
  （HRDecoder segmentor）与 MA residual head（DetailBranch/DetailHead 残差）
  一律不进入新模型；H 臂仅以 adapter 方式复用旧 HRNet-W48 主干 + FCNHead。
- 所有契约常量从 tools/contract_common.py 导入（单一权威定义），不重复定义。
- fail closed：占位符、test split、静默 resize、静默部分加载一律拒绝。
"""

__version__ = "1.0.0"

CHANNEL_ORDER_NOTE = (
    "通道序固定 EX/HE/SE/MA（tools/contract_common.LESION_CHANNEL_ORDER），"
    "通道 i 对应 DDR class id i+1。"
)
