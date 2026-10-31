# -*- coding: utf-8 -*-
"""trainer.data — 共享数据层：DDR 固定 split 数据集、外部 JSONL manifest 数据集、
共享几何增强。所有契约常量从 tools/contract_common.py 导入（单一权威定义）。"""

from .common import (  # noqa: F401
    LESION_CHANNEL_ORDER,
    DDR_CLASS_INDEX,
    ContractError,
    load_contract_common,
)
from .ddr_dataset import DDRSegDataset  # noqa: F401
from .external_dataset import ExternalLesionDataset  # noqa: F401
