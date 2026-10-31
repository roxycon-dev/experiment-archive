# -*- coding: utf-8 -*-
"""trainer.data.ddr_dataset — DDR 固定 DDR_split_613 数据集。

目录语义（依据真实目录与审计文档第 6 节，不猜路径）：
- <root>/images/<split>/*.jpg          眼底图
- <root>/labels/<split>/*.png          融合 PNG（mode L，值 0..4；训练真值）
- <root>/labels/<split>/{EX,HE,SE,MA}  逐类 TIF 存档（不参与训练接口，绝不读取）

输出：
- image:  float32 [3,H,W]（按配置 mean/std 归一化）
- target: float32 [4,H,W]（通道 i = fused == i+1，即 EX/HE/SE/MA）
- valid:  float32 [4,H,W]（DDR 全类已标注 ⇒ 全 1）

test split 不由本数据集加载；训练/验证 split 由 train.py 从配置解析并
硬拒绝 test（见 trainer.gate / train.py assert_no_test_split）。
"""

from __future__ import annotations

import os

import numpy as np
import torch
from torch.utils.data import Dataset

from ..gate import GateError
from .common import (
    ContractError,
    fused_label_to_targets,
    normalize_image,
    read_ddr_fused_label,
    read_image_rgb,
    sample_rng,
)
from .augment import TrainAugment, eval_sample

IMG_EXT = ".jpg"
LABEL_EXT = ".png"


class DDRSegDataset(Dataset):
    def __init__(
        self,
        ddr_root: str,
        split: str,
        *,
        mean,
        std,
        augment: TrainAugment | None = None,
    ):
        if split not in ("train", "val"):
            raise GateError(
                f"DDR 训练数据集只允许 split train/val，收到 {split!r}；"
                "test 仅允许最终一次独立评估，绝不进入训练/验证数据集"
            )
        self.ddr_root = ddr_root
        self.split = split
        self.augment = augment
        self.epoch = 0          # 由 engine 每个 epoch 调 set_epoch() 推进
        self.mean = list(map(float, mean))
        self.std = list(map(float, std))
        if len(self.mean) != 3 or len(self.std) != 3:
            raise GateError("ddr_normalization mean/std 必须各 3 个数")

        img_dir = os.path.join(ddr_root, "images", split)
        lbl_dir = os.path.join(ddr_root, "labels", split)
        for d in (img_dir, lbl_dir):
            if not os.path.isdir(d):
                raise ContractError(f"DDR 目录缺失: {d}")
        self.img_dir, self.lbl_dir = img_dir, lbl_dir

        self.stems = sorted(
            os.path.splitext(f)[0]
            for f in os.listdir(img_dir)
            if f.lower().endswith(IMG_EXT) and os.path.isfile(os.path.join(img_dir, f))
        )
        if not self.stems:
            raise ContractError(f"DDR split {split} 无图像: {img_dir}")
        missing = [s for s in self.stems
                   if not os.path.isfile(os.path.join(lbl_dir, s + LABEL_EXT))]
        if missing:
            raise ContractError(
                f"DDR split {split} 图像-标签不对齐，缺 {len(missing)} 个标签，"
                f"例如 {missing[:5]}"
            )

    def __len__(self) -> int:
        return len(self.stems)

    def set_epoch(self, epoch: int) -> None:
        """推进增强随机流（train loader 每个 epoch 前由 engine 调用）。"""
        self.epoch = int(epoch)

    def load_raw(self, idx: int):
        """整图评估专用：返回未归一化、未增强的 (uint8 RGB, target, valid, stem)。

        指标路径必须走这里——归一化/pad 由 evaluator 显式完成，
        绝不在数据层偷偷 resize。
        """
        return self._load_arrays(idx)

    def _load_arrays(self, idx: int):
        stem = self.stems[idx]
        img = read_image_rgb(os.path.join(self.img_dir, stem + IMG_EXT))
        fused = read_ddr_fused_label(os.path.join(self.lbl_dir, stem + LABEL_EXT))
        targets = fused_label_to_targets(fused)
        valid = np.ones_like(targets, dtype=np.float32)  # DDR：四类全标注
        return img, targets, valid, stem

    def __getitem__(self, idx: int):
        img, targets, valid, stem = self._load_arrays(idx)
        if self.augment is not None:
            rng = sample_rng(torch.initial_seed(), idx, self.epoch)
            img, targets, valid = self.augment(img, targets, valid, rng)
        else:
            img, targets, valid = eval_sample(img, targets, valid)
        img_t = torch.from_numpy(normalize_image(img, self.mean, self.std).transpose(2, 0, 1).copy())
        return {
            "image": img_t,
            "target": torch.from_numpy(targets.copy()),
            "valid": torch.from_numpy(valid.copy()),
            "meta": {"sample_id": stem, "split": self.split, "source": "DDR"},
        }
