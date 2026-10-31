# -*- coding: utf-8 -*-
"""trainer.data.external_dataset — 统一 JSONL manifest 的外部病灶数据集。

契约（data_contract/manifest.schema.json，v1.0）：
- 每行一个样本：image.path + lesions.{EX,HE,SE,MA}（键序必须恰为通道序）。
- valid_mode 语义（tools/contract_common.py 权威定义）：
    full = 全图已标注 → target=mask, valid=1
    file = 逐像素有效栅格 → target=mask, valid=valid_mask(非零=1)
    none = 该类未标注 → target=0, valid=0（绝不当阴性！缺类不当阴性）
- official_split 过滤：只用配置允许的 split（pretrain_stage.allowed_splits，
  默认 {train}）；命中 forbidden_splits（必含 test）的记录 **被排除并计数**，
  计数写进 `forbidden_excluded` 供审计（不是静默丢弃：数量与 sample_id 都在
  数据集对象与 preflight 报告里）。若 strict_forbidden=True 则整体报错。
  forbidden_splits 与 allowed_splits 交集非空时无条件报错。
- mask 路径解析：绝对路径直用；相对路径必须提供 manifest_base_dir
  （配置字段，不猜路径）。
"""

from __future__ import annotations

import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from ..gate import GateError
from .augment import TrainAugment, eval_sample
from .common import (
    LESION_CHANNEL_ORDER,
    ContractError,
    normalize_image,
    read_image_rgb,
    read_mask_binary,
    sample_rng,
)

MANIFEST_SCHEMA_VERSION = "1.0"


class ExternalLesionDataset(Dataset):
    def __init__(
        self,
        manifest_path: str,
        *,
        mean,
        std,
        manifest_base_dir: str | None = None,
        allowed_splits=("train",),
        forbidden_splits=("test",),
        augment: TrainAugment | None = None,
        strict_forbidden: bool = False,
        sample_ids=None,
        split_label: str = "external_train",
    ):
        self.manifest_path = manifest_path
        self.manifest_base_dir = manifest_base_dir
        self.augment = augment
        self.epoch = 0
        self.draw_counts = {}
        self.split = split_label
        self.mean = list(map(float, mean))
        self.std = list(map(float, std))
        self.allowed_splits = tuple(allowed_splits)
        self.forbidden_splits = tuple(forbidden_splits)
        self.strict_forbidden = bool(strict_forbidden)
        self.wanted_ids = None if sample_ids is None else set(str(s) for s in sample_ids)
        if "test" in self.allowed_splits:
            raise GateError("allowed_splits 不得包含 test（外部预训练绝不接触官方 test）")
        if "test" not in self.forbidden_splits:
            raise GateError("forbidden_splits 必须包含 test")
        if set(self.allowed_splits) & set(self.forbidden_splits):
            raise GateError(
                f"allowed_splits 与 forbidden_splits 交集非空: "
                f"{sorted(set(self.allowed_splits) & set(self.forbidden_splits))}"
            )

        self.forbidden_excluded: list = []
        if not os.path.isfile(manifest_path):
            raise ContractError(f"external manifest 不存在: {manifest_path}")
        self.records = self._load_records(manifest_path)
        if not self.records:
            raise ContractError(f"external manifest 过滤后为空: {manifest_path}")

    # ------------------------------------------------------------------
    def _load_records(self, path: str) -> list:
        records, forbidden_hits = [], []
        with open(path, "r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ContractError(f"manifest {path} line {lineno}: invalid JSON: {e}")
                self._validate_record(rec, path, lineno)
                sp = rec["official_split"]
                if sp in self.forbidden_splits:
                    forbidden_hits.append((lineno, rec.get("sample_id"), sp))
                    continue
                if sp not in self.allowed_splits:
                    continue  # 明确不在阶段允许集（如 val）：过滤但记录
                if self.wanted_ids is not None and str(rec.get("sample_id")) not in self.wanted_ids:
                    continue  # 显式 sample_id 子集（外部 train 派生的 dev 划分）
                records.append(rec)
        self.forbidden_excluded = forbidden_hits
        if forbidden_hits and self.strict_forbidden:
            # fail closed（严格模式）：宁可不开工，不在含 forbidden split 的
            # manifest 上开工，避免"这份 manifest 是否已清理过"的歧义。
            raise ContractError(
                f"manifest 含 forbidden split 记录 {len(forbidden_hits)} 条"
                f"（如 {forbidden_hits[:3]}）——strict_forbidden=True 下拒绝使用"
            )
        return records

    def _validate_record(self, rec: dict, path: str, lineno: int) -> None:
        where = f"{path}:{lineno}"
        if not rec.get("sample_id") or not isinstance(rec["sample_id"], str):
            raise ContractError(f"{where}: sample_id 缺失或非字符串")
        if rec.get("schema_version") != MANIFEST_SCHEMA_VERSION:
            raise ContractError(
                f"{where}: schema_version 必须为 {MANIFEST_SCHEMA_VERSION!r}，"
                f"实际 {rec.get('schema_version')!r}"
            )
        if rec.get("official_split") not in ("train", "val", "test"):
            raise ContractError(f"{where}: official_split 非法: {rec.get('official_split')!r}")
        lesions = rec.get("lesions")
        if not isinstance(lesions, dict) or tuple(lesions.keys()) != LESION_CHANNEL_ORDER:
            raise ContractError(
                f"{where}: lesions 键必须恰为 {list(LESION_CHANNEL_ORDER)}（顺序不可变）"
            )
        for name, entry in lesions.items():
            mode = entry.get("valid_mode")
            if mode not in ("full", "file", "none"):
                raise ContractError(f"{where}: lesions.{name}.valid_mode 非法: {mode!r}")
            mask_path = entry.get("mask_path")
            vmask_path = entry.get("valid_mask_path")
            if mode == "none":
                if mask_path is not None:
                    raise ContractError(
                        f"{where}: valid_mode=none 时 mask_path 必须为 null（{name}）"
                    )
            else:
                if mask_path is None:
                    raise ContractError(f"{where}: valid_mode={mode} 但 mask_path 为 null（{name}）")
                if mode == "file" and vmask_path is None:
                    raise ContractError(
                        f"{where}: valid_mode=file 但 valid_mask_path 为 null（{name}）"
                    )
                if mode == "full" and vmask_path is not None:
                    raise ContractError(
                        f"{where}: valid_mode=full 时 valid_mask_path 必须为 null（{name}）"
                    )
        if not isinstance(rec.get("image"), dict) or not rec["image"].get("path"):
            raise ContractError(f"{where}: image.path 缺失")

    # ------------------------------------------------------------------
    def _resolve(self, p: str) -> str:
        if os.path.isabs(p):
            return p
        if not self.manifest_base_dir:
            raise GateError(
                f"manifest 相对路径需要显式 base：配置 pretrain_stage.manifest_base_dir "
                f"缺失（manifest={self.manifest_path}, path={p!r}）——不猜路径"
            )
        return os.path.join(self.manifest_base_dir, p)

    def _load_arrays(self, idx: int):
        rec = self.records[idx]
        img = read_image_rgb(self._resolve(rec["image"]["path"]))
        h, w = img.shape[:2]
        targets = np.zeros((4, h, w), dtype=np.float32)
        valid = np.zeros((4, h, w), dtype=np.float32)
        for c, name in enumerate(LESION_CHANNEL_ORDER):
            entry = rec["lesions"][name]
            mode = entry["valid_mode"]
            if mode == "none":
                continue  # valid=0：缺类不当阴性，损失端整类屏蔽
            mask = read_mask_binary(self._resolve(entry["mask_path"]))
            if mask.shape != (h, w):
                raise ContractError(
                    f"{rec['sample_id']}: {name} 掩膜尺寸 {mask.shape} 与图像 {(h, w)} 不符"
                )
            targets[c] = mask.astype(np.float32)
            if mode == "full":
                valid[c] = 1.0
            else:  # file：逐像素 valid 栅格（非零=1）
                vmask = read_mask_binary(self._resolve(entry["valid_mask_path"]))
                if vmask.shape != (h, w):
                    raise ContractError(
                        f"{rec['sample_id']}: {name} valid 掩膜尺寸不符"
                    )
                valid[c] = vmask.astype(np.float32)
        return img, targets, valid, rec["sample_id"], rec.get("source", "external")

    def __len__(self) -> int:
        return len(self.records)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        self.draw_counts = {}

    def load_raw(self, idx: int):
        """整图评估专用：返回未归一化、未增强的 (uint8 RGB, target, valid, sample_id)。"""
        img, targets, valid, sample_id, _source = self._load_arrays(idx)
        return img, targets, valid, sample_id

    def __getitem__(self, idx: int):
        img, targets, valid, sample_id, source = self._load_arrays(idx)
        if self.augment is not None:
            draw = self.draw_counts.get(idx, 0)
            self.draw_counts[idx] = draw + 1
            rng = sample_rng(torch.initial_seed() + draw*1000003, idx, self.epoch)
            img, targets, valid = self.augment(img, targets, valid, rng)
        else:
            img, targets, valid = eval_sample(img, targets, valid)
        img_t = torch.from_numpy(normalize_image(img, self.mean, self.std).transpose(2, 0, 1).copy())
        return {
            "image": img_t,
            "target": torch.from_numpy(targets.copy()),
            "valid": torch.from_numpy(valid.copy()),
            "meta": {"sample_id": sample_id, "split": self.split, "source": source},
        }


def deterministic_dev_split(records: list, *, dev_ratio: float, seed: int) -> tuple:
    """按 sample_id 的稳定哈希把外部 **train** 记录切成 (dev_ids, train_ids)。

    只用 official_split=train 的记录（调用方保证），只用于
    `node_selection: external_train_derived_dev_only`：
    预训练期的节点选择只能用外部 train 派生的 dev，绝不接触官方 val/test。
    划分是 sample_id + seed 的纯函数 ⇒ 可复现、无随机漂移。
    """
    import hashlib

    if not 0.0 < float(dev_ratio) < 1.0:
        raise GateError(f"dev_ratio 必须在 (0,1) 内，当前 {dev_ratio}")
    if "test" in {str(r.get("official_split")) for r in records}:
        raise GateError("deterministic_dev_split 收到 official_split=test 的记录（拒绝）")
    dev, train = [], []
    for r in records:
        sid = str(r.get("sample_id"))
        h = hashlib.sha256(f"{seed}:{sid}".encode("utf-8")).digest()
        u = int.from_bytes(h[:8], "big") / float(2**64)
        (dev if u < float(dev_ratio) else train).append(sid)
    if not dev:
        raise GateError("dev 划分为空：dev_ratio 过小或记录太少，拒绝静默退化为全量训练")
    if not train:
        raise GateError("train 划分为空：dev_ratio 过大，拒绝在无训练样本下开工")
    return dev, train


def source_balanced_weights(records: list) -> list:
    """source_balanced_sampling：权重 = 1 / 该 source 样本数（按数据源均衡）。"""
    counts = {}
    for r in records:
        s = r.get("source", "unknown")
        counts[s] = counts.get(s, 0) + 1
    if not counts:
        raise GateError("source_balanced_weights: records 为空")
    return [1.0 / counts[r.get("source", "unknown")] for r in records]
