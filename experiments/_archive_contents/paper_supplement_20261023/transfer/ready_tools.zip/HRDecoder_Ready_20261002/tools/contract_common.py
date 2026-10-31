# -*- coding: utf-8 -*-
"""contract_common.py — 统一外部病灶数据契约的共享常量与工具函数。

本模块是四通道顺序契约的唯一权威定义（single source of truth）。
所有工具（build_external_manifest / audit_manifest / inspect_ddr_contract /
preflight_contract）必须从这里导入常量，禁止各自硬编码。

契约依据（见 docs/ASSET_AND_INTERFACE_AUDIT.md 第 1 节）：
- DDR 评估的 sigmoid 输出为 4 通道，通道 i 与 DDR class id i+1 对齐
  （lesion_metric.py:63-79：`label = raw_label == i`，通道 `i-1`）。
- DDR class id：bg=0, EX=1, HE=2, SE=3, MA=4（convert_dataset/ddr.py:39-44）。
- 因此通道序 EX/HE/SE/MA 与 DDR 评估序一致，无需置换；但映射必须显式。
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# 类别与通道契约
# ---------------------------------------------------------------------------

# 病灶四通道顺序（不含背景）。通道 0..3 依此对应 DDR class id 1..4。
LESION_CHANNEL_ORDER: Tuple[str, ...] = ("EX", "HE", "SE", "MA")

# DDR 融合标签的 class id（背景 + 四病灶），与
# D:\HRDecoder\tools\convert_dataset\ddr.py:39-44 逐字一致。
DDR_CLASS_INDEX: Dict[str, int] = {"bg": 0, "EX": 1, "HE": 2, "SE": 3, "MA": 4}

# 通道索引 -> 病灶名（与 LESION_CHANNEL_ORDER 恒等，防呆冗余定义）
CHANNEL_INDEX_OF_LESION: Dict[str, int] = {
    name: idx for idx, name in enumerate(LESION_CHANNEL_ORDER)
}

# 通道 i 对应的 DDR class id 恒等于 i+1（评估端 lesion_metric.py 的对齐规则）。
DDR_CLASS_ID_OF_CHANNEL: Dict[int, int] = {
    idx: DDR_CLASS_INDEX[name] for name, idx in CHANNEL_INDEX_OF_LESION.items()
}
# 反向：DDR class id -> 通道索引（class 0 是背景，无通道）
DDR_CHANNEL_OF_CLASS_ID: Dict[int, int] = {
    v: k for k, v in DDR_CLASS_ID_OF_CHANNEL.items()
}

# 病灶类 -> DDR 逐类 TIF 子目录名（labels/train/EX/... 等存档目录）
DDR_PER_CLASS_DIRNAME: Dict[str, str] = {c: c for c in LESION_CHANNEL_ORDER}

# ---------------------------------------------------------------------------
# manifest 契约
# ---------------------------------------------------------------------------

MANIFEST_SCHEMA_VERSION = "1.0"

# 数据源白名单（当前只做 IDRiD 与 MAPLES-DR；FGADR 在取得授权前不得加入）
ALLOWED_SOURCES = ("IDRiD", "MAPLES-DR")

# official_split 白名单：记录数据集官方划分，禁止改写语义
ALLOWED_OFFICIAL_SPLITS = ("train", "val", "test")

# valid 语义：
#   full = 该类在全图均已标注（valid_mask 隐式全 1，valid_mask_path 为 null）
#   file = 逐像素有效区域存于 valid_mask_path（栅格，非零=有效）
#   none = 该类完全未标注（mask_path 必须为 null；绝不能当阴性）
VALID_MODES = ("full", "file", "none")

# 判定"该类是否标注"的字段名
ANNOTATED_FLAG = "annotated"

# ---------------------------------------------------------------------------
# 哈希 / 近重复
# ---------------------------------------------------------------------------

# 感知哈希比特数（aHash/dHash/pHash 均为 64bit）
PHASH_BITS = 64
# 64bit 汉明距离 <= 该阈值判定为近重复候选（保守值，人工复核用报告呈现）
NEAR_DUP_HAMMING_THRESHOLD = 6


class ContractError(Exception):
    """契约违反（fail closed 用）。所有工具捕获后以非零退出码终止。"""


def sha256_file(path: str, chunk_size: int = 1 << 20) -> str:
    """流式计算文件 SHA256（十六进制）。文件不存在时抛 ContractError。"""
    if not os.path.isfile(path):
        raise ContractError(f"file not found: {path}")
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_manifest(path: str) -> List[dict]:
    """读取 manifest JSONL；每行必须是 JSON 对象。损坏行直接抛错（fail closed）。"""
    records: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ContractError(f"manifest {path} line {lineno}: invalid JSON: {e}")
            if not isinstance(obj, dict):
                raise ContractError(f"manifest {path} line {lineno}: not an object")
            records.append(obj)
    return records


def write_jsonl_atomic(path: str, records: List[dict]) -> None:
    """原子写出 JSONL（先写临时文件再替换，避免半写状态）。"""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        prefix=".tmp_manifest_", dir=os.path.dirname(os.path.abspath(path))
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False, sort_keys=False) + "\n")
        os.replace(tmp, path)
    except Exception:
        if os.path.isfile(tmp):
            os.remove(tmp)
        raise


def write_text_atomic(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        prefix=".tmp_text_", dir=os.path.dirname(os.path.abspath(path))
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        os.replace(tmp, path)
    except Exception:
        if os.path.isfile(tmp):
            os.remove(tmp)
        raise


# ---------------------------------------------------------------------------
# 感知哈希（numpy + PIL 自实现，不引入新全局依赖；PIL 缺失时降级返回 None）
# ---------------------------------------------------------------------------

try:  # 可选依赖降级：无 PIL 时感知哈希不可用，精确 SHA256 仍然可用
    import numpy as np
    from PIL import Image

    _HAS_PIL = True
except Exception:  # pragma: no cover
    _HAS_PIL = False


def _to_gray_thumbnail(path: str, size: int):
    if not _HAS_PIL:
        return None
    with Image.open(path) as im:
        im = im.convert("L").resize((size, size), Image.BILINEAR)
        arr = np.asarray(im, dtype=np.float64)
    return arr


def ahash64(path: str) -> Optional[str]:
    """64bit 平均哈希（8x8 灰度缩略图，逐像素与均值比较）。返回 hex 字符串。"""
    arr = _to_gray_thumbnail(path, 8)
    if arr is None:
        return None
    bits = arr > arr.mean()
    return _bits_to_hex(bits.flatten())


def dhash64(path: str) -> Optional[str]:
    """64bit 差值哈希（9x8 灰度缩略图，横向相邻比较得 8x8=64bit）。"""
    if not _HAS_PIL:
        return None
    with Image.open(path) as im:
        im = im.convert("L").resize((9, 8), Image.BILINEAR)
        arr = np.asarray(im, dtype=np.float64)
    diff = arr[:, 1:] > arr[:, :-1]  # (8, 9) -> (8, 8) = 64 bit
    return _bits_to_hex(diff.flatten())


def _dct_matrix(n: int):
    """n 阶 DCT-II 变换矩阵（正交归一），用 numpy 实现，避免引入 scipy。"""
    k = np.arange(n, dtype=np.float64).reshape(-1, 1)
    r = np.arange(n, dtype=np.float64).reshape(1, -1)
    m = np.cos((2 * r + 1) * k * np.pi / (2 * n))
    m[0, :] *= 1.0 / np.sqrt(2.0)
    return m * np.sqrt(2.0 / n)


_DCT32 = None  # 惰性初始化


def phash64(path: str) -> Optional[str]:
    """64bit pHash（32x32 灰度 DCT，取左上 8x8 低频，中位数二值化）。"""
    global _DCT32
    if not _HAS_PIL:
        return None
    arr = _to_gray_thumbnail(path, 32)
    if arr is None:
        return None
    if _DCT32 is None:
        _DCT32 = _dct_matrix(32)
    dct = _DCT32 @ arr @ _DCT32.T
    low = dct[:8, :8].flatten()
    med = np.median(low[1:])  # 排除直流分量再取中位数
    return _bits_to_hex((low > med).flatten())


def _bits_to_hex(bits) -> str:
    # 守卫：感知哈希必须是精确 64bit，否则 hex 表示长度漂移，下游汉明距离会崩
    if len(bits) != PHASH_BITS:
        raise ContractError(
            f"perceptual hash must be exactly {PHASH_BITS} bits, got {len(bits)}"
        )
    v = 0
    for b in bits:
        v = (v << 1) | int(bool(b))
    return f"{v:016x}"


def hamming_hex(a: str, b: str) -> int:
    """两个 64bit hex 串的汉明距离。"""
    if len(a) != len(b):
        raise ContractError(f"hash length mismatch: {a} vs {b}")
    return bin(int(a, 16) ^ int(b, 16)).count("1")


def compute_perceptual_hashes(path: str) -> Dict[str, str]:
    out = {}
    for name, fn in (("ahash", ahash64), ("dhash", dhash64), ("phash", phash64)):
        v = fn(path)
        if v is not None:
            out[name] = v
    return out
