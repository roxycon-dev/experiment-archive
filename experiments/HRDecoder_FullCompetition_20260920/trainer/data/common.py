# -*- coding: utf-8 -*-
"""trainer.data.common — 数据层共享工具：契约常量导入、图像/标签读取、归一化、
padding、数据指纹（写入 checkpoint 的 data_manifest_sha256）。"""

from __future__ import annotations

import hashlib
import os
import sys

import numpy as np

from ..gate import GateError

# ---------------------------------------------------------------------------
# 契约常量：从 tools/contract_common.py 导入（单一权威定义，禁止重复硬编码）
# ---------------------------------------------------------------------------

_contract = None


def load_contract_common():
    """把 <repo>/tools 加入 sys.path 并导入 contract_common（带缓存）。"""
    global _contract
    if _contract is not None:
        return _contract
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    tools_dir = os.path.join(repo_root, "tools")
    if not os.path.isdir(tools_dir):
        raise GateError(f"找不到 tools 目录（contract_common 权威定义所在）: {tools_dir}")
    if tools_dir not in sys.path:
        sys.path.insert(0, tools_dir)
    try:
        import contract_common as cc  # noqa: PLC0415 — 工作区内部契约模块
    except Exception as e:
        raise GateError(f"导入 contract_common 失败: {e}") from e
    _contract = cc
    return cc


_cc = load_contract_common()
LESION_CHANNEL_ORDER = _cc.LESION_CHANNEL_ORDER            # ('EX','HE','SE','MA')
DDR_CLASS_INDEX = _cc.DDR_CLASS_INDEX                      # {'bg':0,'EX':1,'HE':2,'SE':3,'MA':4}
ContractError = _cc.ContractError

# 通道 i 对应 DDR fused label 的 class id = i+1（lesion_metric 对齐规则，审计文档第1节）
DDR_CLASS_ID_OF_CHANNEL = tuple(DDR_CLASS_INDEX[name] for name in LESION_CHANNEL_ORDER)  # (1,2,3,4)

IMG_EXTS = (".jpg", ".jpeg", ".png")
PAD_DIVISOR = 32  # H:HRNet 需要 /4；U:ConvNeXt 需要 /32 —— 统一取 32


# ---------------------------------------------------------------------------
# 读取
# ---------------------------------------------------------------------------

def read_image_rgb(path: str) -> np.ndarray:
    """读 RGB 图像（uint8 [H,W,3]）。文件缺失时 fail closed。"""
    from PIL import Image

    if not os.path.isfile(path):
        raise ContractError(f"image not found: {path}")
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


def read_mask_binary(path: str) -> np.ndarray:
    """读二值掩膜（uint8 [H,W]，非零=正样本，与契约一致）。"""
    from PIL import Image

    if not os.path.isfile(path):
        raise ContractError(f"mask not found: {path}")
    with Image.open(path) as im:
        arr = np.asarray(im)
    if arr.ndim == 3:  # 调色板/RGB 掩膜：任一通道非零即正（契约：非零=有效/阳性）
        arr = arr.max(axis=2)
    return (arr > 0).astype(np.uint8)


def read_ddr_fused_label(path: str) -> np.ndarray:
    """读 DDR fused PNG（mode L，值域 0..4，audit 实测无 255）。

    返回 [H,W] uint8 class id（0=bg, EX=1, HE=2, SE=3, MA=4）。
    值域超界立即报错（不猜语义）。
    """
    from PIL import Image

    if not os.path.isfile(path):
        raise ContractError(f"DDR fused label not found: {path}")
    with Image.open(path) as im:
        arr = np.asarray(im)
    if arr.ndim == 3:
        raise ContractError(f"DDR fused label 必须是单通道 L 图: {path}")
    if arr.max() > 4:
        raise ContractError(
            f"DDR fused label 值域超界（max={arr.max()}，期望 0..4）: {path}；"
            " refusing to guess label semantics"
        )
    return arr.astype(np.uint8)


def fused_label_to_targets(fused: np.ndarray) -> np.ndarray:
    """fused class-id 图 -> [4,H,W] float32 目标（通道 i = fused == i+1）。

    注意（审计文档第 6/7 节）：融合 PNG 是训练真值（后写类覆盖先写类），
    逐像素只属于一个类；逐类 TIF 仅存档不参与训练接口。
    """
    out = np.zeros((4, fused.shape[0], fused.shape[1]), dtype=np.float32)
    for c, class_id in enumerate(DDR_CLASS_ID_OF_CHANNEL):
        out[c] = (fused == class_id).astype(np.float32)
    return out


def sample_rng(worker_seed: int, idx: int, epoch: int):
    """逐样本增强 RNG：seed = f(worker_seed, idx, epoch)。

    必须带 epoch —— 旧实现只用 (initial_seed + idx)，同一 worker 内每个 epoch
    的增强完全相同，等价于"关掉了增强的随机性"（科学有效性问题）。
    """
    import numpy as np

    return np.random.RandomState(
        np.array(
            [
                int(worker_seed) % (2**32 - 1),
                int(idx) % (2**32 - 1),
                int(epoch) % (2**32 - 1),
            ],
            dtype=np.uint32,
        )
    )


def normalize_image(img: np.ndarray, mean, std) -> np.ndarray:
    """按配置 mean/std 归一化（uint8 -> float32，逐通道减均值除方差）。"""
    arr = img.astype(np.float32)
    mean = np.asarray(mean, dtype=np.float32).reshape(1, 1, 3)
    std = np.asarray(std, dtype=np.float32).reshape(1, 1, 3)
    if np.any(std <= 0):
        raise GateError(f"normalization std 必须为正: {std.flatten().tolist()}")
    return (arr - mean) / std


def pad_to_multiple(img: np.ndarray, targets: np.ndarray, valid: np.ndarray,
                    divisor: int = PAD_DIVISOR):
    """整图推理前把 H/W pad 到 divisor 的倍数（保持纵横比，绝不 resize）。

    target/valid 的 pad 区域置 0（valid=0 ⇒ 不参与损失与指标）。
    返回 (img, targets, valid, orig_hw)。
    """
    h, w = img.shape[:2]
    pad_h = (divisor - h % divisor) % divisor
    pad_w = (divisor - w % divisor) % divisor
    if pad_h == 0 and pad_w == 0:
        return img, targets, valid, (h, w)
    img_p = np.zeros((h + pad_h, w + pad_w, img.shape[2]), dtype=img.dtype)
    img_p[:h, :w] = img
    t_p = np.zeros((targets.shape[0], h + pad_h, w + pad_w), dtype=targets.dtype)
    t_p[:, :h, :w] = targets
    v_p = np.zeros_like(t_p)
    v_p[:, :h, :w] = valid
    return img_p, t_p, v_p, (h, w)


def crop_logits_to_orig(logits, orig_hw):
    """把整图 logits/preds 裁回原始尺寸（抵消 pad_to_multiple）。"""
    import torch

    h, w = orig_hw
    if torch.is_tensor(logits):
        return logits[..., :h, :w]
    return logits[..., :h, :w]


# ---------------------------------------------------------------------------
# 数据指纹（checkpoint.data_manifest_sha256）
# ---------------------------------------------------------------------------

def _sha256_lines(lines) -> str:
    h = hashlib.sha256()
    for line in sorted(lines):
        h.update(line.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def ddr_split_fingerprint(ddr_root: str, splits) -> str:
    """DDR 数据指纹：只覆盖给定 splits 的 images/labels 文件名+大小（绝不读 test）。

    指纹变化 ⇒ 训练数据集内容变化 ⇒ 旧 checkpoint 的统计环境失效。
    """
    lines = []
    for split in splits:
        for sub in ("images", "labels"):
            d = os.path.join(ddr_root, sub, split)
            if not os.path.isdir(d):
                raise ContractError(f"DDR 目录缺失: {d}")
            for f in os.listdir(d):
                p = os.path.join(d, f)
                if os.path.isfile(p):
                    lines.append(f"{sub}/{split}/{f}:{os.path.getsize(p)}")
    if not lines:
        raise ContractError(f"DDR 指纹为空（splits={splits}）: {ddr_root}")
    return _sha256_lines(lines)


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
