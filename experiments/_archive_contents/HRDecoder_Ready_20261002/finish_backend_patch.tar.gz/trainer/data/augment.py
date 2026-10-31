# -*- coding: utf-8 -*-
"""trainer.data.augment — DDR 与外部数据共享的几何/光度增强。

实现 configs augmentation 节语义（对齐旧 hr_ddr_2048.py pipeline 的意图，
但不复用旧 mmseg transform 代码）：
1. RandomResize：fit-in 2048 的等比缩放 × uniform(ratio_range)（保持纵横比）；
2. RandomCrop（含 cat_max_ratio：单类像素占比超限则重采样，最多 10 次；
   foreground-mixed 混合采样：以概率 fg_prob 要求裁窗含 ≥1 个病灶像素）；
3. RandomHFlip；RandomRotate（±rotate_degrees，重采样双线性/最近邻，0 填充）；
4. photometric 'light'：亮度/对比度 ±10% 抖动（只作用于图像，不碰标签）。

几何变换对 image / 4 类 target / valid 掩码严格共享同一参数（逐样本一致）。
所有随机性走 numpy RandomState（由 DataLoader worker 播种，可复现）。
"""

from __future__ import annotations

import cv2
import numpy as np

from ..gate import GateError


class TrainAugment:
    """训练期增强（几何共享 + 轻量光度）。"""

    def __init__(
        self,
        *,
        crop_size: int,
        resize_long_side: int = 2048,
        resize_long_sides=None,
        resize_ratio_range=(0.5, 2.0),
        cat_max_ratio: float = 0.75,
        flip_ratio: float = 0.5,
        rotate_degrees=(-45.0, 45.0),
        photometric: str = "light",
        fg_prob: float = 0.5,
        max_cat_retry: int = 10,
        max_fg_retry: int = 8,
    ):
        if crop_size <= 0:
            raise GateError(f"crop_size 必须 > 0: {crop_size}")
        lo, hi = float(resize_ratio_range[0]), float(resize_ratio_range[1])
        if not (0 < lo <= hi):
            raise GateError(f"resize_ratio_range 非法: {resize_ratio_range}")
        if rotate_degrees is not None and rotate_degrees[0] > rotate_degrees[1]:
            raise GateError(f"rotate_degrees 顺序非法: {rotate_degrees}")
        self.crop_size = int(crop_size)
        self.resize_long_side = int(resize_long_side)
        self.resize_long_sides = tuple(resize_long_sides or [self.resize_long_side])
        self.ratio_lo, self.ratio_hi = lo, hi
        self.cat_max_ratio = float(cat_max_ratio)
        self.flip_ratio = float(flip_ratio)
        self.rotate_degrees = None if rotate_degrees is None else (float(rotate_degrees[0]), float(rotate_degrees[1]))
        self.photometric = photometric
        self.fg_prob = float(fg_prob)
        self.max_cat_retry = int(max_cat_retry)
        self.max_fg_retry = int(max_fg_retry)

    # ------------------------------------------------------------------
    def __call__(self, image: np.ndarray, targets: np.ndarray, valid: np.ndarray, rng: np.random.RandomState):
        """对一张样本施加增强。image uint8[H,W,3]；targets/valid float32[4,H,W]。"""
        img, tgt, vld = self._random_resize(image, targets, valid, rng)
        img, tgt, vld = self._random_crop(img, tgt, vld, rng)
        img, tgt, vld = self._random_flip(img, tgt, vld, rng)
        img, tgt, vld = self._random_rotate(img, tgt, vld, rng)
        img = self._photometric(img, rng)
        return img, tgt, vld

    # ------------------------------------------------------------------
    def _random_resize(self, img, tgt, vld, rng):
        h, w = img.shape[:2]
        side = self.resize_long_sides[rng.randint(len(self.resize_long_sides))] if len(self.resize_long_sides)>1 else self.resize_long_sides[0]
        fit = min(side / h, side / w)
        factor = fit * rng.uniform(self.ratio_lo, self.ratio_hi)
        nh, nw = max(1, int(round(h * factor))), max(1, int(round(w * factor)))
        if (nh, nw) == (h, w):
            return img, tgt, vld
        img_r = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        tgt_r = _resize_masks(tgt, nh, nw)
        vld_r = _resize_masks(vld, nh, nw)
        return img_r, tgt_r, vld_r

    def _random_crop(self, img, tgt, vld, rng):
        cs = self.crop_size
        h, w = img.shape[:2]
        if h < cs or w < cs:
            img, tgt, vld = _pad_bottom_right(img, tgt, vld, cs)
            h, w = img.shape[:2]
        # fg-anchored：以概率 fg_prob 要求裁窗含 ≥1 个病灶像素（random_plus_foreground_mixed）
        want_fg = rng.random() < self.fg_prob
        crop_yx = None
        for _ in range(self.max_fg_retry if want_fg else 1):
            y0 = rng.randint(0, h - cs + 1)
            x0 = rng.randint(0, w - cs + 1)
            if not want_fg or float(tgt[:, y0:y0 + cs, x0:x0 + cs].sum()) > 0.0:
                crop_yx = (y0, x0)
                break
        if crop_yx is None:  # fg 重试用尽：退回随机（不拒绝样本）
            crop_yx = (rng.randint(0, h - cs + 1), rng.randint(0, w - cs + 1))
        y0, x0 = crop_yx
        # cat_max_ratio：单类像素占比超限则重采样（对齐 mmseg 语义，最多 10 次）
        if self.cat_max_ratio < 1.0:
            for _ in range(self.max_cat_retry):
                patch = tgt[:, y0:y0 + cs, x0:x0 + cs] > 0.5
                share = patch.reshape(4, -1).sum(axis=1) / float(cs * cs)
                if float(share.max()) <= self.cat_max_ratio:
                    break
                y0 = rng.randint(0, h - cs + 1)
                x0 = rng.randint(0, w - cs + 1)
        return (img[y0:y0 + cs, x0:x0 + cs], tgt[:, y0:y0 + cs, x0:x0 + cs],
                vld[:, y0:y0 + cs, x0:x0 + cs])

    def _random_flip(self, img, tgt, vld, rng):
        if rng.random() < self.flip_ratio:
            img = img[:, ::-1]
            tgt = tgt[:, :, ::-1]
            vld = vld[:, :, ::-1]
        return np.ascontiguousarray(img), np.ascontiguousarray(tgt), np.ascontiguousarray(vld)

    def _random_rotate(self, img, tgt, vld, rng):
        if self.rotate_degrees is None:
            return img, tgt, vld
        angle = rng.uniform(self.rotate_degrees[0], self.rotate_degrees[1])
        if abs(angle) < 1e-6:
            return img, tgt, vld
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
        img_r = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        M_t = M.copy()
        tgt_r = np.zeros_like(tgt)
        vld_r = np.zeros_like(vld)
        for c in range(tgt.shape[0]):
            tgt_r[c] = cv2.warpAffine(tgt[c], M_t, (w, h), flags=cv2.INTER_NEAREST,
                                      borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            vld_r[c] = cv2.warpAffine(vld[c], M_t, (w, h), flags=cv2.INTER_NEAREST,
                                      borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return img_r, tgt_r, vld_r

    def _photometric(self, img, rng):
        if self.photometric in ("none", "off", None, ""):
            return img
        if self.photometric != "light":
            raise GateError(f"未知 photometric 模式: {self.photometric!r}（实现: none/light）")
        alpha = 1.0 + rng.uniform(-0.1, 0.1)   # 对比度
        beta = rng.uniform(-0.1, 0.1) * 255.0  # 亮度
        out = img.astype(np.float32) * alpha + beta
        return np.clip(out, 0, 255).astype(np.uint8)


def eval_sample(image: np.ndarray, targets: np.ndarray, valid: np.ndarray):
    """验证/整图口径：不做任何几何改变（绝不 resize），只保证可进网络前
    由调用方决定 pad（pad_to_multiple）。"""
    return image, targets, valid


# ---------------------------------------------------------------------------
# 内部工具
# ---------------------------------------------------------------------------

def _resize_masks(masks: np.ndarray, nh: int, nw: int) -> np.ndarray:
    out = np.zeros((masks.shape[0], nh, nw), dtype=masks.dtype)
    for c in range(masks.shape[0]):
        # 最近邻保 0/1 语义（valid/target 绝不能被插值成灰值）
        out[c] = cv2.resize(masks[c], (nw, nh), interpolation=cv2.INTER_NEAREST)
    return out


def _pad_bottom_right(img, tgt, vld, cs: int):
    """图像/掩膜右下 pad 到 ≥ crop_size（图像 0 填充，valid pad 区域置 0）。"""
    h, w = img.shape[:2]
    ph, pw = max(0, cs - h), max(0, cs - w)
    if ph == 0 and pw == 0:
        return img, tgt, vld
    img_p = np.zeros((h + ph, w + pw, img.shape[2]), dtype=img.dtype)
    img_p[:h, :w] = img
    tgt_p = np.zeros((tgt.shape[0], h + ph, w + pw), dtype=tgt.dtype)
    tgt_p[:, :h, :w] = tgt
    vld_p = np.zeros((vld.shape[0], h + ph, w + pw), dtype=vld.dtype)
    vld_p[:, :h, :w] = vld
    return img_p, tgt_p, vld_p
