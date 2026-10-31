# -*- coding: utf-8 -*-
"""trainer.eval — 整图 / 滑窗验证路径（唯一合法的 val 指标来源）。

为什么必须存在这个模块：
验证指标只有在"整图还原 + 逐像素累计 TP/FP/FN"口径下才与最终 DDR test 评估
（`mmseg/core/evaluation/lesion_metric.py` 的 `sigmoid_confused_matrix`）可比。
用训练 crop 的指标冒充 val 会系统性高估召回（crop 由前景采样偏置），
因此正式训练入口在没有本模块时 **fail closed**（见 trainer.engine）。

约定：
- 输入图像绝不 resize：H/W 先 pad 到 32 的倍数（pad 区域 valid=0），
  推理后把 logits 裁回原始 H×W 再累计；
- pad 在 **归一化之后** 进行（pad 值 0 等效于补均值色），避免黑色边框
  扰动 HRNet 的 BN 统计；
- 滑窗模式下重叠区域对 logits 取算术平均（在概率之前平均，口径与 mmseg
  slide 推理一致）；窗口/步长必须由配置显式给出，不猜默认值。
"""

from __future__ import annotations

import numpy as np
import torch

from .gate import GateError
from .metrics import SegMetricsAccumulator
from .data.common import PAD_DIVISOR, crop_logits_to_orig, normalize_image

EVAL_MODES = ("whole_image", "sliding_window")


def pad_normalized_image(img_chw: np.ndarray, divisor: int = PAD_DIVISOR):
    """归一化后的 [3,H,W] float32 → pad 到 divisor 倍数；返回 (padded, orig_hw)。"""
    _, h, w = img_chw.shape
    ph = (divisor - h % divisor) % divisor
    pw = (divisor - w % divisor) % divisor
    if ph == 0 and pw == 0:
        return img_chw, (h, w)
    out = np.zeros((img_chw.shape[0], h + ph, w + pw), dtype=img_chw.dtype)
    out[:, :h, :w] = img_chw
    return out, (h, w)


def sliding_window_logits(model, image: torch.Tensor, window: int, stride: int) -> torch.Tensor:
    """对 [1,3,H,W] 执行滑窗推理，返回与输入同尺寸的 [1,4,H,W] logits。

    重叠区域按 logits 算术平均（用覆盖计数归一，保证边界不衰减）。
    """
    if window <= 0 or stride <= 0 or stride > window:
        raise GateError(f"滑窗参数非法: window={window} stride={stride}（要求 0 < stride <= window）")
    _, _, h, w = image.shape
    if h <= window and w <= window:
        return model.eval_forward(image)
    acc = None
    cnt = None
    ys = list(range(0, max(h - window, 0) + 1, stride)) or [0]
    xs = list(range(0, max(w - window, 0) + 1, stride)) or [0]
    if ys[-1] != h - window:
        ys.append(max(h - window, 0))
    if xs[-1] != w - window:
        xs.append(max(w - window, 0))
    for y0 in ys:
        for x0 in xs:
            patch = image[:, :, y0:y0 + window, x0:x0 + window]
            logit = model.eval_forward(patch)
            if logit.shape[-2:] != (window, window):
                raise RuntimeError(
                    f"滑窗输出 {tuple(logit.shape[-2:])} != 窗口 ({window},{window})"
                )
            if acc is None:
                acc = torch.zeros((1, logit.shape[1], h, w), dtype=torch.float32, device=logit.device)
                cnt = torch.zeros((1, 1, h, w), dtype=torch.float32, device=logit.device)
            acc[:, :, y0:y0 + window, x0:x0 + window] += logit.float()
            cnt[:, :, y0:y0 + window, x0:x0 + window] += 1.0
    return acc / torch.clamp(cnt, min=1.0)


class WholeImageEvaluator:
    """对 DDR val split 做整图/滑窗推理并累计指标。

    只接受"原始尺寸还原"的整图口径；`mode` 与窗口参数必须来自配置。
    """

    def __init__(self, model, dataset, *, device, mean, std, threshold: float = 0.5,
                 mode: str = "whole_image", window_size=None, window_stride=None,
                 max_images=None, progress_cb=None):
        if mode not in EVAL_MODES:
            raise GateError(f"未知 eval_mode={mode!r}（支持 {EVAL_MODES}）")
        if mode == "sliding_window":
            if not window_size or not window_stride:
                raise GateError(
                    "eval_mode=sliding_window 必须由配置显式给出 window_size / window_stride"
                    "（不猜默认值）"
                )
            if int(window_size) % PAD_DIVISOR != 0 or int(window_stride) % PAD_DIVISOR != 0:
                raise GateError(
                    f"window_size/window_stride 必须是 {PAD_DIVISOR} 的倍数，"
                    f"当前 {window_size}/{window_stride}（否则网络内部无法整除）"
                )
        self.model = model
        self.dataset = dataset
        self.device = device
        self.mean = list(map(float, mean))
        self.std = list(map(float, std))
        self.mode = mode
        self.window_size = None if window_size is None else int(window_size)
        self.window_stride = None if window_stride is None else int(window_stride)
        self.max_images = max_images
        self.progress_cb = progress_cb
        self.acc = SegMetricsAccumulator(threshold=threshold)
        self.n_skipped = 0

    @torch.no_grad()
    def run(self) -> dict:
        self.model.eval()
        n = len(self.dataset)
        limit = n if self.max_images is None else min(n, int(self.max_images))
        for i in range(limit):
            img_u8, target, valid, stem = self.dataset.load_raw(i)
            chw = normalize_image(img_u8, self.mean, self.std).transpose(2, 0, 1)
            padded, (h, w) = pad_normalized_image(np.ascontiguousarray(chw))
            x = torch.from_numpy(padded[None]).to(self.device).float()
            if self.mode == "whole_image":
                logits = self.model.eval_forward(x)
            else:
                logits = sliding_window_logits(self.model, x, self.window_size, self.window_stride)
            logits = crop_logits_to_orig(logits, (h, w))     # 还原到原始尺寸
            probs = torch.sigmoid(logits).float().cpu().numpy()
            self.acc.update(probs, target[None], valid[None])
            if self.progress_cb is not None:
                self.progress_cb(i + 1, limit, stem)
        report = self.acc.compute()
        report.update({
            "split": getattr(self.dataset, "split", "val"),
            "mode": self.mode,
            "eval_images": limit,
            "dataset_size": n,
            "eval_inference": "whole_image" if self.mode == "whole_image" else (
                f"sliding_window(window={self.window_size},stride={self.window_stride})"
            ),
        })
        return report


def require_whole_image_evaluator(evaluator, *, mode: str) -> None:
    """正式训练入口的 fail-closed 断言：没有整图 evaluator 就不许开训。"""
    if evaluator is None:
        raise GateError(
            "正式训练缺少整图验证 evaluator：禁止用训练 crop 的指标冒充 val。"
            "请提供 trainer.eval.WholeImageEvaluator（或先在 calibration 模式跑通）。"
        )
    if mode == "formal" and not isinstance(evaluator, WholeImageEvaluator):
        raise GateError(
            f"formal 模式的 evaluator 必须是 WholeImageEvaluator，当前 {type(evaluator).__name__}"
        )
