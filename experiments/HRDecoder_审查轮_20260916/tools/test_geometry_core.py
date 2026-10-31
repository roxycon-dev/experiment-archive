#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""§7.2 平铺几何边界测试（prompt 要求）—— 纯 numpy/torch，不依赖 mmcv。

被测对象：hrplus_core.tiled_forward / _axis_positions / blend_probs / resize_logit

覆盖 prompt §7.2 的全部验收项：
  G1  边长 1,767,768,769,1023,1024,1025,1792,1793,2048,2049（tile=1024,stride=768）
  G2  非正方形 769x1025 / 1025x769 / 513x1501
  G3  常数预测经任意切块/重叠后仍是该常数
  G4  全局坐标 ramp 的 fake predictor 重建正确
  G5  单个亮点位于四角 / 边界 / 重叠区 / 最后一个 tile 边缘，输出保留且不扩散
  G6  权重覆盖图每个有效像素严格 > 0；tile 大于图像时 padding 不混入原图
  G7  无负坐标、无空 tile、无遗漏像素、无 off-by-one（由 G1/G4/G5 覆盖）
  G8  blend_probs 的代数性质（权重 3 vs 1 归一相同；同模型复制等价；常数缩放不变；非法权重拒绝）
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, r"D:\dataset\experiment\hrdecoder_plus3\repo")
from hrplus_core import tiled_forward, _axis_positions, blend_probs  # noqa

CLASSES = 4
REQUIRED = ["G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8"]


def fake_const(c):
    def fn(patch):
        b = patch.shape[0]
        return torch.full((b, CLASSES, patch.shape[2], patch.shape[3]),
                          float(c), dtype=torch.float32)
    return fn


def fake_ramp(h, w):
    """返回一个 forward_fn：输出 = 全局坐标 ramp（tile 内的局部坐标 + tile 原点）。
    通过闭包捕获 tile 原点不可行，改为让 forward_fn 接收 (y0,x0) 的替代实现：
    这里用一个全局可写容器记录最近的 patch 原点。"""
    state = {"origin": [0, 0]}

    def fn(patch):
        y0, x0 = state["origin"]
        b = patch.shape[0]
        yy = torch.arange(patch.shape[2], dtype=torch.float32).view(1, 1, -1, 1) + y0
        xx = torch.arange(patch.shape[3], dtype=torch.float32).view(1, 1, 1, -1) + x0
        r = yy.expand(b, CLASSES, patch.shape[2], patch.shape[3]) * 1000.0 + xx
        return r
    return fn, state


def run_case(size, tile, stride, cases):
    h, w = size
    img = torch.zeros((1, 3, h, w), dtype=torch.float32)
    out = tiled_forward(img, fake_const(3.0), tile, stride, CLASSES)
    # G3 常数
    const_ok = bool(torch.allclose(out, torch.full_like(out, 3.0)))
    # G4 ramp：输入是全局坐标 ramp，预测器为恒等映射（通道 0 复制到 4 通道）。
    # 平铺+重叠平均对"逐像素恒等"的信号必须无损 —— 这才是可检验的等变性。
    img_r = torch.zeros((1, 3, h, w), dtype=torch.float32)
    yy = torch.arange(h, dtype=torch.float32).view(-1, 1)
    xx = torch.arange(w, dtype=torch.float32).view(1, -1)
    ramp = (yy * 10.0 + xx).expand(CLASSES, h, w).unsqueeze(0)   # (1,4,H,W)
    img_r[0] = ramp[0, :3]
    ident = lambda p: p[:, 0:1].expand(-1, CLASSES, -1, -1).contiguous().float()
    out2 = tiled_forward(img_r, ident, tile, stride, CLASSES)
    ref = ramp.contiguous()
    ramp_ok = bool(torch.allclose(out2, ref, atol=1e-2))
    # G5 亮点
    spot_ok = True
    spots = [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1), (h // 2, w // 2),
             (h - 1, w // 2), (0, w // 2), (h // 2, w - 1)]
    for (sy, sx) in spots:
        img2 = torch.zeros((1, 3, h, w), dtype=torch.float32)
        img2[0, :, sy, sx] = 1.0
        o = tiled_forward(
            img2,
            lambda p: p[:, 0:1].expand(-1, CLASSES, -1, -1).contiguous().float(),
            tile, stride, CLASSES)
        # 常数预测下亮点应原位保留：输出在 (sy,sx) 处的 4 通道均非零，
        # 且不在 (sy,sx) 的像素输出为 0（identity predictor）
        if not (o[0, :, sy, sx].abs().sum() > 0):
            spot_ok = False
        mask = torch.ones(h, w, dtype=torch.bool); mask[sy, sx] = False
        if mask.any() and float(o[0, :, mask].abs().max()) != 0.0:
            spot_ok = False
    # G6 覆盖
    cov_ok = True
    th2, tw2 = min(tile, h), min(tile, w)
    cnt2 = torch.zeros((1, 1, h, w))
    for y in _axis_positions(h, tile, stride):
        for x in _axis_positions(w, tile, stride):
            cnt2[:, :, y:y + th2, x:x + tw2] += 1
    if not bool((cnt2 > 0).all()):
        cov_ok = False
    if h < tile or w < tile:
        # tile 大于图像：padding 区不得混入 —— 用常数预测 + 尺寸校验
        if out.shape != (1, CLASSES, h, w):
            cov_ok = False
    return const_ok, ramp_ok, spot_ok, cov_ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    sizes = [(1, 1), (767, 767), (768, 768), (769, 769), (1023, 1023),
             (1024, 1024), (1025, 1025), (1792, 1792), (1793, 1793),
             (2048, 2048), (2049, 2049),
             (769, 1025), (1025, 769), (513, 1501)]
    tile, stride = 1024, 768

    rows, fails = [], []
    for (h, w) in sizes:
        c, r, s, v = run_case((h, w), tile, stride, None)
        ok = c and r and s and v
        rows.append({"case": "G1/G2 %dx%d" % (h, w), "const": c, "ramp": r,
                     "spot": s, "coverage": v, "status": "PASS" if ok else "FAIL"})
        print("  %-22s const=%s ramp=%s spot=%s cov=%s  %s"
              % ("%dx%d" % (h, w), c, r, s, v, "PASS" if ok else "FAIL"))
        if not ok:
            fails.append("%dx%d" % (h, w))

    # G8 blend 代数性质
    a = torch.rand(1, CLASSES, 5, 5)
    b = torch.rand(1, CLASSES, 5, 5)
    g8 = []
    g8.append(("权重1与3归一相同", bool(torch.allclose(
        blend_probs(a, b, 1 / 4.0), blend_probs(a, b, 3 / 12.0)))))
    g8.append(("同模型复制等价", bool(torch.allclose(
        blend_probs(a, a, 0.5), a))))
    # blend 对输入是线性的：缩放输入 k 倍 => 输出也缩放 k 倍（正 k）
    g8.append(("输入线性（缩放 7 倍）", bool(torch.allclose(
        blend_probs(a * 7.0, b * 7.0, 0.5), blend_probs(a, b, 0.5) * 7.0))))
    # prompt §7.4 要求"负权重拒绝"。blend_probs 只断言形状 —— 这是一个真实缺口
    try:
        blend_probs(a, b, -0.1); neg_ok = False
    except AssertionError:
        neg_ok = True
    g8.append(("负权重拒绝（§7.4 要求）", neg_ok))
    for d, ok in g8:
        rows.append({"case": "G8 %s" % d, "status": "PASS" if ok else "FAIL",
                     "note": "" if d != "负权重拒绝（§7.4 要求）" else
                     "真实缺口：blend_probs 不拒绝负权重；ops_eval_ens.py 也不校验"
                     "权重符号/总和。修复位置应在评估器的权重解析处。"})
        print("  G8 %-24s %s" % (d, "PASS" if ok else "FAIL"))
        if not ok:
            fails.append("G8 " + d)

    overall = "PASS" if not fails else "FAIL"
    out = {"suite": "geometry_tiling", "tile": tile, "stride": stride,
           "n_cases": len(rows), "overall_status": overall, "results": rows}
    p = os.path.join(args.out_dir, "geometry_tta_ensemble.json")
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    os.replace(tmp, p)
    print("\n%d/%d PASS overall=%s" % (len(rows) - len(fails), len(rows), overall))
    print("写出:", p)
    return 0 if overall == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
