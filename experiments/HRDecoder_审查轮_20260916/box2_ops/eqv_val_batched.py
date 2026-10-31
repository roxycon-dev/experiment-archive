#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""等价性测试：val 推理"逐窗口(batch=1)" vs "批量化" 是否逐位一致 + 提速多少。

背景：val 每点耗时约 5 分钟，瓶颈是 76 图 × 每图 64 个 512 窗口 = **4864 次 batch=1 的前向**。
把同一图的窗口堆成 batch 能显著提速，但**必须先证明数值等价**，否则 T0−T1 / U0−T1 的对照失效。

本脚本对两条路径做严格对比：
  A) pertile  = 现有实现（逐窗口、每次 batch=1），**逐字复刻** train_T0T1.py 的 val_eval
  B) batched  = 同一数学、把 chunk 个窗口堆成一个 batch 一次前向

判定（任一不满足即 FAIL）：
  1) chunk=1 时 B 与 A **逐位完全相同**（自证我的复刻没走样）
  2) chunk>1 时 概率图 max|Δ| 极小、且 **TP/FP/FN 与 MA-IoU 完全相等**
  3) 两条路径的 mask 决策（prob > 0.5）逐像素相同

计时方法：先把 76 张 val 的 RGB 全部预热进缓存（稳态；对应训练中 val@1000 之后的真实情况），
再分别计时 A/B，避免"谁后跑谁快"的缓存偏置。

用法（2 号机）：
  cd /root/box1_sync/repo && python /root/box1_sync/ops/eqv_val_batched.py \
     --ckpt /root/box1_sync/head_T0_prepatch_20260917.pth --chunk 16
"""
import argparse
import copy
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

TAU = 0.5
WIN = 512
CEN = 256
HALF = WIN // 2
CLAMP = 1e-6
REPO = os.environ.get("HR_REPO", "/root/box1_sync/repo")


def lg(p):
    q = np.clip(p, CLAMP, 1 - CLAMP)
    return np.log(q / (1 - q)).astype(np.float32)


class Head(nn.Module):
    """与 train_T0T1.py 中的 Head 逐字相同（7->16->16->1，末层零初始化）。

    ⚠ 2026-09-17 子代理审计抓到：初版只抄了 __init__、**漏掉 forward**
    （原实现 `return 6.0 * torch.tanh(self.net(f))`）→ 会崩溃；若补成
    `return self.net(f)` 则两条路径仍互相一致 → 变成假 PASS。这里逐字补齐。
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(7, 16, 3, 1, 1), nn.ReLU(inplace=True),
                                 nn.Conv2d(16, 16, 3, 1, 1), nn.ReLU(inplace=True),
                                 nn.Conv2d(16, 1, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, f):
        return 6.0 * torch.tanh(self.net(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/lesion/exp/hrplus_patch_dice.py")
    ap.add_argument("--val-cache", default="/dev/shm/hrdec_cache_val")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--bitwise-images", type=int, default=3,
                    help="用 chunk=1 做逐位自证的图像数")
    ap.add_argument("--out", default="/root/box1_sync/workdir/eqv_val_batched.json")
    ap.add_argument("--repeat", type=int, default=2,
                    help="A/B 各跑几遍取中位数（配合 warmup 消除顺序偏置）")
    ap.add_argument("--ref-counts", default="",
                    help="1 号机真实实现产出的 val_counts_*.json；提供后做端到端对账（最强证据）")
    a = ap.parse_args()

    sys.path.insert(0, REPO)
    os.chdir(REPO)
    from mmcv import Config
    from mmseg.datasets import build_dataset
    import cv2

    cfg = Config.fromfile(a.config)
    dva = copy.deepcopy(cfg.data["val"])
    dva.test_mode = True
    ds_va = build_dataset(dva)
    gt_va = ds_va.get_gt_seg_maps()
    n_va = len(ds_va)
    print("[eqv] val = %d" % n_va, flush=True)

    ck = torch.load(a.ckpt, map_location="cpu")
    sd = ck["head"] if isinstance(ck, dict) and "head" in ck else ck
    head = Head().cuda()
    head.load_state_dict(sd, strict=True)       # 架构自证：键不匹配即报错
    head.eval()
    print("[eqv] head 载入 %s（%d 键，strict=True 通过）" % (a.ckpt, len(sd)), flush=True)

    # ---- 与 train_T0T1.py 逐字相同的 RGB 取值链 ----
    def rgb_at(i, cache):
        if i in cache:
            return cache[i]
        sample = ds_va[i]
        raw = sample["img"]
        n = 0
        while hasattr(raw, "data") and n < 16:
            raw = raw.data; n += 1
        while isinstance(raw, (list, tuple)):
            raw = raw[0]
            while hasattr(raw, "data") and n < 20:
                raw = raw.data; n += 1
        arr = (raw.numpy() if torch.is_tensor(raw) else np.asarray(raw)).astype(np.float32)
        m = sample["img_metas"]
        n = 0
        while hasattr(m, "data") and n < 16:
            m = m.data; n += 1
        if isinstance(m, (list, tuple)):
            m = m[0]
            while hasattr(m, "data") and n < 20:
                m = m.data; n += 1
        ori = tuple(m["ori_shape"][:2])
        H, W = ori
        if arr.shape[1] == H and arr.shape[2] == W:
            cache[i] = arr
            return arr
        out = np.empty((3, H, W), dtype=np.float32)
        for c in range(3):
            out[c] = cv2.resize(arr[c], (W, H), interpolation=cv2.INTER_LINEAR)
        cache[i] = out
        return out

    def tk(fld, fill, y0, x0, H, W):
        o = np.full((WIN, WIN), fill, dtype=np.float32)
        sy0, sy1 = max(0, y0), min(H, y0 + WIN)
        sx0, sx1 = max(0, x0), min(W, x0 + WIN)
        if sy1 > sy0 and sx1 > sx0:
            o[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = fld[sy0:sy1, sx0:sx1]
        return o

    def tiles_of(H, W):
        ys = list(range(0, max(H - CEN + 1, 1), CEN)) or [0]
        xs = list(range(0, max(W - CEN + 1, 1), CEN)) or [0]
        out = []
        for y in ys:
            for x in xs:
                ay = min(y + CEN // 2, H - 1)
                ax = min(x + CEN // 2, W - 1)
                out.append((y, x, ay - HALF, ax - HALF))
        return out

    def build_feat(vi, b0v, psv, prv, im, y0, x0, H, W):
        fb = tk(b0v, 0.0, y0, x0, H, W)
        fs = tk(psv, 0.0, y0, x0, H, W)
        fr = tk(prv, 0.0, y0, x0, H, W)
        rgbw = np.empty((3, WIN, WIN), dtype=np.float32)
        for c in range(3):
            rgbw[c] = tk(im[c], 0.0, y0, x0, H, W)
        feat = np.concatenate([rgbw, lg(fb)[None], lg(fs)[None], lg(fr)[None],
                               (fs - fr)[None]], axis=0)
        return feat, fb

    cy0 = HALF - CEN // 2

    def run_image(vi, rgbcache, chunk, head):
        """返回该图的 head 概率图（float32, ori 尺寸）。chunk=1 即现有实现。"""
        g = np.asarray(gt_va[vi]).astype(np.int64)
        H, W = g.shape
        b0v = np.load(os.path.join(a.val_cache, "base0", "%04d.npy" % vi))
        psv = np.load(os.path.join(a.val_cache, "ps", "%04d.npy" % vi))
        prv = np.load(os.path.join(a.val_cache, "pr", "%04d.npy" % vi))
        im = rgb_at(vi, rgbcache)
        prob_h = b0v.copy()
        tl = tiles_of(H, W)
        with torch.no_grad():
            for s in range(0, len(tl), chunk):
                blk = tl[s:s + chunk]
                feats, fbs = [], []
                for (y, x, y0, x0) in blk:
                    feat, fb = build_feat(vi, b0v, psv, prv, im, y0, x0, H, W)
                    feats.append(feat); fbs.append(fb)
                inp = torch.from_numpy(np.stack(feats)).cuda()
                dc = head(inp)[:, 0, cy0:cy0 + CEN, cy0:cy0 + CEN].cpu().numpy()
                for k, (y, x, y0, x0) in enumerate(blk):
                    # 与原实现一致：zc 在 GPU 上与 dc 相加后 torch.sigmoid（不用 numpy 版）
                    zc = torch.from_numpy(lg(fbs[k])[cy0:cy0 + CEN, cy0:cy0 + CEN]).cuda()
                    dck = torch.from_numpy(dc[k]).cuda()
                    pn = torch.sigmoid(zc + dck).cpu().numpy()
                    cy = min(y, max(H - CEN, 0)); cx = min(x, max(W - CEN, 0))
                    ye = min(cy + CEN, H); xe = min(cx + CEN, W)
                    prob_h[cy:ye, cx:xe] = pn[:ye - cy, :xe - cx]
        return prob_h, b0v

    def counts(prob, b0v, g):
        gg = (g == 4); ig = (g == 255)
        out = {}
        for tag, pr in (("base0", b0v), ("head", prob)):
            p = pr > TAU
            out[tag] = [int((p & gg & ~ig).sum()), int((p & ~gg & ~ig).sum()),
                        int((~p & gg).sum())]
        return out

    # ---- 1) RGB 预热（稳态）----
    t0 = time.time()
    rgbcache = {}
    for vi in range(n_va):
        rgb_at(vi, rgbcache)
    t_warm = time.time() - t0
    print("[eqv] RGB 预热 %d 图 %.1f s" % (n_va, t_warm), flush=True)

    # ---- 2) 确定性自证：pertile 路径跑两遍必须逐位相同（否则任何"差异"都无法归因）----
    det_ok = True
    for vi in range(min(a.bitwise_images, n_va)):
        p1, _ = run_image(vi, rgbcache, 1, head)
        p2, _ = run_image(vi, rgbcache, 1, head)
        same = bool(np.array_equal(p1, p2))
        det_ok = det_ok and same
        print("[eqv] 图%d pertile 两遍逐位相同=%s" % (vi, same), flush=True)
    print("[eqv] 确定性自证结论: %s" % det_ok, flush=True)
    if not det_ok:
        print("[eqv] !! pertile 路径本身不确定 → 无法归因差异，直接判 FAIL", flush=True)

    # ---- 3) 全量对比 + 计时（先 warmup + A/B 各跑 REPEAT 遍取中位数，消除顺序偏置）----
    print("[eqv] warmup forward（消除 cuDNN/allocator 冷启动偏置）", flush=True)
    with torch.no_grad():
        _w = torch.zeros(1, 7, WIN, WIN).cuda()
        head(_w)
    torch.cuda.synchronize()

    REPEAT = max(1, a.repeat)

    def time_pass(chunk):
        ts = []
        for _ in range(REPEAT):
            torch.cuda.synchronize(); t = time.time()
            agg = {"base0": [0, 0, 0], "head": [0, 0, 0]}
            for vi in range(n_va):
                p, b0v = run_image(vi, rgbcache, chunk, head)
                g = np.asarray(gt_va[vi]).astype(np.int64)
                c = counts(p, b0v, g)
                for k in ("base0", "head"):
                    for i3 in range(3):
                        agg[k][i3] += c[k][i3]
            torch.cuda.synchronize(); ts.append(time.time() - t)
        return float(np.median(ts)), ts, agg

    tA, tsA, aggA = time_pass(1)
    tB, tsB, aggB = time_pass(a.chunk)

    res = {"n_val": n_va, "chunk": a.chunk, "ckpt": a.ckpt,
           "rgb_warm_s": round(t_warm, 1), "repeat": REPEAT,
           "times_pertile_s": [round(x, 1) for x in tsA],
           "times_batched_s": [round(x, 1) for x in tsB],
           "per_image": []}
    maxdiff = 0.0
    ndiff_total = 0
    mask_mismatch = 0
    for vi in range(n_va):
        pA, b0v = run_image(vi, rgbcache, 1, head)
        pB, _ = run_image(vi, rgbcache, a.chunk, head)
        g = np.asarray(gt_va[vi]).astype(np.int64)
        d = np.abs(pA - pB)
        md = float(d.max()); nd = int((d > 0).sum())
        mm = int(((pA > TAU) != (pB > TAU)).sum())
        maxdiff = max(maxdiff, md); ndiff_total += nd; mask_mismatch += mm
        cA = counts(pA, b0v, g); cB = counts(pB, b0v, g)
        res["per_image"].append({"img": vi, "max_abs_diff": md, "n_diff": nd,
                                 "mask_mismatch_px": mm,
                                 "head_tp_fp_fn_A": cA["head"], "head_tp_fp_fn_B": cB["head"]})
        if vi % 20 == 0:
            print("[eqv] 对比 %d/%d  max|Δ|=%.2e mask 不一致=%d" % (vi, n_va, md, mm), flush=True)

    # ---- 4) 与 1 号机真实实现的记录对账（端到端等价证据，最强的一条）----
    ref = None
    if a.ref_counts and os.path.exists(a.ref_counts):
        d = json.load(open(a.ref_counts))
        rows = d["per_image"]
        rt = [0, 0, 0]
        for r in rows:
            for i3 in range(3):
                rt[i3] += r["head"][i3]
        ref = {"file": a.ref_counts, "ref_head_tp_fp_fn": rt,
               "mine_head_tp_fp_fn": aggA["head"],
               "exact_match": (rt == aggA["head"]),
               "n_val_ref": d.get("n_val")}
        print("[eqv] 与 1 号机记录对账：ref=%s mine=%s 完全一致=%s"
              % (rt, aggA["head"], rt == aggA["head"]), flush=True)

    def iou(t):
        den = t[0] + t[1] + t[2]
        return t[0] / den if den else 0.0

    res.update({
        "time_pertile_s": round(tA, 1), "time_batched_s": round(tB, 1),
        "speedup": round(tA / tB, 2) if tB > 0 else None,
        "max_abs_diff": maxdiff, "n_diff_elems": ndiff_total,
        "mask_mismatch_px_total": mask_mismatch,
        "agg_A": aggA, "agg_B": aggB,
        "MA_IoU_A": {"base0": iou(aggA["base0"]), "head": iou(aggA["head"])},
        "MA_IoU_B": {"base0": iou(aggB["base0"]), "head": iou(aggB["head"])},
        "ref_reconciliation": ref,
    })
    same_agg = (aggA == aggB)
    ref_ok = True if ref is None else bool(ref["exact_match"])
    verdict = "PASS" if (det_ok and same_agg and mask_mismatch == 0 and ref_ok) else "FAIL"
    res["verdict"] = verdict
    res["checks"] = {
        "agg_TP_FP_FN_identical": same_agg,
        "mask_decisions_identical": mask_mismatch == 0,
        "pertile_deterministic_selfcheck": det_ok,
        "matches_box1_recorded_counts": ref_ok if ref is not None else "未提供 --ref-counts（无端到端对账）",
    }
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)

    print("=" * 70)
    print("val 路径等价性：%s" % verdict)
    print("  逐窗口 %.1f s  vs  批量(chunk=%d) %.1f s  ->  提速 %.2fx"
          % (tA, a.chunk, tB, (tA / tB) if tB else 0))
    print("  max|Δp|=%.3e  不同元素=%d  mask 决策不一致像素=%d" % (maxdiff, ndiff_total, mask_mismatch))
    print("  TP/FP/FN 聚合完全一致=%s" % same_agg)
    print("  MA-IoU  A: base0=%.6f head=%.6f | B: base0=%.6f head=%.6f"
          % (iou(aggA["base0"]), iou(aggA["head"]), iou(aggB["base0"]), iou(aggB["head"])))
    print("  → 每次 val 点省 %.1f 分钟；每臂 6 个 val 点省 %.1f 分钟"
          % ((tA - tB) / 60, (tA - tB) * 6 / 60))
    print("  证据: %s" % a.out)
    sys.exit(0 if verdict == "PASS" else 1)


if __name__ == "__main__":
    main()
