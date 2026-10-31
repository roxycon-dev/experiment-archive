#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""T0 / T1：共同几何 + 两种采样的受控对照（契约 §6 + §8）。

共同规格（T0/T1 完全一致，唯一差异 = 采样）：
  输入 512x512，loss 仅中心 256x256，周围 128 上下文；
  特征 7 通道 = [R,G,B,logit(P_BASE0_MA),logit(P_student_MA),logit(P_reference_MA),P_s-P_r]；
  小头 7->16->16->1（两个 Conv3x3+ReLU + Conv1x1 零初始化）；delta = 6*tanh(head)；
  z_new = logit(clamp(P_BASE0_MA)) + delta；P_new = sigmoid(z_new)；
  非 MA 三类始终取 BASE_0；loss = 0.5*BCEWithLogits + 0.5*Dice（仅中心有效区）；
  时程 T=8000，batch=4，Adam(仅头)，wd 1e-4，clip 5，FP32，
  warmup 300 (1e-4->1e-3)，301..8000 线性降到 1e-5；val 点 0/1000/2000/4000/6000/8000。

T0 采样：复用旧 ForegroundMixedCrop 规则（50% 任意病灶前景锚定 / 50% 均匀）。
T1 采样：每 batch 2 均匀 + 1 MA 阳性锚定 + 1 train BASE_0 MA 误报锚定。
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

REPO = "/root/autodl-tmp/hrplus/repo"
os.chdir(REPO)
sys.path.insert(0, REPO)
CLS = ["EX", "HE", "SE", "MA"]
TAU = 0.5
WIN = 512
CEN = 256
HALF = WIN // 2
CLAMP = 1e-6      # 契约 §5.1 要求 1e-6（旧 H4 用 1e-4，已登记 D-001）
T = 8000
BATCH = 4
WARM = 300
LR_MAX = 1e-3
LR_MIN = 1e-5
LR_WARM_START = 1e-4


def lr_at(k):
    if k <= WARM:
        return LR_WARM_START + (LR_MAX - LR_WARM_START) * (k - 1) / (WARM - 1)
    return LR_MIN + (LR_MAX - LR_MIN) * (T - k) / (T - WARM)


class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(7, 16, 3, 1, 1), nn.ReLU(inplace=True),
                                 nn.Conv2d(16, 16, 3, 1, 1), nn.ReLU(inplace=True),
                                 nn.Conv2d(16, 1, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, f):
        return 6.0 * torch.tanh(self.net(f))


# ---------------------------------------------------------------------------
# U0：P2 的唯一合法首版实现（契约 §7）。不是多尺度 TTA。
#   * 每个卷积块：两次 Conv3x3(padding=1)+ReLU，无归一化 / 无 Dropout
#   * E0 7->32 @512；下采样 Conv3x3(stride=2,padding=1)+ReLU 后接同通道卷积块
#     E1 64@256, E2 128@128, E3 256@64
#   * 解码 bilinear resize 到对应 skip 的显式大小（align_corners=False），
#     拼接 E2/E1/E0 后卷积块输出 128/64/32 通道
#   * 最后 Conv1x1(32->1)，weight/bias 全零；输出仍为 6*tanh 的 MA logit 残差
# ---------------------------------------------------------------------------
def _cblock(cin, cout, stride=1):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, stride, 1), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, 1, 1), nn.ReLU(inplace=True))


class U0Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.e0 = _cblock(7, 32, 1)            # 512
        self.e1 = _cblock(32, 64, 2)           # 256
        self.e2 = _cblock(64, 128, 2)          # 128
        self.e3 = _cblock(128, 256, 2)         # 64
        self.d2 = _cblock(256 + 128, 128)
        self.d1 = _cblock(128 + 64, 64)
        self.d0 = _cblock(64 + 32, 32)
        self.out = nn.Conv2d(32, 1, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x, return_shapes=False):
        e0 = self.e0(x)
        e1 = self.e1(e0)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        d = F.interpolate(e3, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d = self.d2(torch.cat([d, e2], 1))
        d = F.interpolate(d, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d = self.d1(torch.cat([d, e1], 1))
        d = F.interpolate(d, size=e0.shape[-2:], mode="bilinear", align_corners=False)
        d = self.d0(torch.cat([d, e0], 1))
        out = 6.0 * torch.tanh(self.out(d))
        if return_shapes:
            return out, {"E0": tuple(e0.shape), "E1": tuple(e1.shape),
                         "E2": tuple(e2.shape), "E3": tuple(e3.shape),
                         "decode_out": tuple(out.shape), "in": tuple(x.shape)}
        return out


def lg(p):
    q = np.clip(p, CLAMP, 1 - CLAMP)
    return np.log(q / (1 - q)).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=["T0", "T1", "U0"])
    ap.add_argument("--cache-dir", default="/dev/shm/hrdec_cache")
    ap.add_argument("--val-cache", default="/dev/shm/hrdec_cache_val")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--iters", type=int, default=T)
    ap.add_argument("--val-every", type=int, default=1000)
    ap.add_argument("--smoke", type=int, default=0, help=">0 时只跑前 N 步做排错")
    args = ap.parse_args()

    from mmcv import Config
    from mmseg.datasets import build_dataset
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    base = Config.fromfile("configs/lesion/exp/hrplus_patch_dice.py")
    dtr = copy.deepcopy(base.data["train"])
    dtr["pipeline"] = copy.deepcopy(base.data["test"]["pipeline"])
    dtr.test_mode = True
    ds_tr = build_dataset(dtr)
    dva = copy.deepcopy(base.data["val"]); dva.test_mode = True
    ds_va = build_dataset(dva)
    gt_tr = ds_tr.get_gt_seg_maps()
    gt_va = ds_va.get_gt_seg_maps()
    n_tr, n_va = len(ds_tr), len(ds_va)
    print("[%s] train=%d val=%d" % (args.variant, n_tr, n_va), flush=True)

    # ---- 载入缓存 ----
    def load_field(sub, i):
        # mmap_mode='r'：不复制进 RAM，只按页读取（契约 §4 允许已验证等价的 memmap）
        return np.load(os.path.join(args.cache_dir, sub, "%04d.npy" % i),
                       mmap_mode="r")

    val_cache = args.val_cache
    base0 = [load_field("base0", i) for i in range(n_tr)]
    psf = [load_field("ps", i) for i in range(n_tr)]
    prf = [load_field("pr", i) for i in range(n_tr)]
    print("cache loaded", flush=True)

    # ---- 预计算锚点池（T1 需要）----
    ma_pix = []      # (img_idx, y, x) 每图一个列表
    fp_pix = []
    for i in range(n_tr):
        g = np.asarray(gt_tr[i])
        b0 = base0[i]
        m = (g == 4)
        ma_pix.append(np.argwhere(m))
        fp = (b0 > TAU) & (g == 0) & (g != 255)
        fp_pix.append(np.argwhere(fp))
    n_ma_img = sum(1 for a in ma_pix if len(a))
    n_fp_img = sum(1 for a in fp_pix if len(a))
    print("MA 阳性图 = %d, BASE_0 误报图 = %d" % (n_ma_img, n_fp_img), flush=True)

    def sample_anchor(kind, rng, idx_hint=None):
        """返回 (img_idx, ay, ax)。锚点 = 512 窗口中心。"""
        if kind == "uniform":
            i = int(rng.randint(0, n_tr))
            g = np.asarray(gt_tr[i]).shape
            ay = int(rng.randint(0, g[0])); ax = int(rng.randint(0, g[1]))
            return i, ay, ax
        if kind == "ma_pos":
            for _ in range(50):
                i = int(rng.randint(0, n_tr))
                a = ma_pix[i]
                if len(a):
                    y, x = a[rng.randint(0, len(a))]
                    return i, int(y), int(x)
            return sample_anchor("uniform", rng)
        if kind == "base0_fp":
            for _ in range(50):
                i = int(rng.randint(0, n_tr))
                a = fp_pix[i]
                if len(a):
                    y, x = a[rng.randint(0, len(a))]
                    return i, int(y), int(x)
            return sample_anchor("uniform", rng)
        raise ValueError(kind)

    def sample_anchor_T0(rng):
        """旧 ForegroundMixedCrop 规则：50% 任意病灶前景锚定 / 50% 均匀。"""
        if rng.random() < 0.5:
            for _ in range(50):
                i = int(rng.randint(0, n_tr))
                g = np.asarray(gt_tr[i])
                present = [c for c in (1, 2, 3, 4) if (g == c).any()]
                if present:
                    c = present[rng.randint(0, len(present))]
                    a = np.argwhere(g == c)
                    y, x = a[rng.randint(0, len(a))]
                    return i, int(y), int(x)
        return sample_anchor("uniform", rng)

    def build_sample(i, ay, ax):
        """返回 (feat(7,512,512), gt(512,512), valid_mask(512,512))。"""
        g = np.asarray(gt_tr[i]).astype(np.int64)
        H, W = g.shape
        y0, x0 = ay - HALF, ax - HALF
        def take(field, fill):
            out = np.full((WIN, WIN), fill, dtype=np.float32)
            sy0, sy1 = max(0, y0), min(H, y0 + WIN)
            sx0, sx1 = max(0, x0), min(W, x0 + WIN)
            if sy1 > sy0 and sx1 > sx0:
                src = field[sy0:sy1, sx0:sx1]
                dst = out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0]
                assert src.shape == dst.shape, (
                    'take() shape mismatch: field=%s ori=%s win=%d y0=%d x0=%d'
                    % (field.shape, (H, W), WIN, y0, x0))
                dst[...] = src
            return out
        b0 = take(base0[i], 0.0); ps = take(psf[i], 0.0); pr = take(prf[i], 0.0)
        gtw = take(g.astype(np.float32), 255.0).astype(np.int64)
        # RGB: 从原图取同窗口（已归一化）
        im = rgb_at(ds_tr, i, _rgb_cache)
        rgbw = np.zeros((3, WIN, WIN), dtype=np.float32)
        for c in range(3):
            rgbw[c] = take(im[c], 0.0)
        # 中心 256 的 valid mask（GT 非 ignore）
        cm = np.zeros((WIN, WIN), dtype=np.float32)
        cy0 = HALF - CEN // 2; cx0 = HALF - CEN // 2
        cm[cy0:cy0 + CEN, cx0:cx0 + CEN] = 1.0
        valid = cm * (gtw != 255).astype(np.float32)
        feat = np.concatenate([rgbw,
                               lg(b0)[None], lg(ps)[None], lg(pr)[None],
                               (ps - pr)[None]], axis=0)
        return feat, gtw, valid

    # ---- RGB 源：按新几何归一化 ----
    mean = np.array(base.data["test"]["pipeline"][1].get("mean", (123.675, 116.28, 103.53)),
                    dtype=np.float32)
    std = np.array(base.data["test"]["pipeline"][1].get("std", (58.395, 57.12, 57.375)),
                   dtype=np.float32)
    import cv2
    def _raw_arr(ds, i):
        raw = ds[i]["img"]
        n = 0
        while hasattr(raw, "data") and n < 12:
            raw = raw.data; n += 1
        while isinstance(raw, (list, tuple)):
            raw = raw[0]
            while hasattr(raw, "data") and n < 16:
                raw = raw.data; n += 1
        return (raw.numpy() if torch.is_tensor(raw) else np.asarray(raw)).astype(np.float32)

    def _ori_of(ds, i):
        m = ds[i]["img_metas"]
        n = 0
        while hasattr(m, "data") and n < 12:
            m = m.data; n += 1
        if isinstance(m, (list, tuple)):
            m = m[0]
            while hasattr(m, "data") and n < 16:
                m = m.data; n += 1
        return tuple(m["ori_shape"][:2])

    def _to_ori(arr, ori):
        """把管线尺寸的 RGB 插值到 ori_shape，与概率场的坐标一致（都是 bilinear 重采样）。"""
        H, W = ori
        if arr.shape[1] == H and arr.shape[2] == W:
            return arr
        out = np.empty((3, H, W), dtype=np.float32)
        for c in range(3):
            out[c] = cv2.resize(arr[c], (W, H), interpolation=cv2.INTER_LINEAR)
        return out

    _rgb_cache = {}
    def rgb_at(ds, i, cache):
        """一次 ds[i] 管线调用同时取图与 ori_shape；有界 LRU 缓存。"""
        if i in cache:
            return cache[i]
        sample = ds[i]
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
        a = _to_ori(arr, tuple(m["ori_shape"][:2]))
        cache[i] = a
        if len(cache) > 200:                     # 放大 LRU 以提高命中率（约 11 GB 上限）
            cache.pop(next(iter(cache)))
        return a
    print("rgb lazy (single-pass, LRU 200)", flush=True)

    _rgb_cache_va = {}
    print("rgb_va lazy ready", flush=True)

    if args.variant == "U0":
        head = U0Net().cuda()
        n_par = sum(p.numel() for p in head.parameters())
        print("[U0] 网络类=%s 参数量=%d（小 H4 头为 3361，必须不同）"
              % (type(head).__name__, n_par), flush=True)
        with torch.no_grad():
            _x = torch.zeros(1, 7, WIN, WIN).cuda()
            _o, _sh = head(_x, return_shapes=True)
        print("[U0] 结构证据 shapes =", _sh, flush=True)
        assert _sh["E0"][-2:] == (WIN, WIN) and _sh["E1"][-2:] == (WIN // 2, WIN // 2)
        assert _sh["E2"][-2:] == (WIN // 4, WIN // 4) and _sh["E3"][-2:] == (WIN // 8, WIN // 8)
        assert _sh["decode_out"][-2:] == (WIN, WIN), "skip 解码输出必须与输入网格一致"
        assert n_par != 3361, "U0 参数量不能等于小 H4 头"
    else:
        head = Head().cuda()
    opt = torch.optim.Adam(head.parameters(), lr=LR_WARM_START,
                           weight_decay=0.0001)

    def val_eval():
        """固定中心网格：512 窗口 stride 256，每像素恰由一个中心区覆盖。"""
        agg = {"base0": [0, 0, 0], "head": [0, 0, 0]}
        with torch.no_grad():
            for vi in range(n_va):
                g = np.asarray(gt_va[vi]).astype(np.int64)
                H, W = g.shape
                b0v = np.load(os.path.join(val_cache, "base0", "%04d.npy" % vi))
                psv = np.load(os.path.join(val_cache, "ps", "%04d.npy" % vi))
                prv = np.load(os.path.join(val_cache, "pr", "%04d.npy" % vi))
                # 先把 BASE_0 与 head 的输出都算成概率图
                prob_b = b0v.copy()
                prob_h = b0v.copy()
                ys = list(range(0, max(H - CEN + 1, 1), CEN)) or [0]
                xs = list(range(0, max(W - CEN + 1, 1), CEN)) or [0]
                for y in ys:
                    for x in xs:
                        ay = min(y + CEN // 2, H - 1)
                        ax = min(x + CEN // 2, W - 1)
                        y0 = ay - HALF; x0 = ax - HALF
                        def tk(fld, fill):
                            o = np.full((WIN, WIN), fill, dtype=np.float32)
                            sy0, sy1 = max(0, y0), min(H, y0 + WIN)
                            sx0, sx1 = max(0, x0), min(W, x0 + WIN)
                            if sy1 > sy0 and sx1 > sx0:
                                o[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = fld[sy0:sy1, sx0:sx1]
                            return o
                        fb = tk(b0v, 0.0); fs = tk(psv, 0.0); fr = tk(prv, 0.0)
                        im = rgb_at(ds_va, vi, _rgb_cache_va)
                        rgbw = np.zeros((3, WIN, WIN), dtype=np.float32)
                        for c in range(3):
                            rgbw[c] = tk(im[c], 0.0)
                        feat = torch.from_numpy(np.concatenate(
                            [rgbw, lg(fb)[None], lg(fs)[None], lg(fr)[None],
                             (fs - fr)[None]], axis=0)).unsqueeze(0).cuda()
                        zc = torch.from_numpy(lg(fb)[HALF - CEN//2:HALF + CEN//2,
                                                       HALF - CEN//2:HALF + CEN//2]).cuda()
                        with torch.no_grad():
                            dc = head(feat)[0, 0, HALF - CEN//2:HALF + CEN//2,
                                            HALF - CEN//2:HALF + CEN//2]
                        pn = torch.sigmoid(zc + dc).cpu().numpy()
                        cy = min(y, max(H - CEN, 0)); cx = min(x, max(W - CEN, 0))
                        ye = min(cy + CEN, H); xe = min(cx + CEN, W)
                        prob_h[cy:ye, cx:xe] = pn[:ye - cy, :xe - cx]
                gg = (g == 4); ig = (g == 255)
                for tag, pr in (("base0", prob_b), ("head", prob_h)):
                    p = pr > TAU
                    agg[tag][0] += int((p & gg & ~ig).sum())
                    agg[tag][1] += int((p & ~gg & ~ig).sum())
                    agg[tag][2] += int((~p & gg).sum())
        out = {}
        for k, (tp, fp, fn) in agg.items():
            den = tp + fp + fn
            out[k] = tp / den if den else 0.0
        return out

    hist = []
    skip_all_ignore = 0     # 中心区全为 ignore -> 按契约跳过更新
    n_vacuous_ma = 0        # 中心区无 MA 阳性（Dice 退化）-> 仅计数，不改行为
    cnt_pos_px = 0
    rng = np.random.RandomState(args.seed)
    t0 = time.time()
    src_count = {"uniform": 0, "ma_pos": 0, "base0_fp": 0}
    for it in range(1, args.iters + 1):
        batch = []
        if args.variant == "T0":
            kinds = ["T0_rule"] * BATCH
        else:   # T1 与 U0 共用同一完整采样
            kinds = ["uniform", "uniform", "ma_pos", "base0_fp"]
        for k in kinds:
            if k == "T0_rule":
                i, ay, ax = sample_anchor_T0(rng)
                src_count["uniform" if False else "T0_rule"] = src_count.get("T0_rule", 0) + 1
            else:
                i, ay, ax = sample_anchor(k, rng)
                src_count[k] += 1
            batch.append(build_sample(i, ay, ax))
        feats = torch.from_numpy(np.stack([b[0] for b in batch])).cuda()
        gts = torch.from_numpy(np.stack([b[1] for b in batch])).long().cuda()
        val_m = torch.from_numpy(np.stack([b[2] for b in batch])).cuda()
        lr = lr_at(it)
        for gp in opt.param_groups:
            gp["lr"] = lr

        cy0 = HALF - CEN // 2
        z_base = feats[:, 3:4, cy0:cy0 + CEN, cy0:cy0 + CEN]   # = logit(P_BASE0_MA)
        y_c = (gts[:, cy0:cy0 + CEN, cy0:cy0 + CEN] == 4).float()[:, None]
        v_c = val_m[:, cy0:cy0 + CEN, cy0:cy0 + CEN][:, None]
        delta_c = head(feats)[:, :, cy0:cy0 + CEN, cy0:cy0 + CEN]
        z_new = z_base + delta_c
        p_new = torch.sigmoid(z_new)

        n_valid = float(v_c.sum())
        n_pos = float((y_c * v_c).sum())
        cnt_pos_px += int(n_pos)
        # 契约 §8：全 ignore batch 跳过更新并记录
        if n_valid == 0.0:
            skip_all_ignore += 1
            if skip_all_ignore <= 5 or skip_all_ignore % 20 == 0:
                print("  [skip] iter=%d 中心区全 ignore，本步不更新（累计 %d）"
                      % (it, skip_all_ignore), flush=True)
            continue
        if n_pos == 0.0:
            n_vacuous_ma += 1

        denom = v_c.sum().clamp(min=1.0)
        bce = (F.binary_cross_entropy_with_logits(z_new, y_c, reduction="none")
               * v_c).sum() / denom
        pv = p_new * v_c; yv = y_c * v_c
        dice = 1 - (2 * (pv * yv).sum() + 1e-5) / ((pv * pv).sum() + (yv * yv).sum() + 1e-5)
        loss = 0.5 * bce + 0.5 * dice

        if not torch.isfinite(loss):
            print("!! 第 %d 步 loss 非有限，FAILED_IMPLEMENTATION" % it, flush=True)
            sys.exit(3)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(head.parameters(), 5.0)
        opt.step()

        if it % 50 == 0 or it == 1:
            print("  iter=%-5d lr=%.2e loss=%.4f bce=%.4f dice=%.4f gnorm=%.2f %.0fs"
                  % (it, lr, loss, bce, dice, gnorm, time.time() - t0), flush=True)
        if args.smoke and it >= args.smoke:
            print("smoke 完成 %d 步，未启动正式训练" % it)
            return

    print("采样来源计数:", src_count, flush=True)
    print("全 ignore 跳过步 = %d / %d (%.2f%%)" % (
        skip_all_ignore, args.iters, 100.0 * skip_all_ignore / args.iters), flush=True)
    print("中心区无 MA 阳性的 batch = %d / %d (%.2f%%)，MA 阳性像素累计 = %d" % (
        n_vacuous_ma, args.iters, 100.0 * n_vacuous_ma / args.iters, cnt_pos_px), flush=True)
    torch.save({"head": head.state_dict(), "step": args.iters, "variant": args.variant},
               os.path.join(args.out_dir, "head_%s.pth" % args.variant))
    print("训练完成，head 已保存", flush=True)


if __name__ == "__main__":
    main()
