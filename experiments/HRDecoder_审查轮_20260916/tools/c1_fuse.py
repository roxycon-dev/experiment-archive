#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C1（val 融合扫描）与 F1（test 确认）共用脚本 —— 同一套推理代码，--split 切换。

语义（逐字复用 train_T0T1.py 的 val_eval：窗口网格、坐标放置、tk/lg 全同）：
    小头输出的是对 BASE_0 logit 的**残差**  dc = head(feat)
    融合概率  p(w) = sigmoid( lg(base0) + Σ_i w_i · dc_i )
  →   w=1.0        小头修正全量生效（"单头"）
      w=0.5        与 BASE_0 等权（契约里的"等权"组合）
      w=0.3        偏向冠军 BASE_0（契约里的"偏向冠军的权重"）
      w=0          = BASE_0 本身（参照）
  两候选头组合： (w1,w2) = (0.5,0.5) 等权 / (0.25,0.25) 偏向冠军
非 MA 三类恒取 BASE_0（契约：小头只改 MA 通道）⇒ 4 类 mIoU = (EX+HE+SE+MA)/4，前三项为常数。

不变量：
  * 阈值 TAU=0.5、先累计 TP/FP/FN 再求 IoU、空并集记 0
  * val 集/网格与各臂 val_eval 完全一致 → 可与各臂已落盘的逐图计数直接对账
  * 只读缓存与 ckpt，不训练、不写任何训练产物
"""
import argparse
import copy
import json
import os
import sys
import time

import numpy as np
import torch

REPO = "/root/autodl-tmp/hrplus/repo"
os.chdir(REPO)
sys.path.insert(0, REPO)

CLS = ["EX", "HE", "SE", "MA"]
TAU = 0.5
WIN = 512
CEN = 256
HALF = WIN // 2
CLAMP = 1e-6
BASE0_NON_MA = {"EX": 0.457705, "HE": 0.437552, "SE": 0.551806}

# ---- 网络定义：运行时从训练脚本 import，保证「同一份代码」不变量 ----
# 为什么不再手抄：本文件首版手抄的 U0Net 结构与 trainer 里 `_cblock` 版**完全不同**
#   （手抄版 keys = e0/d1/d2/d3/u2/u1/u0/out，trainer 版 keys = e0/e1/e2/e3/d2/d1/d0/out）
#   → `load_state_dict(strict=True)` 必崩。手抄 = 迟早漂移，改成导入即从根上消除这一类缺陷。
import importlib.util
import torch.nn as nn  # noqa: F401  （保留供后续使用）


def _load_trainer():
    cand = [os.path.join(REPO, "..", "ops", "train_T0T1.py"),
            "/root/autodl-tmp/hrplus/ops/train_T0T1.py",
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_T0T1.py")]
    for p in cand:
        p = os.path.normpath(p)
        if os.path.exists(p):
            spec = importlib.util.spec_from_file_location("hrdec_trainer", p)
            m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m)
            return m, p
    raise RuntimeError("找不到 train_T0T1.py，尝试过: %s" % cand)


_TR, TRAINER_PATH = _load_trainer()
Head = _TR.Head
U0Net = _TR.U0Net
lg = _TR.lg


def build_head(variant):
    return U0Net() if variant.upper() == "U0" else Head()


def load_head(variant, ckpt):
    h = build_head(variant).cuda().eval()
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = ck["head"] if isinstance(ck, dict) and "head" in ck else ck
    try:
        h.load_state_dict(sd, strict=True)
    except RuntimeError as e:
        mine = set(build_head(variant).state_dict().keys())
        theirs = set(sd.keys())
        raise RuntimeError(
            "ckpt 与本地网络不匹配（variant=%s, file=%s, trainer=%s）\n"
            "  只在 ckpt: %s\n  只在本地: %s\n  原始错误: %s"
            % (variant, ckpt, TRAINER_PATH,
               sorted(theirs - mine)[:6], sorted(mine - theirs)[:6], e))
    meta = {"ckpt": ckpt, "variant": variant,
            "step": (ck.get("step") if isinstance(ck, dict) else None),
            "n_params": int(sum(v.numel() for v in h.parameters()))}
    return h, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--cache-dir", default="")
    ap.add_argument("--candidates", required=True,
                    help="名=variant:ckpt，分号分隔。例: U0=U0:/path/head_U0_s1.pth;T0=T0:/path/head_T0.pth")
    ap.add_argument("--config-json", default="", help="给定则只评该 config（F1 锁定用），跳过扫描")
    ap.add_argument("--out-counts", default="", help="逐图计数输出路径")
    ap.add_argument("--counts-id", default="", help="指定 --out-counts 导出哪个 config 的逐图计数（默认第一名）")
    ap.add_argument("--out-counts-dir", default="",
                    help="把**每个** config 的逐图计数各写一个 json 到该目录（目录名做安全化）。"
                         "用于事后对任一方案做配对 bootstrap，避免为每个方案重跑一遍推理。")
    ap.add_argument("--out-json", default="", help="汇总输出路径")
    ap.add_argument("--non-ma", default="",
                    help="覆盖非 MA 三类的 BASE_0 IoU（分号分隔，例 EX=0.5386;HE=0.4451;SE=0.4982）。"
                         "test 上必须覆盖：val 常数不可套用到 test。")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    non_ma = dict(BASE0_NON_MA)
    if a.non_ma:
        for kv in a.non_ma.split(";"):
            kv = kv.strip()
            if kv:
                k, v = kv.split("=", 1)
                non_ma[k.strip()] = float(v)
    print("[%s] 非 MA 三类基准 IoU = %s" % (a.split, non_ma), flush=True)

    cache_dir = a.cache_dir or ("/dev/shm/hrdec_cache_val" if a.split == "val" else "/dev/shm/hrdec_cache_test")

    from mmcv import Config
    from mmseg.datasets import build_dataset
    import cv2
    from mmseg.models import build_segmentor  # noqa

    base = Config.fromfile("configs/lesion/exp/hrplus_patch_dice.py")
    d = copy.deepcopy(base.data[a.split])
    d["pipeline"] = copy.deepcopy(base.data["test"]["pipeline"])
    d.test_mode = True
    ds = build_dataset(d)
    n = len(ds) if a.limit <= 0 else min(a.limit, len(ds))
    gt = ds.get_gt_seg_maps()
    print("[%s] n=%d  cache=%s" % (a.split, n, cache_dir), flush=True)

    cands = []
    for sp in a.candidates.split(";"):
        sp = sp.strip()
        if not sp:
            continue
        name, rest = sp.split("=", 1)
        variant, ck = rest.split(":", 1)
        h, meta = load_head(variant, ck)
        cands.append((name.strip(), h, meta))
        print("  候选 %-6s variant=%-3s step=%s <- %s" % (name, variant, meta["step"], ck), flush=True)
    names = [c[0] for c in cands]

    # ---------- 待评 config ----------
    if a.config_json:
        cfgs = json.load(open(a.config_json))
        if isinstance(cfgs, dict):
            cfgs = [cfgs]
    else:
        cfgs = [{"id": "BASE_0(w=0)", "terms": []}]
        for nm in names:
            for w in (1.0, 0.5, 0.3):
                cfgs.append({"id": "%s(w=%.2f)" % (nm, w), "terms": [[nm, w]]})
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                cfgs.append({"id": "%s+%s(0.5/0.5)" % (names[i], names[j]),
                             "terms": [[names[i], 0.5], [names[j], 0.5]]})
                cfgs.append({"id": "%s+%s(0.25/0.25)" % (names[i], names[j]),
                             "terms": [[names[i], 0.25], [names[j], 0.25]]})
    for c in cfgs:
        for nm, _w in c["terms"]:
            assert nm in names, "config 引用了未提供的候选: %s" % nm

    # ---------- 逐图循环（一次遍历，所有 config 同时评）----------
    rows = {c["id"]: [] for c in cfgs}
    agg = {c["id"]: {"base0": [0, 0, 0], "head": [0, 0, 0]} for c in cfgs}

    std = np.array(base.data["test"]["pipeline"][1].get("std", (58.395, 57.12, 57.375)),
                   dtype=np.float32)

    def tk(fld, fill, y0, x0, H, W):
        o = np.full((WIN, WIN), fill, dtype=np.float32)
        sy0, sy1 = max(0, y0), min(H, y0 + WIN)
        sx0, sx1 = max(0, x0), min(W, x0 + WIN)
        if sy1 > sy0 and sx1 > sx0:
            o[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = fld[sy0:sy1, sx0:sx1]
        return o

    _rgb_cache = {}

    def to_ori(arr, ori):
        H, W = ori
        if arr.shape[1] == H and arr.shape[2] == W:
            return arr
        out = np.empty((3, H, W), dtype=np.float32)
        for c in range(3):
            out[c] = cv2.resize(arr[c], (W, H), interpolation=cv2.INTER_LINEAR)
        return out

    def rgb_at(i):
        if i in _rgb_cache:
            return _rgb_cache[i]
        s = ds[i]
        raw = s["img"]
        k = 0
        while hasattr(raw, "data") and k < 16:
            raw = raw.data; k += 1
        while isinstance(raw, (list, tuple)):
            raw = raw[0]
            while hasattr(raw, "data") and k < 20:
                raw = raw.data; k += 1
        arr = (raw.numpy() if torch.is_tensor(raw) else np.asarray(raw)).astype(np.float32)
        m = s["img_metas"]
        k = 0
        while hasattr(m, "data") and k < 16:
            m = m.data; k += 1
        if isinstance(m, (list, tuple)):
            m = m[0]
            while hasattr(m, "data") and k < 20:
                m = m.data; k += 1
        out = to_ori(arr, tuple(m["ori_shape"][:2]))
        _rgb_cache[i] = out
        if len(_rgb_cache) > 60:
            _rgb_cache.pop(next(iter(_rgb_cache)))
        return out

    t0 = time.time()
    for vi in range(n):
        g = np.asarray(gt[vi]).astype(np.int64)
        H, W = g.shape
        b0v = np.load(os.path.join(cache_dir, "base0", "%04d.npy" % vi))
        psv = np.load(os.path.join(cache_dir, "ps", "%04d.npy" % vi))
        prv = np.load(os.path.join(cache_dir, "pr", "%04d.npy" % vi))
        im = rgb_at(vi)
        zc_full = lg(b0v)
        dc = {nm: np.zeros((H, W), dtype=np.float32) for nm in names}
        ys = list(range(0, max(H - CEN + 1, 1), CEN)) or [0]
        xs = list(range(0, max(W - CEN + 1, 1), CEN)) or [0]
        for y in ys:
            for x in xs:
                ay = min(y + CEN // 2, H - 1)
                ax = min(x + CEN // 2, W - 1)
                y0 = ay - HALF; x0 = ax - HALF
                fb = tk(b0v, 0.0, y0, x0, H, W)
                fs = tk(psv, 0.0, y0, x0, H, W)
                fr = tk(prv, 0.0, y0, x0, H, W)
                rgbw = np.zeros((3, WIN, WIN), dtype=np.float32)
                for c in range(3):
                    rgbw[c] = tk(im[c], 0.0, y0, x0, H, W)
                feat = torch.from_numpy(np.concatenate(
                    [rgbw, lg(fb)[None], lg(fs)[None], lg(fr)[None], (fs - fr)[None]],
                    axis=0)).unsqueeze(0).cuda()
                cy = min(y, max(H - CEN, 0)); cx = min(x, max(W - CEN, 0))
                ye = min(cy + CEN, H); xe = min(cx + CEN, W)
                sl = (slice(HALF - CEN // 2, HALF + CEN // 2),
                      slice(HALF - CEN // 2, HALF + CEN // 2))
                with torch.no_grad():
                    for nm, h, _m in cands:
                        dcv = h(feat)[0, 0][sl].cpu().numpy()
                        dc[nm][cy:ye, cx:xe] = dcv[:ye - cy, :xe - cx]
        gg = (g == 4); ig = (g == 255)
        # BASE_0 参照（所有 config 共用一个 base0 行）
        pb = b0v > TAU
        r_base = {"img": int(vi),
                  "base0": [int((pb & gg & ~ig).sum()), int((pb & ~gg & ~ig).sum()),
                            int((~pb & gg).sum())]}
        for c in cfgs:
            z = zc_full.copy()
            for nm, w in c["terms"]:
                z = z + w * dc[nm]
            p = 1.0 / (1.0 + np.exp(-z))
            pm = p > TAU
            row = dict(r_base)
            row["head"] = [int((pm & gg & ~ig).sum()), int((pm & ~gg & ~ig).sum()),
                           int((~pm & gg).sum())]
            rows[c["id"]].append(row)
            for tag in ("base0", "head"):
                agg[c["id"]][tag][0] += row[tag][0]
                agg[c["id"]][tag][1] += row[tag][1]
                agg[c["id"]][tag][2] += row[tag][2]
        if (vi + 1) % 10 == 0 or vi + 1 == n:
            el = time.time() - t0
            print("  %d/%d  %.0fs  ETA %.0fs" % (vi + 1, n, el, el / (vi + 1) * (n - vi - 1)),
                  flush=True)

    # ---------- 汇总 ----------
    def iou(t):
        den = t[0] + t[1] + t[2]
        return t[0] / den if den else 0.0

    res = []
    for c in cfgs:
        i_b = iou(agg[c["id"]]["base0"]); i_h = iou(agg[c["id"]]["head"])
        res.append({"id": c["id"], "terms": c["terms"],
                    "MA_IoU_base0": i_b, "MA_IoU": i_h, "delta": i_h - i_b,
                    "mIoU_base0": float(np.mean([non_ma["EX"], non_ma["HE"],
                                                 non_ma["SE"], i_b])),
                    "mIoU": float(np.mean([non_ma["EX"], non_ma["HE"],
                                           non_ma["SE"], i_h])),
                    "TP_FP_FN": agg[c["id"]]["head"]})
    res.sort(key=lambda r: -r["MA_IoU"])
    print("\n%-28s %9s %9s %10s %9s  %s" % ("config", "MA_IoU", "ΔMA", "mIoU", "ΔmIoU", "TP/FP/FN"))
    for r in res:
        print("%-28s %.6f %+9.6f %.6f %+9.6f  %s" % (
            r["id"], r["MA_IoU"], r["delta"], r["mIoU"], r["mIoU"] - r["mIoU_base0"],
            r["TP_FP_FN"]))

    out_dir = os.path.dirname(a.out_json) if a.out_json else "."
    if a.out_json:
        os.makedirs(out_dir or ".", exist_ok=True)
        with open(a.out_json, "w", encoding="utf-8") as f:
            json.dump({"split": a.split, "cache_dir": cache_dir, "n": n,
                       "non_ma_baseline": non_ma,
                       "tau": TAU, "win": WIN, "stride": CEN,
                       "candidates": [{"name": nm, **m} for nm, _h, m in cands],
                       "table": res}, f, indent=1, ensure_ascii=False)
        print("写出:", a.out_json)
    if a.out_counts:
        win = res[0]
        if a.counts_id:
            hit = [r for r in res if r["id"] == a.counts_id]
            if not hit:
                raise SystemExit("--counts-id 未匹配任何 config: %s（可选：%s）"
                                 % (a.counts_id, [r["id"] for r in res]))
            win = hit[0]
        with open(a.out_counts, "w", encoding="utf-8") as f:
            json.dump({"variant": "+".join("%s*%.2f" % tuple(t) for t in win["terms"]) or "BASE_0",
                       "seed": 0, "iter": 0, "cls_index": 4, "tau": TAU,
                       "split": a.split, "n_val": n,
                       "agg": {"base0": win["MA_IoU_base0"], "head": win["MA_IoU"]},
                       "config": win, "per_image": rows[win["id"]]}, f)
        print("逐图计数写出:", a.out_counts, "（给配对 bootstrap 用）")
    if a.out_counts_dir:
        os.makedirs(a.out_counts_dir, exist_ok=True)

        def safe(s):
            return "".join(ch if (ch.isalnum() or ch in "._-+") else "_" for ch in s)

        for r in res:
            p = os.path.join(a.out_counts_dir, "counts_%s.json" % safe(r["id"]))
            with open(p, "w", encoding="utf-8") as f:
                json.dump({"variant": "+".join("%s*%.2f" % tuple(t) for t in r["terms"]) or "BASE_0",
                           "seed": 0, "iter": 0, "cls_index": 4, "tau": TAU,
                           "split": a.split, "n_val": n,
                           "agg": {"base0": r["MA_IoU_base0"], "head": r["MA_IoU"]},
                           "config": r, "per_image": rows[r["id"]]}, f)
        print("逐图计数（全部 %d 个 config）写出目录: %s" % (len(res), a.out_counts_dir))


if __name__ == "__main__":
    main()
