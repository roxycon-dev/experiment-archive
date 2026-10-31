#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""T0/T1 共用缓存构建：BASE_0 / student / reference 的 MA 概率场（train 454 张）。

严格按契约 §5.1：
  - BASE_0 = 集成(D×3+E×2+B+E4) + 翻转 TTA（8 前向/图）
  - student/reference = D@2000 的 _student_prob / _reference_prob
  - 只保存 **MA 单通道**，float32，at ori_shape（与 rescale=True 的输出坐标一致）
  - 记录来源哈希、sample ID、原图大小、dtype、模型/尺度/TTA 身份
  - **不保存任何 test 概率**

缓存目录走内存盘（/dev/shm），非持久；元数据同时落持久盘。
"""
import argparse
import copy
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch

REPO = "/root/autodl-tmp/hrplus/repo"
os.chdir(REPO)
sys.path.insert(0, REPO)


def unwrap(x):
    n = 0
    while hasattr(x, "data") and n < 8:
        x = x.data; n += 1
    return x


def as_img(x):
    x = unwrap(x)
    while isinstance(x, (list, tuple)):
        x = unwrap(x[0])
    return x


def as_meta(x):
    x = unwrap(x)
    if isinstance(x, dict):
        return [x]
    out = []
    for e in x:
        e = unwrap(e)
        out.extend([e] if isinstance(e, dict) else [unwrap(e)])
    return out


def sha256_file(p, chunk=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="/dev/shm/hrdec_cache")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--models", required=True)
    ap.add_argument("--flip", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--split", default="train")
    args = ap.parse_args()

    from mmcv import Config
    from mmseg.datasets import build_dataset
    from mmseg.models import build_segmentor
    from mmcv.runner import load_checkpoint

    base = Config.fromfile("configs/lesion/exp/hrplus_patch_dice.py")
    d = copy.deepcopy(base.data[args.split])
    d["pipeline"] = copy.deepcopy(base.data["test"]["pipeline"])  # 只要图+meta，不需要 gt
    d.test_mode = True
    ds = build_dataset(d)
    n = len(ds) if args.limit <= 0 else min(args.limit, len(ds))
    print("%s =" % args.split, n, flush=True)

    for sub in ("base0", "ps", "pr"):
        os.makedirs(os.path.join(args.cache_dir, sub), exist_ok=True)

    specs = []
    for sp in args.models.split(";"):
        sp = sp.strip()
        if not sp:
            continue
        cfgp, rest = sp.split("=", 1)
        ck, w = (rest.rsplit(":", 1) + ["1.0"])[:2] if ":" in rest else (rest, "1.0")
        specs.append((cfgp, ck, float(w)))
    members = []
    for cfgp, ck, w in specs:
        c = Config.fromfile(cfgp)
        m = build_segmentor(c.model)
        load_checkpoint(m, ck, map_location="cpu", strict=False)
        members.append((m.cuda().eval(), w))
    ckpt_hashes = {}
    for _c, ck, _w in specs:
        if os.path.isfile(ck):
            ckpt_hashes[os.path.basename(ck)] = sha256_file(ck)
    tot = sum(w for _, w in members) * (2 if args.flip else 1)
    print("BASE_0 权重 =", [s[2] for s in specs], " flip =", args.flip, flush=True)

    dm = build_segmentor(base.model)
    load_checkpoint(dm, "/root/autodl-tmp/hrplus/D_0913_retry/best_mIoU_iter_2000.pth",
                    map_location="cpu", strict=False)
    dm = dm.cuda().eval()

    manifest = []
    t0 = time.time()
    for i in range(n):
        b0p = os.path.join(args.cache_dir, "base0", "%04d.npy" % i)
        psp = os.path.join(args.cache_dir, "ps", "%04d.npy" % i)
        prp = os.path.join(args.cache_dir, "pr", "%04d.npy" % i)
        s = ds[i]
        meta = as_meta(s["img_metas"])
        ori = tuple(meta[0]["ori_shape"][:2])
        if all(os.path.isfile(p) for p in (b0p, psp, prp)):
            manifest.append({"idx": i, "ori_shape": list(ori), "cached": True})
            continue
        img = as_img(s["img"]).unsqueeze(0).cuda()
        with torch.no_grad():
            acc = None
            for m, w in members:
                p = m(return_loss=False, rescale=True, img=[img], img_metas=[meta])[0]
                p = np.asarray(p[0] if isinstance(p, tuple) else p, dtype=np.float32)
                acc = p[3] * w if acc is None else acc + p[3] * w
                if args.flip:
                    imf = img.flip(-1)
                    mf = copy.deepcopy(meta)
                    mf[0]["flip"] = True; mf[0]["flip_direction"] = "horizontal"
                    p2 = m(return_loss=False, rescale=True, img=[imf], img_metas=[mf])[0]
                    p2 = np.asarray(p2[0] if isinstance(p2, tuple) else p2, dtype=np.float32)
                    acc = acc + p2[3] * w
            np.save(b0p, (acc / tot).astype(np.float32))
            np.save(psp, dm._student_prob(img, meta, True)[0][3].cpu().numpy().astype(np.float32))
            np.save(prp, dm._reference_prob(img, meta, True)[0][3].cpu().numpy().astype(np.float32))
        manifest.append({"idx": i, "ori_shape": list(ori),
                         "filename": meta[0].get("ori_filename", ""), "cached": False})
        if (i + 1) % 25 == 0 or i + 1 == n:
            el = time.time() - t0
            print("  %d/%d  %.0fs  ETA %.0fs" % (i + 1, n, el,
                  el / (i + 1) * (n - i - 1)), flush=True)

    mf = {"cache_dir": args.cache_dir, "n_images": n,
          "fields": ["base0_MA", "student_MA", "reference_MA"],
          "dtype": "float32", "coords": "ori_shape (rescale=True)",
          "base0_identity": {"members": [{"cfg": a, "ckpt": b, "w": c} for a, b, c in specs],
                             "flip_tta": args.flip,
                             "forward_passes_per_image": len(specs) * (2 if args.flip else 1)},
          "ckpt_sha256": ckpt_hashes,
          "student_ref_source": "D@2000 (configs/lesion/exp/hrplus_patch_dice.py)",
          "split": args.split + " (never runs test probability extraction when split != test)",
          "images": manifest}
    os.makedirs(os.path.join(args.out_dir, "cache"), exist_ok=True)
    p = os.path.join(args.out_dir, "cache", "cache_manifest.json")
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(mf, f, indent=1, ensure_ascii=False)
    os.replace(tmp, p)
    sz = sum(os.path.getsize(os.path.join(dp, f))
             for dp, _, fs in os.walk(args.cache_dir) for f in fs)
    print("\n缓存完成：%d 图 x 3 场，共 %.2f GB" % (n, sz / 1e9))
    print("写出:", p)


if __name__ == "__main__":
    main()
