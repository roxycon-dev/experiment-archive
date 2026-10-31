#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""多模型 + 翻转 TTA 评估器（v2：手动翻转，不依赖 pipeline 的 RandomFlip）。

v1 的坑：本仓库 test pipeline 的 MultiScaleFlipAug.transforms 里没有 RandomFlip，
开 flip=True 只会设置 img_meta['flip'] 而不会真的翻转图像，模型却按该标志把输出
翻回去，导致 mIoU 归零。v2 直接绕过 DataLoader，手动翻转 img 张量并同步改写
img_meta.flip，再逐模型前向。

指标口径不变：复用 dataset.evaluate -> lesion_metrics（全数据集逐类汇总 TP/FP/FN）。
"""
import argparse
import copy
import json
import os
import sys
import time

import numpy as np
import torch
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmseg.datasets import build_dataset
from mmseg.models import build_segmentor

REPO = '/root/autodl-tmp/hrplus/repo'
os.chdir(REPO)
sys.path.insert(0, REPO)


def unwrap(x):
    """剥掉 DataContainer 包装，拿到真正的 list/dict。"""
    seen = 0
    while hasattr(x, 'data') and seen < 8:
        x = x.data
        seen += 1
    return x


def as_meta_list(x):
    """规范成 list[dict]。"""
    x = unwrap(x)
    if isinstance(x, dict):
        return [x]
    if isinstance(x, (list, tuple)):
        out = []
        for e in x:
            out.extend(as_meta_list(e) if not isinstance(e, dict) else [e])
        return out
    raise TypeError('unsupported img_metas type %r' % (type(x),))


def as_img_tensor(x):
    """规范成 (C,H,W) tensor。"""
    x = unwrap(x)
    while isinstance(x, (list, tuple)):
        if len(x) == 0:
            raise ValueError('empty img list')
        x = unwrap(x[0])
    if not torch.is_tensor(x):
        raise TypeError('unsupported img type %r' % (type(x),))
    return x


def forward_prob(model, img, img_meta):
    """img: (C,H,W) tensor; img_meta: list[dict] -> np prob (4,H,W)

    BaseSegmentor.forward(img, img_metas, ...) -> forward_test(imgs=img, ...)
    所以关键字必须是 img= / img_metas=。
    """
    img_b = img.unsqueeze(0).cuda()
    with torch.no_grad():
        out = model(return_loss=False, rescale=True, img=[img_b], img_metas=[img_meta])
    seg = out[0]
    prob = seg[0] if isinstance(seg, (tuple, list)) else seg
    return np.asarray(prob, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base-config', required=True)
    ap.add_argument('--models', required=True, help='"cfg=ckpt[:w]" 分号分隔')
    ap.add_argument('--flip', action='store_true')
    ap.add_argument('--scales', default=None,
                    help='逗号分隔的输入边长，如 "2048,2560"；多尺度 TTA')
    ap.add_argument('--out', required=True)
    ap.add_argument('--split', default='test')
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()

    base = Config.fromfile(args.base_config)
    ds_cfg = copy.deepcopy(base.data[args.split])
    ds_cfg.test_mode = True
    dataset = build_dataset(ds_cfg)
    n = len(dataset) if args.limit <= 0 else min(args.limit, len(dataset))
    print('dataset %s: %d images' % (args.split, n), flush=True)

    # 多尺度：每个尺度一份数据集（只改 MultiScaleFlipAug.img_scale）
    scales = [None]
    if args.scales:
        scales = [int(s) for s in args.scales.split(',')]
    dsets = {}
    for sc in scales:
        d = copy.deepcopy(base.data[args.split])
        d.test_mode = True
        if sc is not None:
            for p in d.pipeline:
                if p.get('type') == 'MultiScaleFlipAug':
                    p.img_scale = (sc, sc)
                    p.img_ratios = None
        dsets[sc] = dataset if sc is None else build_dataset(d)
        print('  scale %s -> %d imgs' % (sc, len(dsets[sc])), flush=True)

    specs = []
    for spec in args.models.split(';'):
        spec = spec.strip()
        if not spec:
            continue
        cfgp, rest = spec.split('=', 1)
        if ':' in rest:
            ck, w = rest.rsplit(':', 1)
            w = float(w)
        else:
            ck, w = rest, 1.0
        specs.append((cfgp, ck, w))
    print('models:', [(os.path.basename(k), w) for _, k, w in specs], flush=True)

    models, load_report = [], []
    for cfgp, ck, w in specs:
        c = Config.fromfile(cfgp)
        m = build_segmentor(c.model)
        r = load_checkpoint(m, ck, map_location='cpu', strict=False)
        try:
            miss, unexp = r
        except Exception:
            miss, unexp = [], []
        nunet = [k for k in miss if 'num_batches_tracked' not in k]
        print('LOAD %-28s type=%s sem_ref=%s missing=%d unexpected=%d'
              % (os.path.basename(ck), c.model.get('type'),
                 c.model.get('semantic_refinement'), len(nunet), len(unexp)),
              flush=True)
        if nunet:
            print('  !! missing:', nunet[:8], flush=True)
        if unexp:
            print('  !! unexpected:', unexp[:8], flush=True)
        load_report.append({'ckpt': os.path.basename(ck),
                            'missing': nunet[:8], 'unexpected': unexp[:8]})
        m.eval()
        models.append(m.cuda())
        print('  loaded %.3f' % w, flush=True)

    npass = len(scales) * (2 if args.flip else 1)
    totals = sum(w for _, _, w in specs) * npass
    acc = [None] * n
    t0 = time.time()
    for sc in scales:
        ds = dsets[sc]
        for i in range(n):
            sample = ds[i]
            img = as_img_tensor(sample['img'])          # (C,H,W)
            img_meta = [dict(m) for m in as_meta_list(sample['img_metas'])]

            for m, (cfgp, ck, w) in zip(models, specs):
                p = forward_prob(m, img, copy.deepcopy(img_meta)) * w
                acc[i] = p if acc[i] is None else acc[i] + p

            if args.flip:
                img_f = img.flip(-1)                 # 水平翻转
                meta_f = copy.deepcopy(img_meta)
                meta_f[0]['flip'] = True
                meta_f[0]['flip_direction'] = 'horizontal'
                for m, (cfgp, ck, w) in zip(models, specs):
                    p = forward_prob(m, img_f, copy.deepcopy(meta_f)) * w
                    acc[i] = acc[i] + p

            if (i + 1) % 25 == 0 or (i + 1) == n:
                print('  scale=%s %d/%d  %.1fs'
                      % (sc, i + 1, n, time.time() - t0), flush=True)

    avg = [a / totals for a in acc]
    results = [(a, True, True) for a in avg]

    import contextlib
    import io
    import re as _re
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        out = dataset.evaluate(results, metric='mIoU')
    txt = buf.getvalue()
    print(txt, flush=True)

    per_class = {}
    for line in txt.splitlines():
        mm = _re.match(r'^(bg|EX|HE|SE|MA)\s+([0-9.]+|nan)\s', line.strip() + ' ')
        if mm:
            per_class[mm.group(1)] = mm.group(2)

    summary = {
        'models': [{'config': c, 'ckpt': k, 'weight': w} for c, k, w in specs],
        'load_report': load_report,
        'flip_tta': bool(args.flip),
        'scales': [s for s in scales],
        'split': args.split,
        'n_images': n,
        'mIoU': float(out.get('mIoU')),
        'mF1': float(out.get('mF1')),
        'mPPV': float(out.get('mPPV')),
        'mS': float(out.get('mS')),
        'mAUPR': float(out.get('mAUPR')),
        'per_class_iou': per_class,
    }
    with open(args.out, 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print('RESULT ' + json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
