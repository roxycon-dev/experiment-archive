#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P0: BN 运行统计重估（Codex 优先级一里最便宜的一条）。

动机：D/L/E 的学生分支在训练时 freeze_bn=True，BN running stats 一直沿用
来源 checkpoint（B@3000 ← E4@36000）的数值，而学生实际处理的是 1024×1024
裁块 —— 分布与来源不一致。这里用训练集实际分布重估 BN 统计。

做法：momentum=None（累计平均）+ reset_running_stats()，只把 BN 设为 train，
其余保持 eval；对训练集跑一遍 forward（no_grad），然后另存 checkpoint。

注意：这是零训练成本的诊断/干预，不改任何可学习参数。
"""
import argparse
import copy
import json
import os
import sys
import time

import torch
from mmcv import Config
from mmcv.parallel import collate
from mmcv.runner import load_checkpoint
from mmseg.datasets import build_dataset
from mmseg.models import build_segmentor

REPO = '/root/autodl-tmp/hrplus/repo'
os.chdir(REPO)
sys.path.insert(0, REPO)
import hrplus_patch  # noqa
try:
    import ma_residual  # noqa
except Exception:
    pass


def unwrap(x):
    n = 0
    while hasattr(x, 'data') and n < 8:
        x = x.data
        n += 1
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out-ckpt', required=True)
    ap.add_argument('--report', required=True)
    ap.add_argument('--which', default='student', choices=['student', 'all'])
    ap.add_argument('--max-batches', type=int, default=0)  # 0 = 全量一遍
    args = ap.parse_args()

    t0 = time.time()
    old_fp = None
    rep = {'config': args.config, 'ckpt': args.ckpt, 'which': args.which}

    cfg = Config.fromfile(args.config)
    model = build_segmentor(cfg.model).cuda()
    load_checkpoint(model, args.ckpt, map_location='cpu', strict=False)
    model.eval()

    # 收集要重估的 BN
    def want(name):
        if args.which == 'all':
            return True
        return not name.startswith('reference.')

    bns = [(n, m) for n, m in model.named_modules()
           if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)
           and want(n)]
    rep['bn_count'] = len(bns)
    print('待重估 BN 模块数 =', len(bns), flush=True)

    # 记录可学习参数的指纹，证明「没动任何 weight」
    import hashlib
    h = hashlib.sha256()
    for n, p in sorted(model.named_parameters()):
        h.update(n.encode())
        h.update(p.detach().cpu().contiguous().numpy().tobytes())
    before_param_fp = h.hexdigest()

    # 只把 BN 设为 train 并重置统计；其余保持 eval
    for _, m in bns:
        m.reset_running_stats()
        m.momentum = None      # 累计平均
        m.train()
    model.eval()
    for _, m in bns:
        m.train()

    dtr = copy.deepcopy(cfg.data['train'])
    dtr.test_mode = False
    ds = build_dataset(dtr)
    bs = cfg.data.samples_per_gpu
    n = len(ds)
    nb = (n + bs - 1) // bs
    if args.max_batches:
        nb = min(nb, args.max_batches)
    print('训练集 %d 张 / batch %d / 共 %d 个 batch' % (n, bs, nb), flush=True)

    model.eval()
    for _, m in bns:
        m.train()

    seen = 0
    tt = time.time()
    for bi in range(nb):
        batch = [ds[i] for i in range(bi * bs, min((bi + 1) * bs, n))]
        data = collate(batch, samples_per_gpu=bs)
        img = unwrap(data['img'])[0].cuda()
        metas = unwrap(data['img_metas'])[0]
        with torch.no_grad():
            model.encode_decode(img, metas)
        seen += img.shape[0]
        if (bi + 1) % 25 == 0 or bi + 1 == nb:
            print('  %d/%d batchs, %d imgs, %.1fs'
                  % (bi + 1, nb, seen, time.time() - tt), flush=True)

    # 复原 eval
    for _, m in bns:
        m.eval()
    model.eval()

    # 校验参数未被改动
    h2 = hashlib.sha256()
    for n2, p in sorted(model.named_parameters()):
        h2.update(n2.encode())
        h2.update(p.detach().cpu().contiguous().numpy().tobytes())
    after_param_fp = h2.hexdigest()
    rep['learnable_params_unchanged'] = (before_param_fp == after_param_fp)
    rep['param_fp'] = before_param_fp[:16]
    print('可学习参数指纹 %s -> %s  (%s)'
          % (before_param_fp[:12], after_param_fp[:12],
             'PASS' if before_param_fp == after_param_fp else 'FAIL'),
          flush=True)

    # BN 变化的幅度统计
    changed = {}
    for name, m in bns:
        if m.running_mean is not None:
            changed[name] = {
                'mean_abs': float(m.running_mean.abs().mean()),
                'var_mean': float(m.running_var.mean()),
            }
    rep['bn_after'] = {k: v for k, v in list(changed.items())[:5]}
    rep['bn_changed_count'] = len(changed)

    torch.save({'state_dict': model.state_dict(),
                'meta': {'source': args.ckpt, 'renorm': args.which,
                         'batches': nb, 'imgs': seen}},
               args.out_ckpt)
    rep['out_ckpt'] = args.out_ckpt
    rep['elapsed_s'] = round(time.time() - t0, 1)
    rep['status'] = 'success'
    with open(args.report, 'w') as f:
        json.dump(rep, f, indent=2, ensure_ascii=False)
    print('DONE', json.dumps({k: rep[k] for k in
                              ('bn_count', 'bn_changed_count',
                               'learnable_params_unchanged', 'elapsed_s')}))


if __name__ == '__main__':
    main()
