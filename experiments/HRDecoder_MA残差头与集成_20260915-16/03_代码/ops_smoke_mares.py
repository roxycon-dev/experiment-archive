#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P1 残差头预检：结构不变量 + 恒等性 + 冻结是否真的成立。

检验项：
  1. 只有 ma_refiner.* 可训练，其余全部冻结
  2. 恒等性：HRPlusMAResidual 在 0 步时输出 == HRPlusRefinement(D) 输出（逐位）
  3. 一步训练后，冻结主干指纹不变
  4. 一步训练后，输出里 EX/HE/SE 三通道仍与 D 逐位相同（只有 MA 变了）
  5. 梯度确实流到 ma_refiner，且没有流到冻结主干
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
from mmseg.datasets import build_dataset
from mmseg.models import build_segmentor

REPO = '/root/autodl-tmp/hrplus/repo'
os.chdir(REPO)
sys.path.insert(0, REPO)
import hrplus_patch  # noqa
import ma_residual   # noqa


def unwrap(x):
    n = 0
    while hasattr(x, 'data') and n < 8:
        x = x.data
        n += 1
    return x


def as_img_tensor(x):
    x = unwrap(x)
    while isinstance(x, (list, tuple)):
        x = unwrap(x[0])
    assert torch.is_tensor(x), type(x)
    return x


def as_meta_list(x):
    x = unwrap(x)
    if isinstance(x, dict):
        return [x]
    out = []
    for e in x:
        out.extend([e] if isinstance(e, dict) else [unwrap(e)])
    return out


def forward_prob(model, img, meta):
    with torch.no_grad():
        out = model(return_loss=False, rescale=True,
                    img=[img.unsqueeze(0).cuda()], img_metas=[meta])
    seg = out[0]
    return np.asarray(seg[0] if isinstance(seg, (tuple, list)) else seg,
                      dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--ref-config',
                    default='configs/lesion/exp/hrplus_patch_dice.py')
    ap.add_argument('--ref-ckpt', default=None,
                    help='若给出，参照模型加载该 checkpoint（用于与 D@2000 对比）')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    rep = {'config': args.config, 'checks': {}}
    cfg = Config.fromfile(args.config)
    ds_cfg = copy.deepcopy(cfg.data['val'])
    ds_cfg.test_mode = True
    ds = build_dataset(ds_cfg)
    s = ds[0]
    img = as_img_tensor(s['img'])
    meta = [dict(m) for m in as_meta_list(s['img_metas'])]
    gt = s['gt_semantic_seg'] if 'gt_semantic_seg' in s else None

    print('== 构建 HRPlusMAResidual ==', flush=True)
    model = build_segmentor(cfg.model).cuda().eval()

    # ---- 1) 冻结检查 ----
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    nontrain = [n for n, p in model.named_parameters() if not p.requires_grad]
    bad = [n for n in trainable if not n.startswith('ma_refiner.')]
    rep['checks']['1_only_head_trainable'] = {
        'trainable_count': len(trainable),
        'frozen_count': len(nontrain),
        'violations': bad[:10],
        'pass': len(bad) == 0 and len(trainable) > 0,
    }
    print('1) 可训练 %d 个 / 冻结 %d 个 / 违规 %d'
          % (len(trainable), len(nontrain), len(bad)), flush=True)

    # ---- 2) 恒等性：与 D 逐位相同 ----
    print('== 构建参照模型 ==', flush=True)
    ref_cfg = Config.fromfile(args.ref_config)
    ref_model = build_segmentor(ref_cfg.model).cuda().eval()
    if args.ref_ckpt:
        from mmcv.runner import load_checkpoint
        load_checkpoint(ref_model, args.ref_ckpt, map_location='cpu',
                        strict=False)
        print('   参照已加载:', args.ref_ckpt, flush=True)

    p_new = forward_prob(model, img, meta)
    p_ref = forward_prob(ref_model, img, meta)
    diff = float(np.abs(p_new - p_ref).max())
    rep['checks']['2_identity_at_init'] = {
        'max_abs_diff': diff, 'pass': diff == 0.0}
    print('2) 初始输出与 D 的最大绝对差 = %.3e  (需为 0)' % diff, flush=True)

    # ---- 3/4/5) 一步训练（用 train pipeline 的样本，含 gt）----
    print('== 取一个 train 样本 ==', flush=True)
    dtr = copy.deepcopy(cfg.data['train'])
    dtr.test_mode = False
    ds_tr = build_dataset(dtr)
    s_tr = ds_tr[0]
    img_tr = as_img_tensor(s_tr['img'])
    meta_tr = [dict(m) for m in as_meta_list(s_tr['img_metas'])]
    gt = unwrap(s_tr['gt_semantic_seg'])
    assert torch.is_tensor(gt), type(gt)
    gt_t = gt.cuda()
    while gt_t.dim() < 4:
        gt_t = gt_t.unsqueeze(0)
    print('   train img', tuple(img_tr.shape), 'gt', tuple(gt_t.shape), flush=True)

    fp_before = model.frozen_base_fingerprint()
    model.train()
    losses = model.forward_train(img_tr.unsqueeze(0).cuda(), meta_tr, gt_t)
    names = sorted(k for k in losses if 'loss' in k)
    loss = sum(losses[k] for k in names)
    rep['checks']['losses'] = {'names': names,
                               'values': {k: float(losses[k]) for k in names},
                               'finite': bool(torch.isfinite(loss))}
    print('3) 一步 loss 项 =', {k: round(float(losses[k]), 4) for k in names},
          flush=True)
    loss.backward()

    fp_after = model.frozen_base_fingerprint()
    rep['checks']['3_frozen_base_unchanged'] = {
        'before': fp_before[:16], 'after': fp_after[:16],
        'pass': fp_before == fp_after}
    print('4) 冻结主干指纹 %s -> %s  (%s)'
          % (fp_before[:12], fp_after[:12],
             'PASS' if fp_before == fp_after else 'FAIL'), flush=True)

    head_grad = sum(float(p.grad.abs().sum()) for n, p in model.named_parameters()
                    if n.startswith('ma_refiner.') and p.grad is not None)
    base_grad = sum(float(p.grad.abs().sum()) for n, p in model.named_parameters()
                    if not n.startswith('ma_refiner.') and p.grad is not None)
    rep['checks']['5_grad_flow'] = {
        'head_grad_sum': head_grad, 'base_grad_sum': base_grad,
        'pass': head_grad > 0 and base_grad == 0}
    print('5) 头的梯度总量 %.4f / 主干梯度总量 %.4f (需 主干=0)'
          % (head_grad, base_grad), flush=True)

    # 走一步优化器，再看 EX/HE/SE 是否仍与 D 一致
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                          lr=1e-3, momentum=0.9)
    opt.step()
    model.eval()
    p_after = forward_prob(model, img, meta)
    d_exhe = float(np.abs(p_after[:3] - p_ref[:3]).max())
    d_ma = float(np.abs(p_after[3] - p_ref[3]).max())
    rep['checks']['4_other_classes_untouched'] = {
        'max_diff_EX_HE_SE': d_exhe, 'max_diff_MA': d_ma,
        'pass': d_exhe == 0.0 and d_ma > 0.0}
    print('6) 一步后 EX/HE/SE 与 D 的差 = %.3e (需 0)；MA 的差 = %.3e (需 >0)'
          % (d_exhe, d_ma), flush=True)

    rep['status'] = 'success' if all(
        v.get('pass', True) for v in rep['checks'].values()
        if isinstance(v, dict) and 'pass' in v) else 'FAILED'
    rep['versions'] = {'torch': torch.__version__}
    with open(args.out, 'w') as f:
        json.dump(rep, f, indent=2, ensure_ascii=False)
    print('STATUS =', rep['status'])
    print('RESULT ' + json.dumps(rep['checks'], ensure_ascii=False))


if __name__ == '__main__':
    main()
