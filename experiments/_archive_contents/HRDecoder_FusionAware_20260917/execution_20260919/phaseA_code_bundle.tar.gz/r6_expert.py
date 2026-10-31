# -*- coding: utf-8 -*-
"""Round-6 P2: MA-native-resolution expert (framework-level addition).

Mechanism (why this axis): the whole champion stack runs on a 2048x2048
resize of the original image, which resamples MA lesions (5-15 px) and is
the class with the highest residual error (champion test MA IoU 28.66).
The expert is a small U-Net trained ONLY on MA-positive/negative patches at
NATIVE resolution from the existing training labels (zero new data), and is
fused into the MA channel:

    MA_fused(alpha) = alpha * expert + (1 - alpha) * champion_MA

alpha is swept on val only ({0.3, 0.5, 0.7, 1.0}); the pre-registered gate
is MA val IoU >= champion + 0.010.  Nothing else in the stack is touched.
"""
import json
import os
import time

import cv2
import numpy as np
import torch
import torch.nn as nn

import r6_lib
import run_protocol as rp
import runtime as rt

PATCH = 512
STRIDE = 256
STEPS = 3000
BATCH = 8
LR = 1e-3
ALPHAS = [0.3, 0.5, 0.7, 1.0]
VAL_EVERY = 500
VAL_PATCH_IMAGES = 20          # per-epoch patch-val images (cheap, fixed set)


class UNet(nn.Module):
    """Small 4-level U-Net (1.9M params) for native-resolution MA patches."""

    def __init__(self, base=32):
        super(UNet, self).__init__()

        def blk(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, 1, 1),
                                 nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                                 nn.Conv2d(o, o, 3, 1, 1),
                                 nn.BatchNorm2d(o), nn.ReLU(inplace=True))

        self.e1 = blk(3, base)
        self.e2 = blk(base, base * 2)
        self.e3 = blk(base * 2, base * 4)
        self.b = blk(base * 4, base * 8)
        self.pool = nn.MaxPool2d(2)
        self.u3 = nn.ConvTranspose2d(base * 8, base * 4, 2, 2)
        self.d3 = blk(base * 8, base * 4)
        self.u2 = nn.ConvTranspose2d(base * 4, base * 2, 2, 2)
        self.d2 = blk(base * 4, base * 2)
        self.u1 = nn.ConvTranspose2d(base * 2, base, 2, 2)
        self.d1 = blk(base * 2, base)
        self.out = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        b = self.b(self.pool(e3))
        d3 = self.d3(torch.cat([self.u3(b), e3], 1))
        d2 = self.d2(torch.cat([self.u2(d3), e2], 1))
        d1 = self.d1(torch.cat([self.u1(d2), e1], 1))
        return self.out(d1)


class _LRU(object):
    """Small decoded-image cache: native jpg decode is ~200ms, and the
    expert trainer draws 8 crops/step from random images -- without this the
    training run would be dominated by JPEG decoding (~80 min for 3000
    steps).  Memory bound: maxlen x ~15MB (RGB uint8)."""

    def __init__(self, maxlen=140):
        self.maxlen = int(maxlen)
        self.d = {}
        self.order = []

    def get(self, key, loader):
        if key in self.d:
            return self.d[key]
        val = loader()
        self.d[key] = val
        self.order.append(key)
        while len(self.order) > self.maxlen:
            old = self.order.pop(0)
            self.d.pop(old, None)
        return val


_RGB_LRU = _LRU(140)
_LAB_LRU = _LRU(200)


def load_native_rgb(path):
    """RGB float32 [0,1] at native resolution (BGR->RGB)."""
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError('cannot read image %s' % path)
    return np.ascontiguousarray(bgr[:, :, ::-1]).astype(np.float32) / 255.0


def cached_rgb(path):
    return _RGB_LRU.get(path, lambda: load_native_rgb(path))


def cached_label(path):
    return _LAB_LRU.get(path, lambda: load_native_label(path))


def load_native_label(path):
    lab = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if lab is None:
        raise RuntimeError('cannot read label %s' % path)
    if lab.ndim == 3:
        lab = lab[:, :, 0]
    return lab.astype(np.int64)


def _paths(data_root, split, stem):
    return (os.path.join(data_root, 'images', split, stem + '.jpg'),
            os.path.join(data_root, 'labels', split, stem + '.png'))


def build_patch_plan(settings, run_dir, log, rng):
    """Plan MA-positive/negative patch origins for every train image.
    Cached in r6_state (idempotent).  Returns {stem: {'pos': [...],
    'neg': [...]}}."""
    plan_path = os.path.join(run_dir, 'expert_patch_plan.json')
    if os.path.isfile(plan_path):
        with open(plan_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    data_root = settings['data_root']
    stems = [os.path.splitext(f)[0]
             for f in rp.list_split(data_root, 'train')]
    plan, n_pos_img = {}, 0
    for stem in stems:
        _, lp = _paths(data_root, 'train', stem)
        if not os.path.isfile(lp):
            raise RuntimeError('missing label %s' % lp)
        lab = cached_label(lp)
        pos, neg = r6_lib.ma_patch_plan(lab, PATCH, 4, 4, rng,
                                        min_pos_px=8)
        plan[stem] = {'pos': pos, 'neg': neg}
        n_pos_img += 1 if pos else 0
    log('expert patch plan: %d train images, %d with MA-positive windows'
        % (len(plan), n_pos_img))
    r6_lib.atomic_json(plan_path, plan)
    return plan


def _val_patch_set(settings, stems, rng, per_image=2):
    """Fixed small val patch set: pos+neg windows from val images."""
    data_root = settings['data_root']
    out = []
    for stem in stems[:VAL_PATCH_IMAGES]:
        _, lp = _paths(data_root, 'val', stem)
        lab = load_native_label(lp)
        pos, neg = r6_lib.ma_patch_plan(lab, PATCH, per_image, per_image,
                                        rng, min_pos_px=8)
        for y, x in (pos + neg):
            out.append((stem, y, x))
    return out


def _crop_rgb(path, y, x, size=PATCH):
    img = cached_rgb(path)
    h, w = img.shape[:2]
    y = int(min(y, h - size))
    x = int(min(x, w - size))
    return img[y:y + size, x:x + size]


def _crop_lab(path, y, x, size=PATCH):
    lab = cached_label(path)
    h, w = lab.shape[:2]
    y = int(min(y, h - size))
    x = int(min(x, w - size))
    return (lab[y:y + size, x:x + size] == 4).astype(np.float32)


def _loss(p_logit, target, eps=1.0):
    p = torch.sigmoid(p_logit)
    inter = (p * target).sum()
    dice = 1.0 - (2.0 * inter + eps) / (p.sum() + target.sum() + eps)
    bce = nn.functional.binary_cross_entropy_with_logits(p_logit, target)
    return dice + 0.5 * bce


def val_patch_iou(model, patch_set, settings, log_tag=''):
    data_root = settings['data_root']
    model.eval()
    tp = fp = fn = 0
    with torch.no_grad():
        for stem, y, x in patch_set:
            ip, lp = _paths(data_root, 'val', stem)
            img = _crop_rgb(ip, y, x)
            gt = _crop_lab(lp, y, x)
            t = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).cuda()
            p = torch.sigmoid(model(t))[0, 0].cpu().numpy() > 0.5
            g = gt > 0.5
            tp += int(np.logical_and(p, g).sum())
            fp += int(np.logical_and(p, ~g).sum())
            fn += int(np.logical_and(~p, g).sum())
    return r6_lib.iou(tp, fp, fn)


def train_expert(settings, run_dir, plan, log, deadline, steps=STEPS):
    data_root = settings['data_root']
    torch.manual_seed(r6_lib.SEED)
    rng = np.random.RandomState(r6_lib.SEED)
    model = UNet().cuda()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps,
                                                       eta_min=LR / 10.0)
    val_stems = [os.path.splitext(f)[0]
                 for f in rp.list_split(data_root, 'val')]
    patch_set = _val_patch_set(settings, val_stems, rng)
    stems = [s for s, v in plan.items() if v['pos'] or v['neg']]
    if not stems:
        raise RuntimeError('expert patch plan is empty')
    curve, best = [], {'iou': -1.0, 'state': None, 'step': None}
    t0 = time.time()
    for step in range(steps):
        if time.time() > deadline:
            raise r6_lib.PhaseTimeout('expert training (phase budget)')
        model.train()
        opt.zero_grad()
        imgs, gts = [], []
        for b in range(BATCH):
            stem = stems[rng.randint(0, len(stems))]
            use_pos = (b % 2 == 0) and plan[stem]['pos']
            cand = plan[stem]['pos'] if use_pos else plan[stem]['neg']
            if not cand:
                cand = plan[stem]['pos'] + plan[stem]['neg']
            if not cand:
                continue
            y, x = cand[rng.randint(0, len(cand))]
            ip, lp = _paths(data_root, 'train', stem)
            imgs.append(_crop_rgb(ip, y, x).transpose(2, 0, 1))
            gts.append(_crop_lab(lp, y, x)[None])
        if not imgs:
            continue
        t = torch.from_numpy(np.stack(imgs)).cuda()
        g = torch.from_numpy(np.stack(gts)).cuda()
        loss = _loss(model(t), g)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        sched.step()
        if float(loss) != float(loss):
            raise RuntimeError('NaN loss at expert step %d' % step)
        if (step + 1) % 250 == 0:
            log('M1 step %d/%d loss %.4f (%.0fs)'
                % (step + 1, steps, float(loss), time.time() - t0))
        if (step + 1) % VAL_EVERY == 0 or step + 1 == steps:
            vi = val_patch_iou(model, patch_set, settings)
            curve.append({'step': step + 1, 'val_patch_iou': vi})
            log('M1 step %d val patch IoU %.4f' % (step + 1, vi))
            if vi > best['iou']:
                best = {'iou': vi, 'step': step + 1,
                        'state': {k: v.detach().cpu().clone()
                                  for k, v in model.state_dict().items()}}
    if best['state'] is None:
        raise RuntimeError('expert training produced no validation point')
    model.load_state_dict(best['state'])
    model.eval()
    out_dir = os.path.join(run_dir, 'expert')
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, 'ma_expert.pth')
    torch.save({'state_dict': model.state_dict(), 'best_step': best['step'],
                'val_patch_iou': best['iou'], 'patch': PATCH,
                'stride': STRIDE, 'seed': r6_lib.SEED, 'steps': steps},
               path)
    r6_lib.atomic_json(os.path.join(out_dir, 'expert_curve.json'),
                       {'curve': curve, 'best': best['step'],
                        'val_patch_iou': best['iou']})
    log('M1 DONE best step %d val patch IoU %.4f -> %s (sha %s)'
        % (best['step'], best['iou'], path,
           r6_lib.sha256_file(path)[:16]))
    return {'model': model, 'path': path, 'val_patch_iou': best['iou'],
            'best_step': best['step'], 'curve': curve,
            'sha256': r6_lib.sha256_file(path)}


def load_expert(path):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    model = UNet()
    model.load_state_dict(ck['state_dict'], strict=True)
    return model.cuda().eval(), ck


def expert_full_prob(model, rgb_path, log=None):
    """Sliding-window MA probability at native resolution (average over
    overlaps; stride STRIDE, window PATCH, edge-clamped)."""
    img = load_native_rgb(rgb_path)
    h, w = img.shape[:2]
    if h < PATCH or w < PATCH:
        raise RuntimeError('image smaller than patch: %s' % rgb_path)
    acc = np.zeros((h, w), dtype=np.float32)
    cnt = np.zeros((h, w), dtype=np.float32)
    with torch.no_grad():
        for y, x in r6_lib.window_origins(h, w, PATCH, STRIDE):
            t = torch.from_numpy(
                img[y:y + PATCH, x:x + PATCH].transpose(2, 0, 1)
            ).unsqueeze(0).cuda()
            p = torch.sigmoid(model(t))[0, 0].cpu().numpy()
            acc[y:y + PATCH, x:x + PATCH] += p
            cnt[y:y + PATCH, x:x + PATCH] += 1.0
    return acc / np.maximum(cnt, 1e-6)


def counts_for_class(prob, gt, cls_idx):
    pred = prob > 0.5
    g = gt == cls_idx
    ig = gt == 255
    tp = int(np.logical_and(pred, g).sum())
    fp = int(np.logical_and(pred, np.logical_and(~g, ~ig)).sum())
    fn = int(np.logical_and(~pred, g).sum())
    return [tp, fp, fn]


def eval_split_with_expert(settings, champion, expert_model, alphas, split,
                           log, deadline, dump_counts=None):
    """ONE pass over `split`.

    For every image: champion.predict -> (4,H,W) probs -> per-class counts
    (any adopted correction heads included, since they live in the champion
    object), plus the expert's native-res MA prob.  Then for every alpha in
    `alphas` a fused MA count set is produced by re-thresholding
    alpha*expert + (1-alpha)*champ_MA on the same image grid.

    Returns {'champ': per_image(4 classes), 'fused': {alpha: per_image(MA
    only)}, 'champ_counts': {...}, 'fused_totals': {alpha: [tp,fp,fn]},
    'champ_ma_iou': float, 'fused_ma_iou': {alpha: float}}.
    """
    if expert_model is None and alphas:
        raise ValueError('alphas supplied without an expert model')
    import copy
    from mmseg.datasets import build_dataset
    data_root = settings['data_root']
    cfg = copy.deepcopy(settings['_cfg'].data[split])
    cfg.test_mode = True
    cfg.pop('split', None)
    cfg.img_dir = os.path.join(data_root, 'images', split)
    cfg.ann_dir = os.path.join(data_root, 'labels', split)
    ds = build_dataset(cfg)
    n = len(ds)
    expected = {'val': 76, 'test': 227}[split]
    if n != expected:
        raise RuntimeError('%s has %d items, expected %d' % (split, n, expected))
    champ_pi, fused_pi = {}, {a: {} for a in alphas}
    tot = {c: [0, 0, 0] for c in rt.CLASSES}
    fused_tot = {a: [0, 0, 0] for a in alphas}
    gt_maps = ds.get_gt_seg_maps()          # hoisted: O(n) per call otherwise
    t0 = time.time()
    for i in range(n):
        if time.time() > deadline:
            raise r6_lib.PhaseTimeout('expert eval %s (phase budget)' % split)
        s = ds[i]
        img = rt._unwrap_img(s['img']).unsqueeze(0).cuda()
        meta = rt._unwrap_meta(s['img_metas'])
        stem = os.path.splitext(os.path.basename(meta[0]['ori_filename']))[0]
        prob = np.asarray(champion.predict(img, meta).detach().cpu().numpy(),
                          dtype=np.float32)
        ori = tuple(meta[0]['ori_shape'][:2])
        if prob.shape[1:] != ori:
            raise RuntimeError('champion prob %r != ori %r'
                               % (prob.shape[1:], ori))
        gt = np.asarray(gt_maps[i]).astype(np.int64)
        if gt.shape != ori:
            gt = cv2.resize(gt.astype(np.uint8), (ori[1], ori[0]),
                            interpolation=cv2.INTER_NEAREST).astype(np.int64)
        img_path = os.path.join(data_root, 'images', split, stem + '.jpg')
        if expert_model is not None:
            ex = expert_full_prob(expert_model, img_path)
            if ex.shape != ori:
                ex = cv2.resize(ex, (ori[1], ori[0]),
                                interpolation=cv2.INTER_LINEAR)
        champ_pi[stem] = {}
        for c in range(4):
            cnt = counts_for_class(prob[c], gt, c + 1)
            champ_pi[stem][rt.CLASSES[c]] = cnt
            for j in range(3):
                tot[rt.CLASSES[c]][j] += cnt[j]
        for a in alphas:
            fused = a * ex + (1.0 - a) * prob[3]
            cnt = counts_for_class(fused, gt, 4)
            fused_pi[a][stem] = {'MA': cnt}
            for j in range(3):
                fused_tot[a][j] += cnt[j]
        del img, prob
        if expert_model is not None:
            del ex
        torch.cuda.empty_cache()
        if (i + 1) % 25 == 0:
            log('expert eval %s %d/%d (%.0fs)'
                % (split, i + 1, n, time.time() - t0))
    champ_iou = r6_lib.counts_iou(tot)
    fused_iou = {a: r6_lib.iou(*fused_tot[a]) for a in alphas}
    out = {'split': split, 'champ': champ_pi, 'fused': fused_pi,
           'champ_counts': tot, 'champ_iou': champ_iou,
           'fused_totals': {str(a): fused_tot[a] for a in alphas},
           'fused_ma_iou': {str(a): fused_iou[a] for a in alphas}}
    log('expert eval %s done: champion mIoU %.4f / MA %.4f; fused MA: %s'
        % (split, champ_iou['mIoU'], champ_iou['MA'],
           {str(a): round(fused_iou[a], 4) for a in alphas}))
    if dump_counts:
        r6_lib.atomic_json(dump_counts, {
            'split': split, 'miou': champ_iou['mIoU'],
            'per_class': {c: champ_iou[c] for c in rt.CLASSES},
            'global_counts': tot, 'per_image': champ_pi})
    return out
