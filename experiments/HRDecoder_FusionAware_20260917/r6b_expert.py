# -*- coding: utf-8 -*-
"""Round-6b: MA expert, revised recipe (diagnosis-driven, pre-registered).

What failed in R6 P2 (evidence): 512x512 native patches (MA fraction ~1e-4)
trained with Dice + 0.5*BCE collapsed to the null solution -- loss stuck at
~1.00 with val patch IoU 0.0000 for 1000+ steps, because the BCE gradient is
dominated by ~10^5 negative pixels.

Authorised revision (user approval 2026-09-18 16:34; recorded in
DEVIATIONS.jsonl), changing ONLY the three items implicated by that
diagnosis:
    patch      512 -> 256          (4x higher positive fraction)
    loss       Dice + 0.5*BCE -> Tversky(alpha=0.3, beta=0.7)  (the standard
                                    imbalance-robust objective for tiny
                                    lesions: beta weights false negatives)
    steps      3000 -> 2000        (smaller patches, same wall-clock)
UNCHANGED: the U-Net topology, the native-resolution patch采样 with a
positive-pixel floor, the alpha-fusion into the champion MA channel
(alpha in {0.3,0.5,0.7,1.0} swept on val), and the gate
MA val IoU >= champion + 0.010 -> test confirmation.
"""
import os
import time

import numpy as np
import torch
import torch.nn as nn

import run_protocol as rp
import r6_expert as X
import r6_lib
import runtime as rt

PATCH = 256
STRIDE = 128
STEPS = 2000
BATCH = 8
LR = 1e-3
MIN_POS_PX = 32          # ~0.05% of a 256x256 window
TVERSKY_ALPHA = 0.3      # weight on false positives
TVERSKY_BETA = 0.7       # weight on false negatives (tiny-lesion priority)
VAL_EVERY = 250


def tversky_loss(logit, target, alpha=TVERSKY_ALPHA, beta=TVERSKY_BETA,
                 eps=1.0):
    p = torch.sigmoid(logit)
    p = p.reshape(-1)
    y = target.reshape(-1)
    tp = (p * y).sum()
    fp = (p * (1 - y)).sum()
    fn = ((1 - p) * y).sum()
    return 1.0 - (tp + eps) / (tp + alpha * fp + beta * fn + eps)


def build_plan(settings, run_dir, log, rng):
    path = os.path.join(run_dir, 'r6b_patch_plan.json')
    if os.path.isfile(path):
        import json
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    import json
    data_root = settings['data_root']
    stems = [os.path.splitext(f)[0] for f in rp.list_split(data_root,
                                                           'train')]
    plan, n_pos = {}, 0
    for stem in stems:
        _, lp = X._paths(data_root, 'train', stem)
        lab = X.load_native_label(lp)
        pos, neg = r6_lib.ma_patch_plan(lab, PATCH, 8, 8, rng,
                                        min_pos_px=MIN_POS_PX)
        plan[stem] = {'pos': pos, 'neg': neg}
        n_pos += 1 if pos else 0
    log('R6b patch plan: %d images, %d with >=%d-pixel MA windows'
        % (len(plan), n_pos, MIN_POS_PX))
    r6_lib.atomic_json(path, plan)
    return plan


def _batch(plan, stems, rng, data_root):
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
        ip, lp = X._paths(data_root, 'train', stem)
        imgs.append(X._crop_rgb(ip, y, x, PATCH).transpose(2, 0, 1))
        gts.append(X._crop_lab(lp, y, x, PATCH)[None])
    if not imgs:
        return None, None
    return (torch.from_numpy(np.stack(imgs)).cuda(),
            torch.from_numpy(np.stack(gts)).cuda())


def val_patch_iou(model, patch_set, settings):
    tc = tp = fp = fn = 0
    model.eval()
    with torch.no_grad():
        for stem, y, x in patch_set:
            ip, lp = X._paths(settings['data_root'], 'val', stem)
            img = X._crop_rgb(ip, y, x, PATCH)
            gt = X._crop_lab(lp, y, x, PATCH)
            t = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).cuda()
            p = torch.sigmoid(model(t))[0, 0].cpu().numpy() > 0.5
            g = gt > 0.5
            tp += int(np.logical_and(p, g).sum())
            fp += int(np.logical_and(p, ~g).sum())
            fn += int(np.logical_and(~p, g).sum())
    return r6_lib.iou(tp, fp, fn)


def val_patch_set(settings, rng, per_image=3, n_images=20):
    stems = [os.path.splitext(f)[0]
             for f in rp.list_split(settings['data_root'], 'val')][:n_images]
    out = []
    for stem in stems:
        _, lp = X._paths(settings['data_root'], 'val', stem)
        lab = X.load_native_label(lp)
        pos, neg = r6_lib.ma_patch_plan(lab, PATCH, per_image, per_image, rng,
                                        min_pos_px=MIN_POS_PX)
        out += [(stem, y, x) for y, x in (pos + neg)]
    return out


def train(settings, run_dir, plan, log, deadline, steps=STEPS):
    data_root = settings['data_root']
    torch.manual_seed(r6_lib.SEED)
    rng = np.random.RandomState(r6_lib.SEED)
    model = X.UNet().cuda()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps,
                                                       eta_min=LR / 10.0)
    pset = val_patch_set(settings, rng)
    stems = [s for s, v in plan.items() if v['pos'] or v['neg']]
    log('R6b: %d train images with windows; %d val patches; Tversky(%.1f,%.1f)'
        % (len(stems), len(pset), TVERSKY_ALPHA, TVERSKY_BETA))
    curve, best = [], {'iou': -1.0, 'state': None, 'step': None}
    t0 = time.time()
    for step in range(steps):
        if time.time() > deadline:
            raise r6_lib.PhaseTimeout('R6b expert training (budget)')
        model.train()
        t, g = _batch(plan, stems, rng, data_root)
        if t is None:
            continue
        opt.zero_grad()
        loss = tversky_loss(model(t), g)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        sched.step()
        if float(loss) != float(loss):
            raise RuntimeError('NaN loss at R6b step %d' % step)
        if (step + 1) % 250 == 0:
            log('R6b step %d/%d loss %.4f (%.0fs)'
                % (step + 1, steps, float(loss), time.time() - t0))
        if (step + 1) % VAL_EVERY == 0 or step + 1 == steps:
            vi = val_patch_iou(model, pset, settings)
            curve.append({'step': step + 1, 'val_patch_iou': vi})
            log('R6b step %d val patch IoU %.4f' % (step + 1, vi))
            if vi > best['iou']:
                best = {'iou': vi, 'step': step + 1,
                        'state': {k: v.detach().cpu().clone()
                                  for k, v in model.state_dict().items()}}
    if best['state'] is None:
        raise RuntimeError('R6b training produced no validation point')
    model.load_state_dict(best['state'])
    model.eval()
    out_dir = os.path.join(run_dir, 'expert')
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, 'ma_expert_r6b.pth')
    torch.save({'state_dict': model.state_dict(), 'best_step': best['step'],
                'val_patch_iou': best['iou'], 'patch': PATCH,
                'stride': STRIDE, 'loss': 'tversky(0.3,0.7)',
                'steps': steps, 'seed': r6_lib.SEED}, path)
    r6_lib.atomic_json(os.path.join(out_dir, 'r6b_curve.json'),
                       {'curve': curve, 'best': best['step'],
                        'val_patch_iou': best['iou'],
                        'recipe': {'patch': PATCH, 'stride': STRIDE,
                                   'loss': 'tversky(0.3,0.7)',
                                   'min_pos_px': MIN_POS_PX,
                                   'steps': steps}})
    log('R6b DONE best step %d patch IoU %.4f -> %s'
        % (best['step'], best['iou'], path))
    return {'model': model, 'path': path, 'best_step': best['step'],
            'val_patch_iou': best['iou'], 'curve': curve,
            'sha256': r6_lib.sha256_file(path)}


def full_prob(model, rgb_path):
    """Sliding window at native resolution with the R6b patch/stride."""
    img = X.load_native_rgb(rgb_path)
    h, w = img.shape[:2]
    acc = np.zeros((h, w), np.float32)
    cnt = np.zeros((h, w), np.float32)
    with torch.no_grad():
        for y, x in r6_lib.window_origins(h, w, PATCH, STRIDE):
            t = torch.from_numpy(
                img[y:y + PATCH, x:x + PATCH].transpose(2, 0, 1)
            ).unsqueeze(0).cuda()
            p = torch.sigmoid(model(t))[0, 0].cpu().numpy()
            acc[y:y + PATCH, x:x + PATCH] += p
            cnt[y:y + PATCH, x:x + PATCH] += 1.0
    return acc / np.maximum(cnt, 1e-6)
