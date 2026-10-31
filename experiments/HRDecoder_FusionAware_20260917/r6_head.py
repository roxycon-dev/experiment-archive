# -*- coding: utf-8 -*-
"""Round-6 P1: per-class correction-head training on cached champion
features (H4 recipe, class-generic).

Facts anchored to the reference implementation (runtime.CorrectionHead /
glm_ref_h5_ensemble_corrector.Head -- verified identical):
    Head: Conv2d(7,16,3,1,1)-ReLU-Conv2d(16,16,3,1,1)-ReLU-Conv2d(16,1,1)
    out : 6.0 * tanh(net(f7));  p_new = sigmoid(logit(pf_c) + out)
    f7  : [img_f, logit(pf_c), logit(ps_c), logit(pr_c), ps_c - pr_c]

Design decisions (pre-registered):
  * final conv is ZERO-initialised -> at step 0 the head is exactly the
    identity p_new == pf_c (self-check asserted before training);
  * features are stored fp16; the SAME rounding is applied at inference
    (round_features) so training and inference distributions match;
  * loss = Dice on p_new vs (gt == class), faithful to the H-series heads;
  * validation = full-image pass over the val feature cache, exporting
    per-image TP/FP/FN counts for class c only (the gate + bootstrap need
    nothing else: the other three classes are unchanged by construction).
"""
import json
import os
import time

import numpy as np
import torch

import r6_lib
import r6_features
import runtime as rt

CROP = 512
LABEL_LRU = 48          # decoded label cache (13.7GB if ALL 454 were held)
STEPS = 3000
BATCH = 8
LR = 1e-3
LR_MIN = 1e-4
VAL_EVERY = 500
POS_RATIO = 0.6          # fraction of crop samples centred on the class


class _LabelLRU(object):
    """Decoded-label cache.  Holding all 454 labels costs 13.7GB of RAM;
    an LRU keeps the hot set small while avoiding a full PNG decode per crop
    (8 crops/step x 3000 steps would otherwise dominate)."""

    def __init__(self, maxlen=LABEL_LRU):
        self.maxlen = int(maxlen)
        self.d, self.order = {}, []

    def get(self, data_root, split, stem):
        key = (split, stem)
        if key in self.d:
            return self.d[key]
        lab = r6_lib.load_label(data_root, split, stem)
        self.d[key] = lab
        self.order.append(key)
        while len(self.order) > self.maxlen:
            self.d.pop(self.order.pop(0), None)
        return lab


_LABELS = _LabelLRU()


def round_features(t):
    """fp16 storage rounding, applied identically in training and at
    inference (declared, not silent)."""
    return t.half().float()


def f7_from_arrays(img_f16, fields_f16):
    """(img_f16 (3,h,w), fields_f16 (3,h,w)) -> (1,7,h,w) float32 tensor.

    Both input groups are fp16-rounded -- the single declared parity
    mechanism shared with inference (runtime.corrected_class_prob applies
    the same rounding when the head sets r6_round_inputs).  Logits come from
    rt._logit (single source of truth)."""
    img3 = round_features(torch.as_tensor(np.asarray(img_f16,
                                                     dtype=np.float32)))
    t = round_features(torch.as_tensor(np.asarray(fields_f16,
                                                  dtype=np.float32)))
    pf, ps, pr = t[0:1], t[1:2], t[2:3]
    return torch.cat([img3, rt._logit(pf), rt._logit(ps), rt._logit(pr),
                      ps - pr], dim=0).unsqueeze(0)


def _f7_on_device(img_f16, fields_f16, head):
    """f7 built on the head's device (the ONLY device-move site)."""
    f7 = f7_from_arrays(img_f16, fields_f16)
    return f7.to(next(head.parameters()).device)


def _dice_loss(p, target, eps=1.0):
    """Soft Dice on probabilities vs binary target (H-series recipe)."""
    p = p.reshape(-1)
    y = target.reshape(-1)
    inter = (p * y).sum()
    return 1.0 - (2.0 * inter + eps) / (p.sum() + y.sum() + eps)


def _crop(img_u8, fields, label, y, x, size):
    return (np.asarray(img_u8[:, y:y + size, x:x + size]),
            np.asarray(fields[:, y:y + size, x:x + size]),
            np.asarray(label[y:y + size, x:x + size]))


def _sample_crop(img_u8, fields, label, cls_idx, rng, size=CROP):
    """Class-biased crop: with prob POS_RATIO a window containing the class,
    else a random window."""
    h, w = label.shape[:2]
    size = int(min(size, h, w))
    if rng.random() < POS_RATIO:
        pos, _ = r6_lib.label_patch_plan(label, cls_idx, size, 1, 0, rng,
                                         min_pos_px=1)
        if pos:
            y, x = pos[0]
            return _crop(img_u8, fields, label, y, x, size)
    y = rng.randint(0, h - size + 1) if h > size else 0
    x = rng.randint(0, w - size + 1) if w > size else 0
    return _crop(img_u8, fields, label, y, x, size)


def _val_full_image(img_u8, fields, label, head, cls_idx):
    """Full-image corrected probability for class c + per-image counts."""
    dev = next(head.parameters()).device
    f7 = _f7_on_device(img_u8, fields, head)
    with torch.no_grad():
        delta = head(f7)[0, 0]
        pf = round_features(torch.as_tensor(
            np.asarray(fields[0:1], dtype=np.float32))).to(dev)
        p_new = torch.sigmoid(rt._logit(pf) + delta).squeeze(0).cpu().numpy()
    gt = np.asarray(label)
    pred = p_new > 0.5
    g = gt == cls_idx
    ig = gt == 255
    tp = int(np.logical_and(pred, g).sum())
    fp = int(np.logical_and(pred, np.logical_and(~g, ~ig)).sum())
    fn = int(np.logical_and(~pred, g).sum())
    return p_new, [tp, fp, fn]


def evaluate_val(cache_val, data_root, head, cls, max_images=None):
    """per_image counts for class `cls` over the val feature cache."""
    cls_idx = rt.CLASSES.index(cls) + 1        # label value 1..4
    per_image = {}
    stems = r6_features.stems_of(cache_val)
    if max_images:
        stems = stems[:max_images]
    for stem in stems:
        img_u8, fields = r6_features.load_entry(cache_val, stem)
        label = _LABELS.get(data_root, 'val', stem)   # any split, see r6_lib
        _, cnt = _val_full_image(img_u8, fields, label, head, cls_idx)
        per_image[stem] = {cls: cnt}
    return per_image


def counts_iou_class(per_image, cls):
    tp = fp = fn = 0
    for s in per_image:
        v = per_image[s][cls]
        tp += v[0]
        fp += v[1]
        fn += v[2]
    return r6_lib.iou(tp, fp, fn), [tp, fp, fn]


def train_head(settings, run_dir, cls, cache_train, cache_val, log, deadline,
               steps=STEPS, val_every=VAL_EVERY, val_max_images=None,
               out_dir_override=None, save=True):
    """Trains one correction head for `cls`.  Returns dict with the head,
    val counts, curve and the gate inputs."""
    cls_idx = rt.CLASSES.index(cls) + 1
    data_root = settings['data_root']
    torch.manual_seed(r6_lib.SEED)
    rng = np.random.RandomState(r6_lib.SEED)
    head = rt.CorrectionHead().cuda()
    # zero-init the final conv -> identity at step 0 (asserted below)
    with torch.no_grad():
        head.net[4].weight.zero_()
        head.net[4].bias.zero_()
    train_stems = r6_features.stems_of(cache_train)

    def _lab(stem):
        return _LABELS.get(data_root, 'train', stem)
    # ---- identity self-check (step 0) ----
    s0 = train_stems[0]
    img0, fld0 = r6_features.load_entry(cache_train, s0)
    a_i, a_f, _l = _crop(img0, fld0, _lab(s0), 0, 0, 256)
    f7 = _f7_on_device(a_i, a_f, head)
    with torch.no_grad():
        d0 = head(f7)
    if float(d0.abs().max()) > 1e-6:
        raise RuntimeError('zero-init head is not identity: max|delta|=%.3g'
                           % float(d0.abs().max()))
    log('H6[%s] identity self-check OK (zero-init head -> delta==0)' % cls)
    opt = torch.optim.Adam(head.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps,
                                                       eta_min=LR_MIN)
    curve = []
    best = {'miou': -1.0, 'state': None, 'step': None, 'val': None}
    t0 = time.time()
    for step in range(steps):
        if time.time() > deadline:
            raise r6_lib.PhaseTimeout('head training %s (phase budget)'
                                      % cls)
        head.train()
        opt.zero_grad()
        xs, ys = [], []
        for b in range(BATCH):
            stem = train_stems[rng.randint(0, len(train_stems))]
            img_u8, fields = r6_features.load_entry(cache_train, stem)
            a_i, a_f, lab = _sample_crop(img_u8, fields, _lab(stem),
                                         cls_idx, rng)
            xs.append(f7_from_arrays(a_i, a_f)[0])   # stacked then .cuda()
            ys.append(torch.as_tensor((np.asarray(lab) == cls_idx)
                                      .astype(np.float32)))
        f7 = torch.stack(xs, 0).cuda()
        target = torch.stack(ys, 0).unsqueeze(1).cuda()
        pf = f7[:, 3:4]
        p_new = torch.sigmoid(rt._logit(pf) + head(f7))
        loss = _dice_loss(p_new, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 5.0)
        opt.step()
        sched.step()
        if float(loss) != float(loss):
            raise RuntimeError('NaN loss at step %d' % step)
        if (step + 1) % 250 == 0:
            log('H6[%s] step %d/%d loss %.4f (%.0fs)'
                % (cls, step + 1, steps, float(loss), time.time() - t0))
        if (step + 1) % max(1, int(val_every)) == 0 or step + 1 == steps:
            head.eval()
            per_image = evaluate_val(cache_val, data_root, head, cls,
                                     max_images=val_max_images)
            miou_c, tot_c = counts_iou_class(per_image, cls)
            curve.append({'step': step + 1, 'val_cls_iou': miou_c,
                          'val_counts': tot_c})
            log('H6[%s] step %d val %s IoU %.4f (counts %s)'
                % (cls, step + 1, cls, miou_c, tot_c))
            if miou_c > best['miou']:
                best = {'miou': miou_c,
                        'state': {k: v.detach().cpu().clone()
                                  for k, v in head.state_dict().items()},
                        'step': step + 1, 'val': per_image}
    if best['state'] is None:
        raise RuntimeError('head training produced no validation point')
    head.load_state_dict(best['state'])
    head.eval()
    if not save:
        log('H6[%s] rehearsal complete (nothing written)' % cls)
        return {'head': head, 'val_counts': best['val'],
                'val_iou': best['miou'], 'best_step': best['step'],
                'curve': curve, 'path': None, 'sha256': None}
    out_dir = out_dir_override or os.path.join(run_dir, 'heads')
    os.makedirs(out_dir, exist_ok=True)
    head_path = os.path.join(out_dir, 'head_%s.pth' % cls)
    torch.save({'head': {k: v for k, v in head.state_dict().items()},
                'cls': cls, 'best_step': best['step'],
                'val_cls_iou': best['miou'],
                'seed': r6_lib.SEED, 'steps': steps, 'crop': CROP},
               head_path)
    r6_lib.atomic_json(os.path.join(out_dir, 'head_%s_val_counts.json' % cls),
                       {'split': 'val', 'cls': cls,
                        'per_image': best['val'],
                        'val_cls_iou': best['miou'],
                        'best_step': best['step'], 'curve': curve})
    log('H6[%s] DONE best step %d val %s IoU %.4f -> %s'
        % (cls, best['step'], cls, best['miou'], head_path))
    return {'head': head, 'val_counts': best['val'], 'val_iou': best['miou'],
            'best_step': best['step'], 'curve': curve,
            'path': head_path, 'sha256': r6_lib.sha256_file(head_path)}
