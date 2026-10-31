# -*- coding: utf-8 -*-
"""Matched C/S continuation experiment for HRDecoder.

C: current 2048 -> rotate -> foreground-mixed 1024 crop.
S: same optimization, with half of samples chosen class -> image -> instance.
Validation always uses the independent student probability, never model.inference().
"""
import argparse
import json
import os
import random
import sys
import time

import cv2
import numpy as np
import torch


def atomic_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def load_settings(path):
    with open(path, encoding='utf-8') as f:
        s = json.load(f)
    required = ('repo', 'config', 'data_root', 'student_checkpoint',
                'reference_checkpoint')
    missing = [k for k in required if not s.get(k)]
    if missing:
        raise RuntimeError('missing settings: %s' % missing)
    s['_abs'] = {
        'student_checkpoint': os.path.abspath(s['student_checkpoint']),
        'reference_checkpoint': os.path.abspath(s['reference_checkpoint']),
    }
    return s


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def build_index(data_root):
    labdir = os.path.join(data_root, 'labels', 'train')
    imgdir = os.path.join(data_root, 'images', 'train')
    stems = sorted(os.path.splitext(x)[0] for x in os.listdir(imgdir)
                   if x.lower().endswith(('.jpg', '.jpeg', '.png')))
    by_class = {c: [] for c in range(1, 5)}
    for stem in stems:
        lab = cv2.imread(os.path.join(labdir, stem + '.png'), 0)
        if lab is None:
            raise RuntimeError('unreadable label ' + stem)
        for c in range(1, 5):
            if np.any(lab == c):
                by_class[c].append(stem)
    if len(stems) != 454 or any(not by_class[c] for c in by_class):
        raise RuntimeError('bad train index: %d %s' %
                           (len(stems), {c: len(v) for c, v in by_class.items()}))
    return stems, by_class


def choose_sample(arm, stems, by_class, rng):
    conditional = arm == 'S' and rng.rand() < 0.5
    if not conditional:
        return stems[int(rng.randint(len(stems)))], None
    cls = int(rng.randint(1, 5))
    pool = by_class[cls]
    return pool[int(rng.randint(len(pool)))], cls


def _component_mask(label, cls, rng):
    mask = (label == cls).astype(np.uint8)
    n, cc, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1:
        raise RuntimeError('selected image has no class %d instance' % cls)
    iid = int(rng.randint(1, n))
    return (cc == iid).astype(np.uint8), int(stats[iid, cv2.CC_STAT_AREA])


def geometry(image, label, rng, instance_class=None, size=2048, crop=1024):
    """Geometry matched to fusion_core.paired_geometry, without ref tensor."""
    from fusion_core import foreground_crop_bbox, IMG_MEAN, IMG_STD
    inst = area = None
    if instance_class is not None:
        inst, area = _component_mask(label, instance_class, rng)
    image = cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)
    label = cv2.resize(label.astype(np.uint8), (size, size),
                       interpolation=cv2.INTER_NEAREST)
    if inst is not None:
        inst = cv2.resize(inst, (size, size), interpolation=cv2.INTER_NEAREST)
    angle = float(rng.uniform(-45.0, 45.0))
    center = ((size - 1) / 2.0, (size - 1) / 2.0)
    mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    image = cv2.warpAffine(image, mat, (size, size), flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    label = cv2.warpAffine(label, mat, (size, size), flags=cv2.INTER_NEAREST,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    if inst is not None:
        inst = cv2.warpAffine(inst, mat, (size, size), flags=cv2.INTER_NEAREST,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        ys, xs = np.where(inst > 0)
        if not len(ys):
            raise RuntimeError('selected instance disappeared after geometry')
        pick = int(rng.randint(len(ys)))
        y, x = int(ys[pick]), int(xs[pick])
        y0 = int(np.clip(y - int(rng.randint(crop)), 0, size - crop))
        x0 = int(np.clip(x - int(rng.randint(crop)), 0, size - crop))
    else:
        y0, x0 = foreground_crop_bbox(label, crop, fg_prob=0.5,
                                       rng=rng, num_fg=4)
    img_c = image[y0:y0 + crop, x0:x0 + crop]
    lab_c = label[y0:y0 + crop, x0:x0 + crop]
    inst_positive = 0
    if inst is not None:
        inst_positive = int(inst[y0:y0 + crop, x0:x0 + crop].sum())
        if inst_positive < 1:
            raise RuntimeError('conditional crop lost selected instance')
    flip = bool(rng.rand() < 0.5)
    if flip:
        img_c = img_c[:, ::-1]
        lab_c = lab_c[:, ::-1]
    img_c = (img_c.astype(np.float32) - IMG_MEAN) / IMG_STD
    out_img = np.ascontiguousarray(img_c.transpose(2, 0, 1))
    out_lab = np.ascontiguousarray(lab_c.astype(np.int64))[None]
    trace = {
        'conditional': instance_class is not None,
        'selected_class': instance_class,
        'instance_area_original': area,
        'selected_instance_pixels_in_crop': inst_positive,
        'positive_pixels': int(((lab_c >= 1) & (lab_c <= 4)).sum()),
        'valid_pixels': int((lab_c != 255).sum()),
        'angle': angle, 'y0': y0, 'x0': x0, 'flip': flip,
    }
    return out_img, out_lab, trace


def student_predict(model, img, meta):
    p = model._student_prob(img, meta, True)
    if isinstance(p, (tuple, list)):
        p = p[0]
    return p[0] if p.dim() == 4 else p


def evaluate(model, settings, cfg, out_path, deadline):
    import runtime as rt
    model.eval()
    with torch.no_grad():
        return rt.evaluate_split(lambda x, m: student_predict(model, x, m),
                                 'val', settings['data_root'], cfg, deadline,
                                 lambda x: print(x, flush=True),
                                 dump_counts=out_path)


def train_arm(settings, arm, steps, seed, run_dir, timing=False):
    import runtime as rt
    import run_protocol as rp
    seed_all(seed)
    rt.inject_repo(settings['repo'])
    model, cfg = rp.build_student(settings)
    model.train(); rt.freeze_bn(model)
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                          lr=5e-4, momentum=0.9, weight_decay=5e-4)
    stems, by_class = build_index(settings['data_root'])
    rng = np.random.RandomState(seed)
    losses, grads, history = [], [], []
    stats = {'samples': 0, 'conditional': 0, 'classes': {str(c): 0 for c in range(1, 5)},
             'positive_windows': 0, 'positive_pixels': 0, 'valid_pixels': 0,
             'selected_survived': 0}
    snap, bn = rt.snapshot_state(model), rt.snapshot_buffers(model)
    start_time = time.time()
    val_points = [] if timing else sorted(set([max(1, steps // 2), steps]))
    for it in range(1, steps + 1):
        model.train(); rt.freeze_bn(model); opt.zero_grad()
        ims, labs, traces = [], [], []
        for _ in range(2):
            last_error = None
            for _attempt in range(8):
                stem, cls = choose_sample(arm, stems, by_class, rng)
                image, label = rt.load_image_label(settings['data_root'], 'train', stem)
                try:
                    im, lab, tr = geometry(image, label, rng, cls)
                    break
                except RuntimeError as exc:
                    # A selected tiny component can leave the canvas after rotation.
                    # Retry from a fresh sample, but fail loudly after a finite limit.
                    last_error = exc
            else:
                raise RuntimeError(f'failed to build a valid sample after 8 attempts: {last_error}')
            ims.append(torch.from_numpy(im)); labs.append(torch.from_numpy(lab)); traces.append(tr)
        ib = torch.stack(ims).cuda(); lb = torch.stack(labs).cuda()
        logits = model.encode_decode(ib, [{}])
        ref = torch.zeros((2, 4, 1024, 1024), device=logits.device)
        loss = rt.loss_pair(logits, lb, ref, 'A')
        if not torch.isfinite(loss):
            raise RuntimeError('nonfinite loss at step %d' % it)
        loss.backward()
        raw_norm = torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 5.0)
        if not torch.isfinite(raw_norm):
            raise RuntimeError('nonfinite grad at step %d' % it)
        opt.step(); rt.freeze_bn(model)
        losses.append(float(loss.detach().cpu())); grads.append(float(raw_norm))
        if it == 1:
            evidence = rt.first_update_evidence(model, snap, bn)
            if not all(evidence.values()):
                raise RuntimeError('first update failed %s' % evidence)
            atomic_json(os.path.join(run_dir, '%s_first_update.json' % arm), evidence)
            del snap, bn
        for tr in traces:
            stats['samples'] += 1
            stats['positive_windows'] += int(tr['positive_pixels'] > 0)
            stats['positive_pixels'] += tr['positive_pixels']; stats['valid_pixels'] += tr['valid_pixels']
            if tr['conditional']:
                stats['conditional'] += 1
                stats['classes'][str(tr['selected_class'])] += 1
                stats['selected_survived'] += int(tr['selected_instance_pixels_in_crop'] > 0)
        if it % 200 == 0:
            med = float(np.median(losses[-200:])); gmed = float(np.median(grads[-200:]))
            print('%s step %d loss_med %.6f grad_med %.4f' % (arm, it, med, gmed), flush=True)
            if len(losses) >= 400:
                prev = float(np.median(losses[-400:-200])); pg = float(np.median(grads[-400:-200]))
                if med > 1.3 * prev and (gmed > max(50.0, 5.0 * max(pg, 1e-8))):
                    raise RuntimeError('divergence gate at step %d' % it)
        if it in val_points:
            counts = os.path.join(run_dir, '%s_val_%d_counts.json' % (arm, it))
            m, pc, _ = evaluate(model, settings, cfg, counts, time.time() + 7200)
            history.append({'step': it, 'miou': m, 'per_class': pc})
            # Every pre-registered validation point must remain recoverable.  A
            # transient best score is otherwise unusable for the locked
            # downstream fusion evaluation.
            node_payload = {
                'state_dict': model.state_dict(),
                'optimizer': opt.state_dict(),
                'arm': arm,
                'seed': seed,
                'steps': it,
                'history': list(history),
            }
            node_tmp = os.path.join(run_dir, '%s_step_%d.pth.tmp' % (arm, it))
            node_out = os.path.join(run_dir, '%s_step_%d.pth' % (arm, it))
            torch.save(node_payload, node_tmp)
            os.replace(node_tmp, node_out)
            model.train(); rt.freeze_bn(model)
            print('%s VAL %d %.8f %s' % (arm, it, m, pc), flush=True)
    elapsed = time.time() - start_time
    stats['positive_window_rate'] = stats['positive_windows'] / max(stats['samples'], 1)
    stats['positive_pixel_rate'] = stats['positive_pixels'] / max(stats['valid_pixels'], 1)
    stats['elapsed_sec'] = elapsed; stats['sec_per_step'] = elapsed / max(steps, 1)
    payload = {'state_dict': model.state_dict(), 'optimizer': opt.state_dict(),
               'arm': arm, 'seed': seed, 'steps': steps, 'history': history,
               'stats': stats, 'loss_tail': losses[-400:]}
    os.makedirs(run_dir, exist_ok=True)
    tmp = os.path.join(run_dir, '%s_final.pth.tmp' % arm)
    torch.save(payload, tmp); os.replace(tmp, os.path.join(run_dir, '%s_final.pth' % arm))
    atomic_json(os.path.join(run_dir, '%s_summary.json' % arm),
                {'arm': arm, 'seed': seed, 'steps': steps,
                 'history': history, 'stats': stats,
                 'loss_first200_median': float(np.median(losses[:200])),
                 'loss_last200_median': float(np.median(losses[-200:]))})
    return stats, history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--settings', required=True)
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--mode', choices=('smoke', 'timing', 'train'), required=True)
    ap.add_argument('--arms', nargs='+', choices=('C', 'S'), default=['C', 'S'])
    ap.add_argument('--steps', type=int, default=200)
    ap.add_argument('--seed', type=int, default=42)
    a = ap.parse_args()
    s = load_settings(a.settings)
    os.makedirs(a.run_dir, exist_ok=True)
    steps = 1 if a.mode == 'smoke' else a.steps
    for arm in a.arms:
        train_arm(s, arm, steps, a.seed, a.run_dir, timing=a.mode != 'train')


if __name__ == '__main__':
    main()
