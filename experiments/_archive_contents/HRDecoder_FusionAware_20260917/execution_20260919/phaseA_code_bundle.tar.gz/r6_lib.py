# -*- coding: utf-8 -*-
"""Round-6 shared library: state/deviations, disk guards, counts math,
paired bootstrap, patch grid sampling.  Pure stdlib + numpy (+torch only in
the fusion helper); NO mmcv import here so the unit tests are cheap."""
import hashlib
import json
import os
import time

import numpy as np

CLASSES = ['EX', 'HE', 'SE', 'MA']
SEED = 20260918
BOOTSTRAP_N = 2000


class PhaseTimeout(Exception):
    """Raised when a PHASE budget is exhausted; the orchestrator catches it,
    records a deviation and moves on (distinct from the global deadline,
    which stops the whole round via run_protocol.ShutdownGraceful)."""


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, obj):
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, sort_keys=True)
    os.replace(tmp, path)


def make_log(run_dir):
    path = os.path.join(run_dir, 'r6.log')

    def log(msg):
        line = '[%s] %s' % (time.strftime('%Y-%m-%d %H:%M:%S'), msg)
        print(line, flush=True)
        with open(path, 'a', encoding='utf-8') as f:
            f.write(line + '\n')
    return log


class State(object):
    """r6_state.json: idempotent phases + recorded results."""

    def __init__(self, run_dir):
        self.path = os.path.join(run_dir, 'r6_state.json')
        self.d = {}
        if os.path.isfile(self.path):
            with open(self.path, 'r', encoding='utf-8') as f:
                self.d = json.load(f)

    def save(self):
        atomic_json(self.path, self.d)

    def set(self, key, value):
        self.d[key] = value
        self.save()


def deviation(run_dir, kind, detail):
    path = os.path.join(run_dir, 'DEVIATIONS.jsonl')
    entry = {'time': time.strftime('%Y-%m-%dT%H:%M:%S'), 'kind': kind,
             'detail': detail}
    with open(path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(entry, sort_keys=True) + '\n')
    print('[DEVIATION] %s %s' % (kind, json.dumps(detail, sort_keys=True)),
          flush=True)


def free_gb(path):
    st = os.statvfs(path) if hasattr(os, 'statvfs') else None
    if st is not None:
        return st.f_bavail * st.f_frsize / (1024.0 ** 3)
    import shutil
    return shutil.disk_usage(path).free / (1024.0 ** 3)


# disk cleanup candidates, in deletion order (all provably redundant:
# round-5 finished with summary written; the BNEW member checkpoints are
# the ONLY round-1/r5 artifacts round 6 still needs)
# Removal order (all provably redundant for round 6): the round-1 refcache
# is unused (round 6 builds its own features), the round-5 run dir has been
# archived locally (summary/logs pulled 13:16) and the guard's emergency tar
# duplicates the live run dir.  Glob patterns are supported.
CLEANUP_CANDIDATES = [
    '/root/autodl-tmp/hrplus/cache/ref_train_v3',       # ~15G r1 refcache
    '/root/autodl-tmp/hrplus/glm/backups/*.tar',        # ~6.5G r5 tar
    '/root/autodl-tmp/hrplus/glm/runs/r1',              # ~1.4G r1 leftovers
]


def ensure_space(path, need_gb, run_dir, log, allow_cleanup=True):
    """Returns free GB.  If free < need_gb and allow_cleanup, removes the
    redundant artifacts above (each removal is logged + recorded)."""
    free = free_gb(path)
    log('disk: %.1fG free (need %.1fG) at %s' % (free, need_gb, path))
    if free >= need_gb or not allow_cleanup:
        return free
    import glob as _glob
    import shutil
    expanded = []
    for cand in CLEANUP_CANDIDATES:
        expanded.extend(sorted(_glob.glob(cand)) if _glob.has_magic(cand)
                        else [cand])
    for cand in expanded:
        if free >= need_gb:
            break
        if not os.path.exists(cand):
            continue
        if os.path.isdir(cand):
            size = sum(os.path.getsize(os.path.join(dp, f))
                       for dp, _, fs in os.walk(cand) for f in fs
                       if os.path.exists(os.path.join(dp, f)))
            shutil.rmtree(cand, ignore_errors=True)
        else:
            size = os.path.getsize(cand)
            os.remove(cand)
        deviation(run_dir, 'disk_cleanup',
                  {'removed': cand, 'bytes': int(size),
                   'reason': 'free below %.1fG' % need_gb})
        log('removed %s (%.2fG) to make room' % (cand, size / 1024.0 ** 3))
        free = free_gb(path)
    log('disk after cleanup: %.1fG free' % free)
    if free < need_gb:
        raise RuntimeError('insufficient disk: %.1fG free < %.1fG needed'
                           % (free, need_gb))
    return free


# ---------------------------------------------------------------- counts ---
def iou(tp, fp, fn):
    union = tp + fp + fn
    return 0.0 if union == 0 else float(tp) / float(union)


def counts_iou(counts):
    """counts: {'EX': [tp,fp,fn], ...} -> {'EX': iou, ..., 'mIoU': m}."""
    out = {}
    for c in CLASSES:
        tp, fp, fn = counts[c][:3]
        out[c] = iou(tp, fp, fn)
    out['mIoU'] = sum(out[c] for c in CLASSES) / 4.0
    return out


def load_counts(path):
    """Loads an evaluate_split counts JSON -> (per_image, global_iou)."""
    with open(path, 'r', encoding='utf-8') as f:
        d = json.load(f)
    per_image = d['per_image']
    tot = {c: [0, 0, 0] for c in CLASSES}
    for _, cnt in per_image.items():
        for c in CLASSES:
            for j in range(3):
                tot[c][j] += int(cnt[c][j])
    return per_image, counts_iou(tot)


def paired_bootstrap(per_image_a, per_image_b, n=BOOTSTRAP_N, seed=SEED):
    """Image-level paired bootstrap over GLOBAL IoU (accumulate counts across
    resampled images, then compute IoU -- never average per-image IoUs).
    Returns {'delta': mIoU_b - mIoU_a, 'ci': [lo, hi], 'crosses_zero': bool,
    'n': n, 'seed': seed}.  Both dicts must share the exact same key set."""
    if set(per_image_a) != set(per_image_b):
        raise ValueError('paired bootstrap requires identical image sets')
    stems = sorted(per_image_a)
    a = np.array([[per_image_a[s][c][j] for c in CLASSES for j in range(3)]
                  for s in stems], dtype=np.float64)      # (N, 12)
    b = np.array([[per_image_b[s][c][j] for c in CLASSES for j in range(3)]
                  for s in stems], dtype=np.float64)
    n_img = len(stems)
    rng = np.random.RandomState(seed)
    deltas = np.empty(n, dtype=np.float64)
    for k in range(n):
        idx = rng.randint(0, n_img, n_img)
        sa = a[idx].sum(axis=0)
        sb = b[idx].sum(axis=0)
        ma = np.mean([iou(sa[3 * c], sa[3 * c + 1], sa[3 * c + 2])
                      for c in range(4)])
        mb = np.mean([iou(sb[3 * c], sb[3 * c + 1], sb[3 * c + 2])
                      for c in range(4)])
        deltas[k] = mb - ma
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    tot_a = a.sum(axis=0)
    tot_b = b.sum(axis=0)
    point_a = np.mean([iou(tot_a[3 * c], tot_a[3 * c + 1], tot_a[3 * c + 2])
                       for c in range(4)])
    point_b = np.mean([iou(tot_b[3 * c], tot_b[3 * c + 1], tot_b[3 * c + 2])
                       for c in range(4)])
    return {'delta': float(point_b - point_a),
            'iou_a': float(point_a), 'iou_b': float(point_b),
            'ci': [float(lo), float(hi)],
            'crosses_zero': bool(lo <= 0.0 <= hi),
            'n': int(n), 'seed': int(seed)}


def class_delta(per_image_new, per_image_base, cls):
    """Global IoU delta for ONE class over the same image set."""
    def acc(pi):
        tp = fp = fn = 0
        for s in pi:
            v = pi[s][cls]
            tp += v[0]
            fp += v[1]
            fn += v[2]
        return iou(tp, fp, fn)
    return acc(per_image_new) - acc(per_image_base)


def load_label(data_root, split, stem, validate=True):
    """Label map for ANY split.  runtime.load_image_label is train-only, so
    val/test evaluation must go through here.  Semantics mirrored exactly:
    grayscale uint8, values in {0..4, 255} (255 = ignore)."""
    import cv2
    p = os.path.join(data_root, 'labels', split, stem + '.png')
    lab = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    if lab is None:
        raise ValueError('missing label %s' % p)
    if validate and not np.isin(lab, [0, 1, 2, 3, 4, 255]).all():
        raise ValueError('invalid label values in %s' % p)
    return lab


# ---------------------------------------------------------------- patches --
def window_origins(h, w, win, stride):
    """Sliding-window origins covering [0,h)x[0,w) with edge clamping (the
    last window in each axis is shifted back to fit)."""
    if win <= 0 or stride <= 0:
        raise ValueError('win/stride must be positive')
    ys = list(range(0, max(1, h - win + 1), stride))
    xs = list(range(0, max(1, w - win + 1), stride))
    if ys[-1] != h - win:
        ys.append(max(0, h - win))
    if xs[-1] != w - win:
        xs.append(max(0, w - win))
    return [(y, x) for y in sorted(set(ys)) for x in sorted(set(xs))]


def label_patch_plan(gt, label_value, win, n_pos, n_neg, rng,
                     min_pos_px=8):
    """Sample (y, x) top-left origins: positives centred on pixels of
    `label_value` with >= min_pos_px such pixels in the window; negatives
    with zero such pixels.  Returns (pos_list, neg_list) of (y, x)."""
    gt = np.asarray(gt)
    h, w = gt.shape[:2]
    win = int(min(win, h, w))
    pos_mask = (gt == label_value)
    ys, xs = np.nonzero(pos_mask)
    pos = []
    if len(ys) > 0:
        for _ in range(int(n_pos)):
            i = rng.randint(0, len(ys))
            y = int(np.clip(ys[i] - win // 2, 0, h - win))
            x = int(np.clip(xs[i] - win // 2, 0, w - win))
            if int(pos_mask[y:y + win, x:x + win].sum()) >= min_pos_px:
                pos.append((y, x))
    neg = []
    tries = 0
    while len(neg) < int(n_neg) and tries < 40 * int(n_neg) + 40:
        tries += 1
        y = rng.randint(0, h - win + 1) if h > win else 0
        x = rng.randint(0, w - win + 1) if w > win else 0
        if not pos_mask[y:y + win, x:x + win].any():
            neg.append((y, x))
    return pos, neg


def ma_patch_plan(gt, win, n_pos, n_neg, rng, min_pos_px=8):
    """MA (label 4) convenience wrapper."""
    return label_patch_plan(gt, 4, win, n_pos, n_neg, rng, min_pos_px)
