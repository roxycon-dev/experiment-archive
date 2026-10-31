#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""Baseline 42.20 -> champion 44.21 (verified historical) -> target 47.20.

Usage:
  python3 run_protocol.py --settings S.json --run-dir R
  python3 run_protocol.py --settings S.json --run-dir R --dry-run

Budget: the global deadline comes from the ledger (ops.deadline_of), NOT
from the hardcoded 24h.  preparation (cache + D-val + champion-val + timing)
is SHARED 3h (not 3h each).  pair(seed42) <= 10h, pair(seed43) <= 6h,
final <= 5h.  The final reserve is settings['final_reserve_hours']
(default 5h); with a short lease (e.g. 10h) set it ~2.5h.  If the second
pair does not fit the remaining budget, the protocol SKIPS it and runs the
final evaluation with the seed-42 best B (it never exits without a final
evaluation once the first gate passed).  Phase-start times are persisted in
protocol_state.json; a restart NEVER extends a phase.
"""
import argparse
import datetime
import hashlib
import json
import os
import random
import sys
import time

import numpy as np
import torch

import ops  # budget ledger / gate helpers (no mmcv)

PHASE_CAPS = {  # seconds; 'preparation' is the SHARED 3h prep block
    'preparation': 3 * 3600,
    'pair_seed42': 10 * 3600,
    'pair_seed43': 6 * 3600,
    'final': 5 * 3600,
}
FINAL_RESERVE_DEFAULT = 5 * 3600
GLOBAL_BUDGET = 24 * 3600
PHASE_ORDER = ['preparation', 'pair_seed42', 'pair_seed43', 'final']
# Restart re-measurement window: on resume the durable prep artifacts
# (reference cache, counts JSONs) already exist, so only D-val,
# champion-val and the pilot re-run.  A stale preparation phase deadline
# must not kill that re-measurement; the GLOBAL ledger deadline is never
# extended (phase_deadline itself stays pure/persisted).
PREP_RESUME_MIN = 45 * 60

IMG_EXT = ('.jpg', '.jpeg', '.png')


class ShutdownGraceful(Exception):
    pass


class Log(object):
    def __init__(self, run_dir):
        self.path = os.path.join(run_dir, 'protocol.log')

    def __call__(self, msg):
        line = '[%s] %s' % (time.strftime('%Y-%m-%d %H:%M:%S'), msg)
        print(line, flush=True)
        with open(self.path, 'a', encoding='utf-8') as f:
            f.write(line + '\n')


# ---------------------------------------------------------------------------
# settings / fingerprint
# ---------------------------------------------------------------------------
REQUIRED_KEYS = ['repo', 'config', 'data_root', 'student_checkpoint',
                 'reference_checkpoint', 'cache_dir', 'champion',
                 'expected_sha256']


def _abs(repo, rel):
    return rel if os.path.isabs(rel) else os.path.join(repo, rel)


def validate_settings(path):
    with open(path, 'r', encoding='utf-8') as f:
        s = json.load(f)
    missing = [k for k in REQUIRED_KEYS if k not in s]
    if missing:
        raise RuntimeError('settings missing keys: %s' % missing)
    targets = {
        'config': s['config'],
        'student_checkpoint': s['student_checkpoint'],
        'reference_checkpoint': s['reference_checkpoint'],
        'champion_head': s['champion']['head'],
    }
    for m in s['champion']['members']:
        targets['member_%s_cfg' % m['name']] = m['config']
        targets['member_%s_ckpt' % m['name']] = m['checkpoint']
    hashes, pending = {}, []
    for key, rel in sorted(targets.items()):
        full = _abs(s['repo'], rel)
        if not os.path.isfile(full):
            pending.append(full)
            continue
        exp = s['expected_sha256'].get(rel)
        if exp is None:
            raise RuntimeError('no expected_sha256 for %s; no unhashed '
                               'fallback (fail closed)' % rel)
        if exp != sha256(full):
            raise RuntimeError('sha256 mismatch for %s' % rel)
        hashes[key] = exp
    s['_hashes'] = hashes
    s['_abs'] = {k: _abs(s['repo'], v) for k, v in targets.items()}
    s['_pending'] = pending
    return s


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _resolve_obj(o, repo):
    """Recursively resolve an mmcv config to JSON-safe primitives."""
    if isinstance(o, dict):
        return {str(k): _resolve_obj(v, repo) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_resolve_obj(v, repo) for v in o]
    if isinstance(o, (str, int, float, bool)) or o is None:
        return o
    return repr(o)


def config_fingerprint(settings):
    """Fingerprint = recursive resolved cfg + ALL referenced source files."""
    import runtime as rt
    cfg = rt.load_config(settings['repo'], settings['config'])
    d = cfg.dump() if hasattr(cfg, 'dump') else json.dumps(
        _resolve_obj(cfg._cfg_dict if hasattr(cfg, '_cfg_dict') else {}, settings['repo']))
    blob = {'config_text_sha256': hashlib.sha256(d.encode()).hexdigest()}
    srcs = set()
    def walk(o):
        if isinstance(o, dict):
            for v in o.values():
                walk(v)
        elif isinstance(o, (list, tuple)):
            for v in o:
                walk(v)
        elif isinstance(o, str) and o.endswith('.py'):
            full = _abs(settings['repo'], o)
            if os.path.isfile(full):
                srcs.add(full)
    walk(_resolve_obj(cfg._cfg_dict if hasattr(cfg, '_cfg_dict') else {}, settings['repo']))
    blob['source_files'] = {}
    for directory in (settings['repo'], os.path.dirname(os.path.abspath(__file__))):
        for name in os.listdir(directory):
            if name.endswith('.py'):
                srcs.add(os.path.join(directory, name))
    for p in sorted(srcs):
        blob['source_files'][p] = sha256(p)
    return blob


# ---------------------------------------------------------------------------
# protocol state: persisted phase starts; restarts cannot extend phases
# ---------------------------------------------------------------------------
def load_protocol_state(run_dir, settings):
    """Returns (state, deadline_epoch).  The global run start is persisted on
    first invocation and reused on restart -> budgets never grow."""
    path = os.path.join(run_dir, 'protocol_state.json')
    if os.path.isfile(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    st = {'run_start': time.time(), 'chosen_T': None,
          'identity': {'hashes': settings['_hashes'],
                       'fingerprint': settings.get('_fingerprint')},
          'phase_start': {}, 'val_history': {}}
    return st


def save_protocol_state(run_dir, st):
    tmp = os.path.join(run_dir, 'protocol_state.json.tmp')
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(st, f, indent=2)
    os.replace(tmp, os.path.join(run_dir, 'protocol_state.json'))


def _identity_drift_kind(old, new):
    """None = identical; 'code' = fingerprint-only drift (amendable);
    'fatal' = asset/config hashes differ (silent protocol change)."""
    if old == new:
        return None
    if old.get('hashes') == new.get('hashes'):
        return 'code'
    return 'fatal'


def ensure_identity(run_dir, st, settings):
    """chosen_T and identity (hashes + fingerprint) recorded BEFORE training."""
    fp = config_fingerprint(settings)
    ident = {'hashes': settings['_hashes'], 'fingerprint': fp}
    old = st.get('identity')
    if old and old != ident:
        # Asset/config hashes MUST match.  The source fingerprint may drift
        # only via a recorded amendment (a deliberate restart fix); anything
        # else is a silent protocol change and stays fatal.
        if _identity_drift_kind(old, ident) != 'code':
            raise RuntimeError('identity changed since first run; refusing')
        record_amendment(run_dir, 'code_fingerprint_drift', {
            'old_fingerprint': old.get('fingerprint'),
            'new_fingerprint': ident['fingerprint']})
        st['identity'] = ident      # adopt; later restarts compare vs new
    if st.get('identity') is None:
        st['identity'] = ident
    return ident


def phase_deadline(st, phase):
    """Absolute deadline of a phase, from persisted starts; capped by the
    global deadline derived from the persisted run start.  A phase that was
    never started (budget-skipped, e.g. pair_seed43 on a short lease) is
    tolerated: its window is simply unused."""
    idx = PHASE_ORDER.index(phase)
    base = st['run_start']
    for p in PHASE_ORDER[:idx]:
        if p in st['phase_start']:
            base = max(base, st['phase_start'][p] + PHASE_CAPS[p])
    start = st['phase_start'].get(phase, base)
    return min(start + PHASE_CAPS[phase], st['run_start'] + GLOBAL_BUDGET)


def mark_phase(run_dir, st, phase):
    if phase not in st['phase_start']:
        st['phase_start'][phase] = time.time()
        save_protocol_state(run_dir, st)
    return phase_deadline(st, phase)


# ---------------------------------------------------------------------------
# model build: patch source_checkpoint (absolute) BEFORE construction
# ---------------------------------------------------------------------------
def build_student(settings):
    import runtime as rt
    cfg = rt.load_config(settings['repo'], settings['config'])
    # patch BEFORE constructing HRPlusRefinement
    cfg.model['source_checkpoint'] = settings['_abs']['reference_checkpoint']
    model = rt.build_model(cfg)            # constructor uses B (init once)
    rt.strict_initialize(model,
                         settings['_abs']['student_checkpoint'],
                         settings['_abs']['reference_checkpoint'])  # then D2000
    rt.configure_trainable(model)
    settings['_cfg'] = cfg
    return model.cuda(), cfg


# ---------------------------------------------------------------------------
# seeding: everything, before EACH arm construction
# ---------------------------------------------------------------------------
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# ---------------------------------------------------------------------------
# checkpoints: atomic + full payload
# ---------------------------------------------------------------------------
def save_checkpoint(path, model, optimizer, iter_idx, seed, T, arm,
                    settings, split_hashes, geom_rng, order_rng,
                    val_history):
    payload = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'iter': iter_idx, 'seed': seed, 'T': T, 'arm': arm,
        'identity': settings['_hashes'], 'split_hashes': split_hashes,
        'settings': {k: settings[k] for k in
                     ('config', 'data_root', 'student_checkpoint',
                      'reference_checkpoint')},
        'rng': get_rng_state(),
        'geom_rng': geom_rng.get_state(),
        'order_rng': order_rng.get_state(),
        'val_history': val_history,
    }
    tmp = path + '.tmp'
    torch.save(payload, tmp)
    os.replace(tmp, path)


def resume_checkpoint(path, model, optimizer, expect_seed, expect_T,
                      expect_arm, split_hashes):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    if (ck['seed'] != expect_seed or ck['T'] != expect_T or
            ck['arm'] != expect_arm):
        raise RuntimeError('resume validation failed: seed/T/arm mismatch')
    _check_split_hashes(path, ck.get('split_hashes'), split_hashes)
    model.load_state_dict(ck['state_dict'], strict=True)
    optimizer.load_state_dict(ck['optimizer'])
    set_rng_state(ck['rng'])
    return ck['iter'], ck.get('val_history', []), ck['geom_rng'], \
        ck['order_rng']


def record_amendment(run_dir, kind, detail):
    """Append-only, machine-checkable record of a deliberate restart-time
    protocol amendment (never a silent change)."""
    entry = {'time': time.strftime('%Y-%m-%dT%H:%M:%S'), 'kind': kind,
             'detail': detail}
    path = os.path.join(run_dir, 'protocol_amendments.jsonl')
    with open(path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(entry, sort_keys=True) + '\n')
    print('[AMENDMENT] %s %s' % (kind, json.dumps(detail, sort_keys=True)),
          flush=True)


def _check_split_hashes(path, saved, expect):
    """config/split/assets hashes must match EXACTLY.  The 'code' entry
    (whole-source fingerprint) may differ ONLY as a fingerprint-only drift
    recorded as a protocol amendment -- e.g. a restart-scheduling fix
    deployed as a deliberate code change.  Any other mismatch is fatal."""
    if saved == expect:
        return
    code_drift = (
        isinstance(saved, dict) and isinstance(expect, dict)
        and 'code' in saved and 'code' in expect
        and all(saved.get(k) == expect.get(k)
                for k in ('config', 'split', 'assets'))
        and saved.get('code') != expect.get('code'))
    if not code_drift:
        raise RuntimeError('resume validation failed: split/config mismatch')
    run_dir = os.path.dirname(os.path.dirname(os.path.abspath(path)))
    record_amendment(run_dir, 'code_fingerprint_drift_on_resume', {
        'checkpoint': os.path.abspath(path),
        'saved_code': saved.get('code'),
        'expected_code': expect.get('code')})


def arm_credit(run_dir, arm, seed):
    """(steps_done, evals_done) already persisted in an arm's last.pth.

    Sizes the REMAINING budget on restart: a completed arm re-entering
    train_arm loads last.pth and returns after the len(hist)==4 check
    without re-executing any step; a partial arm resumes at its saved
    iter.  Returns (0, 0) when nothing usable is found -- fail closed:
    the caller then demands the full-pair budget."""
    path = os.path.join(run_dir, 'ckpt_%s_s%d' % (arm, seed), 'last.pth')
    if not os.path.isfile(path):
        return 0, 0
    try:
        ck = torch.load(path, map_location='cpu', weights_only=False)
        if ck.get('arm') != arm or int(ck.get('seed', -1)) != int(seed):
            return 0, 0
        return max(0, int(ck.get('iter', 0))), \
            max(0, len(ck.get('val_history', []) or []))
    except Exception:
        return 0, 0


# ---------------------------------------------------------------------------
# RNG state
# ---------------------------------------------------------------------------
def get_rng_state():
    return {'py': random.getstate(), 'np': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all()}


def set_rng_state(st):
    random.setstate(st['py'])
    np.random.set_state(st['np'])
    torch.set_rng_state(st['torch'].cpu() if hasattr(st['torch'], 'cpu')
                        else st['torch'])
    torch.cuda.set_rng_state_all([s.cpu() if hasattr(s, 'cpu') else s
                                  for s in st['cuda']])


# ---------------------------------------------------------------------------
# data auditing (no preloading into RAM!)
# ---------------------------------------------------------------------------
def audit_labels(data_root, split, assignment, log):
    """Verify each label png: readable, shape == image, values in 0..4/255.
    Streams file-by-file; nothing is kept in RAM."""
    import cv2
    lab_dir = os.path.join(data_root, 'labels', split)
    img_dir = os.path.join(data_root, 'images', split)
    for fname in sorted(assignment):
        stem = os.path.splitext(fname)[0]
        lp = os.path.join(lab_dir, stem + '.png')
        img = cv2.imread(os.path.join(img_dir, fname), 1)
        if img is None:
            raise RuntimeError('unreadable image: %s' % fname)
        lab = cv2.imread(lp, 0)
        if lab is None:
            raise RuntimeError('unreadable label: %s' % lp)
        if lab.shape[:2] != img.shape[:2]:
            raise RuntimeError('label/image shape mismatch: %s' % fname)
        u = np.unique(lab)
        ok = np.all((u == 255) | ((u >= 0) & (u <= 4)))
        if not ok:
            raise RuntimeError('label values out of range 0..4/255: %s' % lp)
    log('labels OK for %s (%d files)' % (split, len(assignment)))


def list_split(data_root, split):
    d = os.path.join(data_root, 'images', split)
    fs = [f for f in sorted(os.listdir(d))
          if f.lower().endswith(IMG_EXT)]
    return fs


# ---------------------------------------------------------------------------
# training one arm
# ---------------------------------------------------------------------------
def train_arm(settings, run_dir, arm, seed, T, deadline, log, ref_cache,
              split_hashes, loss_fn=None, val_points=None):
    """Train one arm.  loss_fn(logits, lab_b, ref_b, it) defaults to
    rt.loss_pair(..., arm); val_points defaults to the four quarters
    [T//4, T//2, 3T//4, T] (exactly the original behavior).  Round-5 arms
    pass a weighted-fusion loss and their own val point list; everything
    else (seeding, resume, checkpoint payload, RNG restore, freeze,
    first-update evidence) is shared verbatim."""
    import runtime as rt
    if val_points is None:
        val_points = [T * q // 4 for q in (1, 2, 3, 4)]
    val_points = sorted(set(int(v) for v in val_points))
    if loss_fn is None:
        _loss = lambda lg, lb, rb, it: rt.loss_pair(lg, lb, rb, arm)
    else:
        _loss = loss_fn
    seed_everything(seed)          # BEFORE construction
    model, _ = build_student(settings)
    model.train()
    rt.freeze_bn(model)
    opt = torch.optim.SGD([p for n, p in model.named_parameters()
                           if p.requires_grad],
                          lr=5e-5, momentum=0.9, weight_decay=0.0005)
    ckpt_dir = os.path.join(run_dir, 'ckpt_%s_s%d' % (arm, seed))
    os.makedirs(ckpt_dir, exist_ok=True)
    last_path = os.path.join(ckpt_dir, 'last.pth')
    start, hist, gstate, ostate = 0, [], None, None
    if os.path.isfile(last_path):
        start, hist, gstate, ostate = resume_checkpoint(
            last_path, model, opt, seed, T, arm, split_hashes)
        log('%s s%d resumed at iter %d with %d val entries' %
            (arm, seed, start, len(hist)))
    geom_rng = np.random.RandomState(seed)
    order_rng = np.random.RandomState(seed)
    if gstate is not None:                       # restore LOCAL streams
        geom_rng.set_state(gstate)
        order_rng.set_state(ostate)
    data_root = settings['data_root']
    files = list_split(data_root, 'train')
    stems = [os.path.splitext(f)[0] for f in files]
    trace_path = os.path.join(run_dir, 'trace_%s_s%d.json' % (arm, seed))
    snap, bn_snap, checked = None, None, start > 0
    if not checked:
        snap, bn_snap = rt.snapshot_state(model), rt.snapshot_buffers(model)
    quarters = val_points
    for it in range(start, T):
        if time.time() > deadline:
            raise ShutdownGraceful('deadline during %s s%d' % (arm, seed))
        model.train()
        rt.freeze_bn(model)
        opt.zero_grad()
        idxs = order_rng.permutation(len(stems))[:2]   # batch of 2
        imgs, labs, refs = [], [], []
        for i in idxs:
            stem = stems[i]
            assert 0 <= i < len(stems) and stem in ref_cache
            img, lab = rt.load_image_label(data_root, 'train', stem)
            r = ref_cache[stem]
            im2, la2, rf2, tr = rt.paired_geometry(img, lab, r, geom_rng)
            imgs.append(torch.from_numpy(im2))
            labs.append(torch.from_numpy(la2))
            refs.append(torch.from_numpy(rf2))
            if not os.path.isfile(trace_path):         # first-batch proof
                rt.atomic_json(trace_path, {'arm': arm, 'seed': seed,
                                            'iter': it, 'batches': [tr]})
        img_b = torch.stack(imgs).cuda()
        lab_b = torch.stack(labs).cuda()
        ref_b = torch.stack(refs).cuda()
        logits = model.encode_decode(img_b, [{}])
        loss = _loss(logits, lab_b, ref_b, it)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 5.0)
        opt.step()
        rt.freeze_bn(model)
        if not checked:
            ev = rt.first_update_evidence(model, snap, bn_snap)
            if not all(ev.values()):
                raise RuntimeError('first update violated: %s' % ev)
            rt.atomic_json(os.path.join(run_dir,
                                        'first_update_%s_s%d.json' % (arm, seed)), ev)
            del snap, bn_snap                 # free after pass
            snap = bn_snap = None
            checked = True
        if (it + 1) % 100 == 0 and (it + 1) not in quarters:
            save_checkpoint(last_path, model, opt, it + 1, seed, T, arm,
                            settings, split_hashes, geom_rng, order_rng, hist)
        if (it + 1) in quarters:
            st = get_rng_state()
            was_training = model.training
            model.eval()                      # eval mode under no_grad
            with torch.no_grad():
                miou, _, _ = evaluate_split_local(
                    lambda i_, m_: d_inf(model, i_, m_), 'val', settings, log,
                    deadline)
            model.train(was_training)         # back to train, then freeze BN
            rt.freeze_bn(model)
            set_rng_state(st)                 # restore ALL rng
            hist.append((it + 1, miou))
            save_checkpoint(os.path.join(ckpt_dir, 'iter_%d.pth' % (it + 1)),
                            model, opt, it + 1, seed, T, arm, settings,
                            split_hashes, geom_rng, order_rng, hist)  # AFTER val
            save_checkpoint(last_path, model, opt, it + 1, seed, T, arm, settings,
                            split_hashes, geom_rng, order_rng, hist)
            log('%s s%d iter %d val mIoU %.4f' % (arm, seed, it + 1, miou))
    # resume integrity: every persisted val entry must sit on the CURRENT
    # val grid (a restart with a different grid would silently corrupt the
    # history semantics)
    for step, _m in hist:
        if int(step) not in quarters:
            raise RuntimeError('resumed val history step %d not on the '
                               'current val grid %s' % (step, quarters))
    if len(hist) != len(quarters):
        raise RuntimeError('expected %d validation timepoints, got %d'
                           % (len(quarters), len(hist)))
    best = rt.choose_checkpoint(hist)         # tie -> earlier step
    del model, opt
    torch.cuda.empty_cache()
    return best, hist


def d_inf(model, img, meta):
    """D-model inference; squeeze the batch dim (batch of 1 only)."""
    out = model.inference(img, meta, True)
    if isinstance(out, (tuple, list)):
        out = out[0]
    return out[0] if out.dim() == 4 else out     # (4,H,W)


def evaluate_split_local(predict_fn, split, settings, log, deadline,
                         dump_counts=None):
    import runtime as rt
    return rt.evaluate_split(predict_fn, split, settings['data_root'],
                             settings['_cfg'], deadline, log,
                             dump_counts=dump_counts)


# ---------------------------------------------------------------------------
# final: one image pass evaluates all THREE candidates
# ---------------------------------------------------------------------------
def evaluate_candidates(settings, champion, new_model, split, deadline, log,
                        dump_prefix=None):
    """Shared dataset loop: champion/new/mix probs computed once per image;
    true per-image and global counts kept for all three names; enforced
    expected counts; per-name count JSONs written under dump_prefix."""
    import runtime as rt
    import copy as _c
    import cv2
    from mmseg.datasets import build_dataset
    cfg = settings['_cfg']
    d = _c.deepcopy(cfg.data[split])
    d.test_mode = True
    d.data_root = settings['data_root']
    d.img_dir = os.path.join(settings['data_root'], 'images', split)
    d.ann_dir = os.path.join(settings['data_root'], 'labels', split)
    ds = build_dataset(d)
    expected = {'val': 76, 'test': 227}[split]
    if len(ds) != expected:
        raise RuntimeError('split %s has %d items, expected %d' %
                           (split, len(ds), expected))
    gt_maps = ds.get_gt_seg_maps()
    totals = {'champion': [[0, 0, 0] for _ in range(4)],
              'new': [[0, 0, 0] for _ in range(4)],
              'mix': [[0, 0, 0] for _ in range(4)]}
    per_image = {k: {} for k in totals}
    for i in range(len(ds)):
        if time.time() > deadline:
            raise ShutdownGraceful('deadline during %s candidate eval' % split)
        s = ds[i]
        img = rt._unwrap_img(s['img']).unsqueeze(0).cuda()
        meta = rt._unwrap_meta(s['img_metas'])
        with torch.no_grad():
            pc = champion.predict(img, meta)
            pn = d_inf(new_model, img, meta).detach()
            pm = rt.mix_probs(pc, pn)
        gt = np.asarray(gt_maps[i])
        ori = tuple(meta[0]['ori_shape'][:2])
        if gt.shape != ori:
            gt = cv2.resize(gt.astype(np.uint8), (ori[1], ori[0]),
                            interpolation=cv2.INTER_NEAREST)
        stem = os.path.splitext(os.path.basename(meta[0]['ori_filename']))[0]
        for name, p in (('champion', pc), ('new', pn), ('mix', pm)):
            cnt = rt.per_image_counts(
                p.detach().cpu().numpy().astype(np.float32),
                gt.astype(np.int64))
            per_image[name][stem] = {rt.CLASSES[c]: cnt[c]
                                     for c in range(4)}
            for c in range(4):
                for j in range(3):
                    totals[name][c][j] += cnt[c][j]
    out = {}
    for name in totals:
        miou, per = rt.global_miou([tuple(x) for x in totals[name]])
        out[name] = {'miou': miou, 'per_class': per,
                     'global_counts': totals[name],
                     'per_image': per_image[name]}
        log('final-candidates %s %s mIoU %.4f' % (split, name, miou))
        if dump_prefix:
            rt.atomic_json('%s_%s_counts.json' % (dump_prefix, name),
                           out[name])
    return out



# ---------------------------------------------------------------------------
# timing pilot
# ---------------------------------------------------------------------------
def pilot_timing(settings, run_dir, deadline, log, ref_cache,
                 reuse_eval_seconds=None):
    """50 updates per arm; proper cuda.synchronize; wall time includes data
    prep and checkpoint overhead; 10% margin applied by caller.  Does not
    mutate the run identity, no eval RNG side effects."""
    import runtime as rt
    max_step = 0.0
    for arm in ('A', 'B'):
        seed_everything(1000 + (0 if arm == 'A' else 1))
        model, _ = build_student(settings)
        model.train()
        rt.freeze_bn(model)
        opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                              lr=5e-5, momentum=0.9, weight_decay=0.0005)
        rng = np.random.RandomState(42)
        files = list_split(settings['data_root'], 'train')
        stems = [os.path.splitext(f)[0] for f in files]
        for it in range(50):
            if time.time() > deadline:
                raise ShutdownGraceful('deadline during pilot')
            t0 = time.time()
            idxs = rng.permutation(len(stems))[:2]
            imgs, labs, refs = [], [], []
            for i in idxs:
                stem = stems[i]
                img, lab = rt.load_image_label(settings['data_root'], 'train', stem)
                im2, la2, rf2, _ = rt.paired_geometry(img, lab, ref_cache[stem], rng)
                imgs.append(torch.from_numpy(im2))
                labs.append(torch.from_numpy(la2))
                refs.append(torch.from_numpy(rf2))
            opt.zero_grad()
            logits = model.encode_decode(torch.stack(imgs).cuda(), [{}])
            loss = rt.loss_pair(logits, torch.stack(labs).cuda(),
                                torch.stack(refs).cuda(), arm)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 5.0)
            opt.step()
            if it == 49:
                save_checkpoint(os.path.join(
                    run_dir, 'pilot_%s.pth' % arm), model, opt, 50, 0, 50,
                    arm, settings, {}, rng, rng, [])   # overhead included
            torch.cuda.synchronize()
            max_step = max(max_step, time.time() - t0)
        del model, opt
        torch.cuda.empty_cache()
    t_eval = reuse_eval_seconds
    if t_eval is None:
        model, _ = build_student(settings)
        model.eval()
        t0 = time.time()
        with torch.no_grad():
            evaluate_split_local(lambda i_, m_: d_inf(model, i_, m_),
                                 'val', settings, log, deadline)
        t_eval = time.time() - t0
        del model
        torch.cuda.empty_cache()
    return max_step, t_eval


# ---------------------------------------------------------------------------
# selection lock (immutable) and test markers
# ---------------------------------------------------------------------------
def write_lock(run_dir, lock):
    path = os.path.join(run_dir, 'selection_lock.json')
    if os.path.isfile(path):
        with open(path, 'r', encoding='utf-8') as f:
            old = json.load(f)
        if old != lock:
            raise RuntimeError('selection_lock.json exists and differs; '
                               'locks are immutable')
        return path
    rt_like_atomic(path, lock)
    return path


def rt_like_atomic(path, obj):
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, sort_keys=True)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--settings', required=True)
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    settings = validate_settings(args.settings)
    if settings['_pending'] and not args.dry_run:
        raise RuntimeError('Required assets missing: %s' % settings['_pending'])
    # Final reserve is configurable: with a short (e.g. 10h) lease the
    # default 5h starves the paired training; ~2.5h is enough for the
    # one-pass 3-candidate val + test evaluation.
    FINAL_RESERVE = float(settings.get('final_reserve_hours',
                                       FINAL_RESERVE_DEFAULT / 3600.0)) * 3600.0
    os.makedirs(args.run_dir, exist_ok=True)
    log = Log(args.run_dir)

    # ---- dry-run: no mmcv / GPU / ledger needed; no fake pass ----
    if args.dry_run:
        pending = list(settings.get('_pending', []))
        for split in ('train', 'val', 'test'):
            d = os.path.join(settings['data_root'], 'images', split)
            if os.path.isdir(d):
                n = len(list_split(settings['data_root'], split))
                log('dry-run: %s images = %d' % (split, n))
            else:
                pending.append(d)
        for ph in ('train', 'val'):
            d = os.path.join(settings['data_root'], 'labels', ph)
            if not os.path.isdir(d):
                pending.append(d)
        if pending:
            print('dry-run PENDING missing files/dirs:', flush=True)
            for p in pending:
                print('  MISSING %s' % p, flush=True)
            raise RuntimeError('dry-run failed: %d pending items' % len(pending))
        rt = None  # settings + files + splits validated; no torch build
        rt_like_atomic(os.path.join(args.run_dir, 'dry_run.json'),
                       {'ok': True, 'splits': {
                           s: len(list_split(settings['data_root'], s))
                           for s in ('train', 'val', 'test')}})
        log('dry-run OK: settings/files/splits validated (no mmcv/GPU)')
        return

    import runtime as rt
    rt.inject_repo(settings['repo'])
    settings['_cfg'] = rt.load_config(settings['repo'], settings['config'])
    settings['_cfg'].model['source_checkpoint'] = settings['_abs']['reference_checkpoint']
    settings['_fingerprint'] = config_fingerprint(settings)

    # ---- ledger: budget_ledger.json via ops API ----
    ledger = ops.load_ledger(args.run_dir)      # from budget_ledger.json
    deadline_dt = ops.deadline_of(ledger)       # aware datetime
    global_deadline = deadline_dt.timestamp()
    log('global deadline: %s (%.2fh remaining)' %
        (deadline_dt.isoformat(), (global_deadline - time.time()) / 3600.0))

    resuming = os.path.isfile(os.path.join(args.run_dir,
                                           'protocol_state.json'))
    st = load_protocol_state(args.run_dir, settings)
    ident = ensure_identity(args.run_dir, st, settings)
    save_protocol_state(args.run_dir, st)

    data_root = settings['data_root']
    assignment = rt.audit_split(data_root)
    split_hashes = {'config': settings['_hashes']['config'],
                    'split': hashlib.sha256(json.dumps(assignment, sort_keys=True).encode()).hexdigest(),
                    'assets': settings['_hashes'], 'code': settings['_fingerprint']}
    log('split audit OK')
    audit_labels(data_root, 'train', [f for f, v in assignment.items()
                                      if v['split'] == 'train'], log)

    # reserve check: 5h for final must always fit inside global deadline
    if global_deadline - time.time() < FINAL_RESERVE:
        raise ShutdownGraceful('cannot reserve final 5h')

    # =================== SHARED PREPARATION (3h total) ===================
    d_prep = mark_phase(args.run_dir, st, 'preparation')
    d_prep = min(d_prep, global_deadline)
    if resuming and d_prep < time.time() + PREP_RESUME_MIN:
        # Restart after the preparation window elapsed: only the
        # re-measurement (D-val, champion-val, pilot) re-runs, so re-floor
        # the LOCAL working window; the global ledger deadline -- the
        # money-critical invariant -- is never extended.
        d_prep = min(global_deadline, time.time() + PREP_RESUME_MIN)
        log('resume: prep window re-floored to now+%ds (global ledger '
            'deadline unchanged)' % PREP_RESUME_MIN)

    # reference cache (streamed to disk, never held in RAM wholesale)
    cache = rt.ReferenceCache(settings['cache_dir'], settings)
    stems = [os.path.splitext(f)[0]
             for f in list_split(data_root, 'train')]
    if not cache.complete():
        model, _ = build_student(settings)
        cache.build(model, stems, d_prep, log)
        del model
        torch.cuda.empty_cache()

    # starting D val + champion val (durations reused for pilot eval timing)
    model_d, cfg = build_student(settings)
    model_d.eval()
    t0 = time.time()
    with torch.no_grad():
        d_val, _, _ = evaluate_split_local(
            lambda i_, m_: d_inf(model_d, i_, m_), 'val', settings, log,
            d_prep, dump_counts=os.path.join(args.run_dir,
                                             'baseline_D_val_counts.json'))
    t_eval_real = time.time() - t0            # reused; no extra eval
    log('starting D val mIoU %.4f (historical baseline 42.20)' % d_val)
    del model_d
    torch.cuda.empty_cache()

    champion = rt.ChampionEnsemble(settings, settings['champion'])
    t0 = time.time()
    with torch.no_grad():
        champ_val, _, _ = evaluate_split_local(
            champion.predict, 'val', settings, log, d_prep,
            dump_counts=os.path.join(args.run_dir, 'champion_val_counts.json'))
    t_champ = time.time() - t0        # champion cost, tracked separately
    log('champion val mIoU %.4f (historical reference: h5 champion TEST '
        '44.21; val is computed fresh here); %.1fs/76 imgs'
        % (champ_val, t_champ))

    # t_eval_real stays the D-model (fused student+reference) eval time --
    # that is the cost of the 8 in-training validation passes.  The champion
    # cost only matters for the final phase; warn if the reserve cannot
    # cover the projected one-pass 3-candidate val + test.
    t_champ_est = t_champ * (76 + 227) / 76.0 + 900.0
    if t_champ_est > FINAL_RESERVE:
        log('WARNING: projected final eval %.0fs exceeds reserve %.0fs; '
            'the ledger deadline still hard-stops the run' %
            (t_champ_est, FINAL_RESERVE))

    t_step, _ = pilot_timing(settings, args.run_dir, d_prep, log, cache,
                             reuse_eval_seconds=t_eval_real)
    avail_pair = global_deadline - FINAL_RESERVE - time.time()
    # Resume credit: steps/evals already persisted in the pair-42
    # checkpoints.  train_arm re-enters a completed arm via last.pth and
    # returns without re-executing credited work, and a partial arm
    # resumes at its saved iter -- so the REMAINING work, not the
    # full-pair cost, is what must fit the budget for the seed-42 pair
    # to continue after a restart.
    done_a = arm_credit(args.run_dir, 'A', 42)
    done_b = arm_credit(args.run_dir, 'B', 42)
    done_steps = done_a[0] + done_b[0]
    done_evals = done_a[1] + done_b[1]
    if st['chosen_T'] is not None:
        T = st['chosen_T']                     # locked on the first run
    else:
        try:
            proposed_T = rt.choose_steps(t_step * 1.1, t_eval_real * 1.1,
                                         600.0, avail_pair)
        except ValueError as e:
            raise ShutdownGraceful('step planner: %s' % e)
        T = proposed_T
    st['chosen_T'] = T
    save_protocol_state(args.run_dir, st)
    # full-pair cost: the yardstick for the OPTIONAL seed-43 pair below
    need = 2 * T * t_step * 1.1 + 8 * t_eval_real * 1.1
    need_rest = (max(0, 2 * T - done_steps) * t_step * 1.1 +
                 max(0, 8 - done_evals) * t_eval_real * 1.1)
    if global_deadline - time.time() < need_rest + FINAL_RESERVE:
        raise ShutdownGraceful('insufficient time for pairs + final reserve')
    if done_steps or done_evals:
        log('resume credit: %d/%d steps, %d/8 evals already done -> '
            'remaining pair cost %.0fs (a fresh pair would need %.0fs)' %
            (done_steps, 2 * T, done_evals, need_rest, need))
    log('pilot %.3fs/step, %.1fs/eval -> T=%d (pair budget %.0fs)' %
        (t_step, t_eval_real, T, avail_pair))

    # release champion/model objects before training; rebuild for final only
    champion = None
    torch.cuda.empty_cache()

    # =================== pair seed42 (<=10h) ===================
    d42 = min(mark_phase(args.run_dir, st, 'pair_seed42'), global_deadline - FINAL_RESERVE)
    a_best, a_hist = train_arm(settings, args.run_dir, 'A', 42, T, d42, log,
                               cache, split_hashes)
    b_best, b_hist = train_arm(settings, args.run_dir, 'B', 42, T, d42, log,
                               cache, split_hashes)
    # gate uses the four VALIDATION-TIMEPOINT mIoUs, NOT per-class IoU
    a4 = [m for _, m in a_hist]
    b4 = [m for _, m in b_hist]
    go = ops.gate_first(a4, b4, d_val)
    log('seed42 gate: A=%s B=%s go=%s' %
        (['%.4f' % x for x in a4], ['%.4f' % x for x in b4], go))
    if not go:
        log('gate failed; no test run; retain champion; no fabricated data')
        rt_like_atomic(os.path.join(args.run_dir, 'summary.json'), {
            'baseline_D_val': d_val, 'champion_val': champ_val,
            'gate_passed': False, 'target': 47.20, 'status': 'stopped_no_gate'})
        return

    # =================== pair seed43 (only if budget fits) ================
    b43_best, b43_hist = None, None
    if global_deadline - time.time() >= need + FINAL_RESERVE:
        d43 = min(mark_phase(args.run_dir, st, 'pair_seed43'),
                  global_deadline - FINAL_RESERVE)
        a43_hist = train_arm(settings, args.run_dir, 'A', 43, T, d43, log,
                             cache, split_hashes)[1]
        b43_best, b43_hist = train_arm(settings, args.run_dir, 'B', 43, T,
                                       d43, log, cache, split_hashes)
        if max(m for _, m in b43_hist) <= max(m for _, m in a43_hist):
            log('seed43 reversed (B<=A); reject adoption; retain champion')
            rt_like_atomic(os.path.join(args.run_dir, 'summary.json'), {
                'baseline_D_val': d_val, 'champion_val': champ_val,
                'gate_passed': True, 'seed43_reversed': True,
                'target': 47.20, 'status': 'seed43_rejected'})
            return
    else:
        # Short lease (e.g. 10h): the second pair does not fit.  NEVER exit
        # without a final evaluation once the first gate passed -- run the
        # final with the seed-42 best B instead of aborting.
        log('insufficient budget for second pair; final proceeds with '
            'seed42 best B')

    # choose best B checkpoint across available seeds; tie -> seed 42
    cands = [('B', 42, b_best, b_hist)]
    if b43_hist is not None:
        cands.append(('B', 43, b43_best, b43_hist))
    best_seed, best = _choose_best_seed(cands)
    ckpt_path = os.path.join(args.run_dir, 'ckpt_B_s%d' % best_seed,
                             'iter_%d.pth' % best[0])
    log('best new: seed %d step %d mIoU %.4f file %s' %
        (best_seed, best[0], best[1], ckpt_path))

    # rebuild champion for the final candidate evaluation
    champion = rt.ChampionEnsemble(settings, settings['champion'])
    new_model, _ = build_student(settings)
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    new_model.load_state_dict(ck['state_dict'], strict=True)
    new_model.eval()                                    # final: train False

    # final candidates on val via ONE image pass
    d_cand = min(mark_phase(args.run_dir, st, 'final'), global_deadline)
    with torch.no_grad():
        val_res = evaluate_candidates(settings, champion, new_model, 'val',
                                      d_cand, log)
    champ_v = val_res['champion']['miou']
    new_v = val_res['new']['miou']
    mix_v = val_res['mix']['miou']
    # ties -> champion
    if max(new_v, mix_v) <= champ_v + 1e-12:
        locked = 'champion'
    elif new_v >= mix_v:
        locked = 'new'
    else:
        locked = 'mix'
    log('locked candidate: %s (champ %.4f new %.4f mix %.4f)' %
        (locked, champ_v, new_v, mix_v))

    lock = {
        'candidate': locked,
        'chosen_checkpoint': {'path': ckpt_path,
                              'sha256': sha256(ckpt_path),
                              'step': best[0], 'miou': best[1],
                              'seed': best_seed},
        'hashes': ident['hashes'],
        'fingerprint': ident['fingerprint'],
        'split_hashes': split_hashes,
        'val_results': {'champion': champ_v, 'new': new_v, 'mix': mix_v},
        'created': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }
    write_lock(args.run_dir, lock)          # BEFORE test; immutable

    # =================== final test (may start early) ===================
    final_test(settings, args.run_dir, lock, champion, new_model,
               min(global_deadline, st['phase_start'].get('final',
                   global_deadline - FINAL_RESERVE) + PHASE_CAPS['final']),
               log)
    rt_like_atomic(os.path.join(args.run_dir, 'summary.json'), {
        'baseline_D_val': d_val, 'champion_val': champ_val,
        'gate_passed': True, 'locked': locked,
        'val': {'champion': champ_v, 'new': new_v, 'mix': mix_v},
        'target': 47.20, 'status': 'tested'})


def _choose_best_seed(cands):
    """cands: list of (arm, seed, best(step,miou), hist).  Best by miou;
    ties -> seed 42."""
    best = None
    for arm, seed, bestck, hist in cands:
        if best is None or bestck[1] > best[1][1] + 1e-12:
            best = (seed, bestck)
        elif abs(bestck[1] - best[1][1]) <= 1e-12 and seed < best[0]:
            best = (seed, bestck)
    return best


def final_test(settings, run_dir, lock, champion, new_model, deadline, log):
    tag = os.path.join(run_dir, 'TEST_STARTED')
    done = os.path.join(run_dir, 'TEST_DONE')
    if os.path.isfile(done):
        log('test already complete; skip')
        return
    if os.path.isfile(tag):
        raise RuntimeError('TEST_STARTED exists without TEST_DONE: previous '
                           'test incomplete; fail closed, no automatic rerun')
    with open(tag, 'w') as f:
        f.write(time.strftime('%Y-%m-%dT%H:%M:%S'))
    try:
        with torch.no_grad():
            res = evaluate_candidates(settings, champion, new_model, 'test',
                                      deadline, log,
                                      dump_prefix=os.path.join(run_dir, 'test'))
    except ShutdownGraceful:
        raise
    except Exception:
        os.remove(tag)
        raise
    # winner stays the locked one even if another candidate scores higher
    res['locked_winner'] = lock['candidate']
    res['note'] = ('test results reported for all candidates; selection was '
                   'locked on val before the test; no claim of success')
    rt_like_atomic(os.path.join(run_dir, 'test_report.json'), res)
    log('test results: %s (locked winner: %s)' %
        ({k: round(v['miou'], 4) for k, v in res.items()
          if isinstance(v, dict) and 'miou' in v}, lock['candidate']))
    with open(done, 'w') as f:
        f.write('ok')
    os.remove(tag)


if __name__ == '__main__':
    try:
        main()
    except ShutdownGraceful as e:
        print('[shutdown-guard] %s' % e, flush=True)
        sys.exit(0)
