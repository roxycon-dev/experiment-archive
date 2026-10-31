# -*- coding: utf-8 -*-
"""Fixed-protocol candidate evaluation against the frozen champion.

Protocol (FROZEN -- deliberately not parameterizable):
  * Champion probability comes from the EXISTING runtime.ChampionEnsemble
    pipeline driven by settings['champion']: members D,E,B,E4 with weights
    3,2,1,1 (member E checkpoint = E_1500), horizontal-flip TTA and the
    single MA correction head loaded from h4_best.  Nothing of that
    pipeline is rebuilt or altered here.
  * Candidates are C/S node checkpoints saved by phase_cs.py, i.e. payloads
    {'state_dict': ..., 'arm': 'C'|'S', 'steps': ..., ...}.  The payload is
    read through the project-canonical trusted loader
    (fusion_core._torch_load_trusted; trusted LOCAL files produced by
    phase_cs.py -- no untrusted deserialization happens in this script),
    then loading is strict (load_state_dict(strict=True)) and verified
    BITWISE against the payload tensors; payload arm must be 'C' or 'S'.
  * Candidate probability is the INDEPENDENT student only, obtained by
    calling model._student_prob(img, meta, True).  model.inference() /
    model(return_loss=False) are NEVER used for candidates because they
    blend 0.5*reference + 0.5*student.
  * The only evaluated fusion is mix = 0.75 * champion + 0.25 * candidate
    (FIXED constants below; there is no weight-scan option to find).
  * Metric (identical to runtime.evaluate_split): 4 sigmoid channels,
    strict > 0.5 threshold, per-image TP/FP/FN accumulated over the WHOLE
    split, then IoU = TP/(TP+FP+FN) per class and the arithmetic mean of
    the 4 class IoUs.  union == 0 fails closed (no class is skipped).
  * Split: 'val' by default.  'test' requires BOTH --allow-test AND
    --locked-candidate (the candidate locked on val before any test run);
    the lock is cross-checked against locked_candidate.json /
    --selection-lock when such a file is available.
  * Champion probabilities may be cached on disk (--champion-cache-dir):
    hashed identity (champion spec + all weight/config hashes + eval
    config text), per-image float32 npy, sha256-verified on every access;
    corrupt/incomplete caches fail closed.  Every directory path is
    normalized and rejected if it contains any '..' component; every
    cached file path is rebuilt from the validated directory plus a
    regex-validated stem (no path traversal). Champion probabilities remain
    float32 so caching cannot change decisions near the strict 0.5 threshold.
  * Every JSON output is written atomically via runtime.atomic_json
    (tmp file + os.replace).
  * SHA256 digests of all weights, configs and the split manifest are
    recorded in the summary JSON.
  * ANY exception exits NON-ZERO.  This script never shuts the machine
    down, never reboots, and never terminates with exit 0 on failure.
"""
import argparse
import copy
import hashlib
import json
import os
import re
import sys
import time

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.dirname(_HERE), _HERE):
    if _p and _p not in sys.path:
        sys.path.insert(0, _p)

import runtime as rt          # noqa: E402  (sibling at deploy time, parent here)
import run_protocol as rp     # noqa: E402
import fusion_core            # noqa: E402  (canonical trusted-checkpoint loader)

MIX_CHAMPION = 0.75           # frozen evaluation weights; NO scan possible
MIX_CANDIDATE = 0.25
assert abs(MIX_CHAMPION + MIX_CANDIDATE - 1.0) < 1e-12
EXPECTED_COUNTS = {'val': 76, 'test': 227}
NAME_RE = re.compile(r'^[A-Za-z0-9_.-]+$')


def log(msg):
    print('[%s] %s' % (time.strftime('%Y-%m-%d %H:%M:%S'), msg), flush=True)


def safe_dir(path, what):
    """Normalize a CLI-supplied directory; reject any '..' component."""
    d = os.path.abspath(path)
    if '..' in os.path.normpath(d).split(os.sep):
        raise RuntimeError('%s must not contain "..": %r' % (what, path))
    return d


def load_settings(path):
    """run_protocol.validate_settings + fail closed on missing assets."""
    s = rp.validate_settings(path)
    if s.get('_pending'):
        raise RuntimeError('required assets missing: %s' % s['_pending'])
    return s


def check_stem(stem, where):
    if not NAME_RE.match(stem or ''):
        raise RuntimeError('%s: unsafe stem %r' % (where, stem))
    return stem


def load_node_checkpoint(model, path, name, expect_arm=None):
    """Strictly load a phase_cs C/S node payload into ``model``.

    payload must be a dict with a tensor 'state_dict', 'arm' in (C, S) and
    a positive int 'steps'.  load_state_dict(strict=True) rejects ANY
    missing/unexpected/shape-mismatched key; afterwards every loaded tensor
    is compared bitwise with the payload.  Returns a small report dict.
    """
    payload = fusion_core._torch_load_trusted(path)
    if not isinstance(payload, dict):
        raise RuntimeError('candidate %s: checkpoint payload is not a dict' % name)
    state = payload.get('state_dict')
    if (not isinstance(state, dict) or not state or
            not all(isinstance(v, torch.Tensor) for v in state.values())):
        raise RuntimeError('candidate %s: payload["state_dict"] missing or '
                           'not a tensor dict' % name)
    arm = payload.get('arm')
    if arm not in ('C', 'S'):
        raise RuntimeError('candidate %s: payload arm %r is not C/S' %
                           (name, arm))
    if expect_arm is not None and arm != expect_arm:
        raise RuntimeError('candidate %s: arm %r != expected %r' %
                           (name, arm, expect_arm))
    if not isinstance(payload.get('steps'), int) or payload['steps'] <= 0:
        raise RuntimeError('candidate %s: payload steps missing/invalid' % name)
    state = {k[7:] if k.startswith('module.') else k: v
             for k, v in state.items()}
    model.load_state_dict(state, strict=True)   # fail closed on any mismatch
    loaded = model.state_dict()
    for k in sorted(state):
        if k not in loaded or not torch.equal(loaded[k].detach().cpu(),
                                              state[k].detach().cpu()):
            raise RuntimeError('candidate %s: bitwise load verification '
                               'failed for %s' % (name, k))
    return {'arm': arm, 'steps': payload['steps'], 'seed': payload.get('seed')}


def student_prob(model, img, meta):
    """INDEPENDENT student probability via _student_prob ONLY.

    Mirrors phase_cs.student_predict: unwrap a possible tuple/list return
    and squeeze the batch dim.  Never calls model.inference() -- that path
    returns the 0.5*reference + 0.5*student blend, which is forbidden here.
    """
    p = model._student_prob(img, meta, True)
    if isinstance(p, (tuple, list)):
        p = p[0]
    if torch.is_tensor(p) and p.dim() == 4:
        p = p[0]
    return p


def validate_prob(prob_np, ori, where):
    """(4,H,W), shape == ori_shape, finite, within [0, 1]."""
    if prob_np.ndim != 3 or prob_np.shape[0] != 4:
        raise RuntimeError('%s: prob must be (4,H,W), got %r' %
                           (where, tuple(prob_np.shape)))
    if tuple(prob_np.shape[1:]) != tuple(ori):
        raise RuntimeError('%s: prob shape %r != ori_shape %r' %
                           (where, tuple(prob_np.shape[1:]), tuple(ori)))
    if (not np.isfinite(prob_np).all() or prob_np.min() < 0.0 or
            prob_np.max() > 1.0):
        raise RuntimeError('%s: prob not finite in [0,1]' % where)


class ChampionProbCache(object):
    """Disk cache of champion.predict probabilities (fp16, per image).

    A present-but-inconsistent cache fails closed (version, identity,
    split, image ids, per-entry file sha256 / dtype / shape / finite /
    range are all validated).  ``identity`` is a sha256 over the champion
    spec, ALL hashed weight/config assets from settings and the eval
    config text, so any champion or pipeline change invalidates the cache
    instead of silently reusing stale probabilities.  The cache directory
    is normalized with no '..' component; every cached file path is
    rebuilt from that validated directory plus a regex-validated stem; the
    manifest's stored path string is only compared as text (a mismatch
    fails closed) and is never itself opened.
    """
    VERSION = 'champprob-f32-v1'
    DTYPE = 'float32'

    def __init__(self, cache_dir, settings, split, stems):
        self.dir = safe_dir(cache_dir, 'champion cache dir')
        os.makedirs(self.dir, exist_ok=True)
        self.manifest_path = os.path.join(self.dir,
                                          'manifest_%s.json' % split)
        self.split = split
        self.stems = sorted(check_stem(s, 'cache init') for s in stems)
        self._identity = self._compute_identity(settings)
        self._manifest = None
        self._entries = None
        self.mode = None                       # 'reuse' or 'build'

    def _compute_identity(self, settings):
        blob = {
            'champion': settings['champion'],
            'weights_sha256': settings['_hashes'],
            'config_text_sha256': hashlib.sha256(
                settings['_cfg'].dump().encode('utf-8')).hexdigest(),
        }
        return hashlib.sha256(json.dumps(blob, sort_keys=True,
                                         default=repr).encode('utf-8')
                              ).hexdigest()

    def _canonical_npy(self, stem):
        """Path rebuilt from validated dir + stem; the ONLY path form used
        for any file operation."""
        return os.path.join(self.dir, '%s_%s.npy' % (self.split, stem))

    @staticmethod
    def _fail_if(cond, what):
        if cond:
            raise RuntimeError('champion cache corrupt: %s mismatch' % what)

    def complete(self):
        """True = verified reusable; False = absent (clean); corrupt = raise."""
        if not os.path.isfile(self.manifest_path):
            return False
        try:
            with open(self.manifest_path, 'r', encoding='utf-8') as f:
                man = json.load(f)
        except (IOError, ValueError):
            raise RuntimeError('champion cache manifest unreadable: %s' %
                               self.manifest_path)
        self._fail_if(man.get('version') != self.VERSION, 'version')
        self._fail_if(man.get('identity') != self._identity, 'identity')
        self._fail_if(man.get('split') != self.split, 'split')
        self._fail_if(man.get('dtype') != self.DTYPE, 'dtype')
        self._fail_if(sorted(man.get('image_ids', [])) != self.stems,
                      'image ids')
        entries = man.get('entries', {})
        self._fail_if(not entries, 'no entries')
        for stem, e in sorted(entries.items()):
            check_stem(stem, 'cache entry key')
            npy = self._canonical_npy(stem)    # rebuilt; manifest string unused
            self._fail_if(e.get('npy') != npy, 'pinned path %s' % stem)
            self._fail_if(not os.path.isfile(npy), 'file %s' % stem)
            self._fail_if(rt.sha256(npy) != e.get('sha256'), 'hash %s' % stem)
            shape = e.get('shape')
            self._fail_if(not isinstance(shape, list) or len(shape) != 3 or
                          shape[2] != 4, 'entry shape %s' % stem)
            arr = np.load(npy, mmap_mode='r')
            self._fail_if(arr.dtype != np.float32 or arr.ndim != 3 or
                          tuple(arr.shape) != tuple(shape),
                          'array header %s' % stem)
            a32 = np.asarray(arr, dtype=np.float32)
            self._fail_if(not np.isfinite(a32).all() or
                          a32.min() < 0.0 or a32.max() > 1.0,
                          'finite/range %s' % stem)
            del arr, a32
        self._manifest = man
        return True

    def prepare(self):
        """Verify an existing cache or enter build mode."""
        if self.complete():
            self.mode = 'reuse'
        else:
            self._entries = {}
            self.mode = 'build'
        return self.mode

    def manifest_entries(self):
        if self._manifest is None:
            raise RuntimeError('champion cache not verified')
        return self._manifest['entries']

    def has(self, stem):
        return self.mode == 'reuse' and stem in self.manifest_entries()

    def get(self, stem):
        """(4,H,W) float32 numpy; file sha256 re-verified on EVERY access."""
        if self.mode != 'reuse':
            raise RuntimeError('champion cache get outside reuse mode')
        e = self.manifest_entries()[stem]
        npy = self._canonical_npy(stem)        # rebuilt; manifest string unused
        self._fail_if(e.get('npy') != npy, 'pinned path %s' % stem)
        if rt.sha256(npy) != e.get('sha256'):
            raise RuntimeError('champion cache entry hash mismatch: %s' % stem)
        arr = np.load(npy, mmap_mode='r')
        if (arr.dtype != np.float32 or arr.ndim != 3 or
                tuple(arr.shape) != tuple(e.get('shape', ())) or
                arr.shape[-1] != 4):
            raise RuntimeError('champion cache header mismatch: %s' % stem)
        out = np.ascontiguousarray(
            np.asarray(arr, dtype=np.float32).transpose(2, 0, 1))
        del arr
        return out

    def add(self, stem, prob):
        """prob: (4,H,W) cuda/cpu tensor; stored float32 HWC, atomically.

        The temporary filename is rebuilt from the validated directory
        (never concatenated onto a data-derived path) and the final
        replace is atomic, so a crash never leaves a half-written npy.
        """
        if self.mode != 'build':
            raise RuntimeError('champion cache add outside build mode')
        check_stem(stem, 'cache add')
        if not torch.is_tensor(prob) or prob.dim() != 3 or prob.shape[0] != 4:
            raise RuntimeError('champion cache add: bad prob shape %r' %
                               (tuple(prob.shape),))
        arr = np.ascontiguousarray(
            prob.detach().cpu().numpy().astype(np.float32).transpose(1, 2, 0))
        npy = self._canonical_npy(stem)
        tmp = os.path.join(self.dir, '.building_%s_%s.npy' % (self.split, stem))
        np.save(tmp, arr)
        os.replace(tmp, npy)
        self._entries[stem] = {'npy': npy, 'shape': list(arr.shape),
                               'dtype': self.DTYPE, 'sha256': rt.sha256(npy)}

    def finalize(self):
        """Write the manifest through runtime.atomic_json (tmp+os.replace)."""
        if self.mode != 'build':
            raise RuntimeError('champion cache finalize outside build mode')
        if sorted(self._entries) != self.stems:
            raise RuntimeError('champion cache build incomplete: %d entries, '
                               'expected %d' % (len(self._entries),
                                                len(self.stems)))
        man = {'version': self.VERSION, 'identity': self._identity,
               'split': self.split, 'dtype': self.DTYPE,
               'image_ids': self.stems, 'entries': self._entries}
        rt.atomic_json(self.manifest_path, man)
        self._manifest = man
        self.mode = 'reuse'


def evaluate_candidates(settings, champion, candidates, split, deadline,
                        champ_cache):
    """One dataset pass; champion / candidate / fixed-mix counts.

    Returns {name: {'miou', 'per_class', 'global_counts', 'per_image'}}
    plus '_meta'.  Metric = runtime.per_image_counts (strict >0.5)
    accumulated over the whole split, then runtime.global_miou (raises on
    zero union).
    """
    import cv2
    from mmseg.datasets import build_dataset
    cfg = settings['_cfg']
    d = copy.deepcopy(cfg.data[split])
    d.test_mode = True
    d.data_root = settings['data_root']
    d.img_dir = os.path.join(settings['data_root'], 'images', split)
    d.ann_dir = os.path.join(settings['data_root'], 'labels', split)
    ds = build_dataset(d)
    if len(ds) != EXPECTED_COUNTS[split]:
        raise RuntimeError('split %s has %d items, expected %d' %
                           (split, len(ds), EXPECTED_COUNTS[split]))
    gt_maps = ds.get_gt_seg_maps()

    cand_names = [n for n, _ in candidates]
    names = (['champion'] + ['cand_%s' % n for n in cand_names] +
             ['mix_%s' % n for n in cand_names])
    totals = {n: [[0, 0, 0] for _ in range(4)] for n in names}
    per_image = {n: {} for n in names}
    cache_hits = 0
    t0 = time.time()

    if champ_cache is not None:
        mode = champ_cache.prepare()
        log('champion cache mode: %s (%s)' % (mode, champ_cache.dir))
        if mode == 'reuse':
            entries = champ_cache.manifest_entries()
            missing = [s for s in champ_cache.stems if s not in entries]
            if missing:
                raise RuntimeError('champion cache incomplete: %d images '
                                   'missing (e.g. %s)' %
                                   (len(missing), missing[:3]))

    for i in range(len(ds)):
        if deadline is not None and time.time() > deadline:
            raise RuntimeError('deadline hit during %s candidate eval' % split)
        s = ds[i]
        img = rt._unwrap_img(s['img']).unsqueeze(0).cuda()
        meta = rt._unwrap_meta(s['img_metas'])
        stem = check_stem(
            os.path.splitext(os.path.basename(meta[0]['ori_filename']))[0],
            'image %d' % i)
        ori = tuple(meta[0]['ori_shape'][:2])
        with torch.no_grad():
            if champ_cache is not None and champ_cache.has(stem):
                pc_np = champ_cache.get(stem)
                cache_hits += 1
                pc = torch.from_numpy(pc_np).cuda()
            else:
                if champ_cache is not None and champ_cache.mode == 'reuse':
                    raise RuntimeError('champion cache missing image %s' % stem)
                pc = champion.predict(img, meta).float()
                pc_np = pc.detach().cpu().numpy().astype(np.float32)
                if champ_cache is not None:
                    champ_cache.add(stem, pc)
            validate_prob(pc_np, ori, 'champion[%s]' % stem)

            gt = np.asarray(gt_maps[i]).astype(np.int64)
            if gt.shape != ori:
                gt = cv2.resize(gt.astype(np.uint8), (ori[1], ori[0]),
                                interpolation=cv2.INTER_NEAREST).astype(np.int64)
            if not np.all(np.isin(gt, (0, 1, 2, 3, 4, 255))):
                raise RuntimeError('gt contains invalid labels: %s' % stem)

            cnt = rt.per_image_counts(pc_np, gt)
            per_image['champion'][stem] = {rt.CLASSES[c]: cnt[c]
                                           for c in range(4)}
            for c in range(4):
                for j in range(3):
                    totals['champion'][c][j] += cnt[c][j]

            for name, model in candidates:
                pn = student_prob(model, img, meta)
                pn_np = pn.detach().cpu().numpy().astype(np.float32)
                validate_prob(pn_np, ori, 'cand_%s[%s]' % (name, stem))
                pm = MIX_CHAMPION * pc + MIX_CANDIDATE * pn
                pm_np = pm.detach().cpu().numpy().astype(np.float32)
                validate_prob(pm_np, ori, 'mix_%s[%s]' % (name, stem))
                for tag, pnp in (('cand_%s' % name, pn_np),
                                 ('mix_%s' % name, pm_np)):
                    cnt = rt.per_image_counts(pnp, gt)
                    per_image[tag][stem] = {rt.CLASSES[c]: cnt[c]
                                            for c in range(4)}
                    for c in range(4):
                        for j in range(3):
                            totals[tag][c][j] += cnt[c][j]
        if (i + 1) % 25 == 0 or (i + 1) == len(ds):
            log('eval %s %d/%d (%.1fs)' %
                (split, i + 1, len(ds), time.time() - t0))

    if champ_cache is not None and champ_cache.mode == 'build':
        champ_cache.finalize()
        log('champion cache finalized: %d entries' % len(champ_cache.stems))

    results = {}
    for name in names:
        miou, per = rt.global_miou([tuple(x) for x in totals[name]])
        results[name] = {'miou': miou, 'per_class': per,
                         'global_counts': totals[name],
                         'per_image': per_image[name]}
        log('%s %s mIoU %.6f' % (split, name, miou))
    results['_meta'] = {'cache_hits': cache_hits}
    return results


def main():
    ap = argparse.ArgumentParser(
        description='Fixed 0.75*champion + 0.25*candidate evaluation '
                    '(no weight scan; any failure exits non-zero).')
    ap.add_argument('--settings', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--prefix', default='evalcand')
    ap.add_argument('--split', choices=('val', 'test'), default='val')
    ap.add_argument('--candidate', action='append', required=True,
                    metavar='NAME=PATH',
                    help='C/S node checkpoint (repeatable), e.g. '
                         'C=/run/C_step_1500.pth')
    ap.add_argument('--candidate-arm', action='append', default=[],
                    metavar='NAME=ARM',
                    help='optional expected payload arm check (C or S)')
    ap.add_argument('--champion-cache-dir', default=None,
                    help='enable champion probability cache in this directory')
    ap.add_argument('--deadline', default=None,
                    help='ISO-8601 or epoch seconds; exceeding it is a '
                         'hard failure (exit non-zero)')
    ap.add_argument('--selection-lock', default=None,
                    help='optional selection_lock.json to cross-check '
                         '--locked-candidate')
    ap.add_argument('--allow-test', action='store_true',
                    help='grant test split (required together with '
                         '--locked-candidate)')
    ap.add_argument('--locked-candidate', default=None,
                    help='candidate name locked on val before the test run')
    ap.add_argument('--write-lock', action='store_true',
                    help='val only: write locked_candidate.json (best mIoU, '
                         'ties -> champion)')
    args = ap.parse_args()

    if not NAME_RE.match(args.prefix or ''):
        raise RuntimeError('bad --prefix: %r' % args.prefix)
    args.out_dir = safe_dir(args.out_dir, '--out-dir')

    # ---- locked candidate: validate + cross-check whenever given ----
    if args.locked_candidate is not None:
        if not NAME_RE.match(args.locked_candidate):
            raise RuntimeError('bad --locked-candidate: %r' %
                               args.locked_candidate)
        lock_path = args.selection_lock or os.path.join(
            args.out_dir, 'locked_candidate.json')
        if os.path.isfile(lock_path):
            with open(lock_path, 'r', encoding='utf-8') as f:
                prev = json.load(f).get('candidate')
            if prev is not None and prev != args.locked_candidate:
                raise RuntimeError('locked candidate mismatch: recorded %r '
                                   '!= given %r' % (prev, args.locked_candidate))

    # ---- test gate: BOTH --allow-test AND --locked-candidate mandatory ----
    if args.split == 'test':
        if not args.allow_test:
            raise RuntimeError('test split requires --allow-test')
        if args.locked_candidate is None:
            raise RuntimeError('test split requires --locked-candidate '
                               '(locked on val BEFORE the test run)')
    if args.write_lock and args.split != 'val':
        raise RuntimeError('--write-lock is val-only')

    # ---- candidates ----
    candidates, seen, arm_expect = [], set(), {}
    for spec in args.candidate:
        if '=' not in spec:
            raise RuntimeError('bad --candidate %r (need NAME=PATH)' % spec)
        name, path = spec.split('=', 1)
        name = name.strip()
        if not NAME_RE.match(name) or name in seen:
            raise RuntimeError('bad/duplicate candidate name %r' % name)
        if name == 'champion' or name.startswith(('cand_', 'mix_')):
            raise RuntimeError('candidate name %r is reserved' % name)
        if not os.path.isfile(path):
            raise RuntimeError('candidate checkpoint missing: %s' % path)
        seen.add(name)
        candidates.append((name, path))
    for spec in args.candidate_arm:
        if '=' not in spec:
            raise RuntimeError('bad --candidate-arm %r (need NAME=ARM)' % spec)
        n, a = spec.split('=', 1)
        if a not in ('C', 'S') or n.strip() not in seen:
            raise RuntimeError('bad --candidate-arm %r' % spec)
        arm_expect[n.strip()] = a

    # ---- settings / repo / determinism ----
    settings = load_settings(args.settings)
    os.makedirs(args.out_dir, exist_ok=True)
    rt.inject_repo(settings['repo'])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    deadline = rt.parse_deadline(args.deadline) if args.deadline else None

    # ---- split audit + manifest sha256 ----
    assignment = rt.audit_split(settings['data_root'])
    split_manifest_sha = hashlib.sha256(json.dumps(
        assignment, sort_keys=True).encode('utf-8')).hexdigest()
    stems = sorted(v['stem'] for v in assignment.values()
                   if v['split'] == args.split)
    log('split audit OK; %s stems = %d' % (args.split, len(stems)))

    # ---- candidate models: fresh strict build, then strict node load ----
    cand_models, cand_meta = [], {}
    for name, path in candidates:
        model, _ = rp.build_student(settings)   # fresh build + strict B/D init
        model.eval()
        info = load_node_checkpoint(model, path, name, arm_expect.get(name))
        for p in model.parameters():
            p.requires_grad_(False)
        cand_models.append((name, model))
        cand_meta[name] = {'path': os.path.abspath(path),
                           'sha256': rp.sha256(path), **info}
        log('candidate %s loaded: arm=%s steps=%s seed=%s' %
            (name, info['arm'], info['steps'], info.get('seed')))
    torch.cuda.empty_cache()

    # ---- champion: EXISTING ChampionEnsemble (E_1500 member + h4_best MA head) ----
    champion = rt.ChampionEnsemble(settings, settings['champion'])
    log('champion built: members=%s head=%s' %
        ([m['name'] for m in settings['champion']['members']],
         settings['champion']['head']))

    champ_cache = None
    if args.champion_cache_dir:
        champ_cache = ChampionProbCache(args.champion_cache_dir, settings,
                                        args.split, stems)

    with torch.no_grad():
        results = evaluate_candidates(settings, champion, cand_models,
                                      args.split, deadline, champ_cache)

    # ---- persist: per-name counts + summary (all atomic) ----
    cand_names = [n for n, _ in cand_models]
    names = (['champion'] + ['cand_%s' % n for n in cand_names] +
             ['mix_%s' % n for n in cand_names])
    for name in names:
        payload = dict(results[name])
        payload['split'] = args.split
        payload['name'] = name
        payload['locked_candidate'] = (args.locked_candidate
                                       if args.split == 'test' else None)
        rt.atomic_json(os.path.join(args.out_dir,
                                    '%s_%s_counts.json' % (args.prefix, name)),
                       payload)

    if args.split == 'val' and args.write_lock:
        best = None
        for name in names:                 # champion first: ties -> champion
            m = results[name]['miou']
            if best is None or m > best[1] + 1e-12:
                best = (name, m)
        rt.atomic_json(os.path.join(args.out_dir, 'locked_candidate.json'),
                       {'candidate': best[0], 'miou': best[1],
                        'split': 'val',
                        'created': time.strftime('%Y-%m-%dT%H:%M:%S')})
        log('locked candidate (val): %s (%.6f)' % best)

    cache_mode = (champ_cache.mode if champ_cache is not None else 'off')
    summary = {
        'script': os.path.abspath(__file__),
        'created': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'split': args.split,
        'n_images': len(stems),
        'mix': {'champion': MIX_CHAMPION, 'candidate': MIX_CANDIDATE,
                'weight_scan': False},
        'champion_pipeline': settings['champion'],
        'candidates': [cand_meta[n] for n in cand_names],
        'champion_cache': None if champ_cache is None else {
            'dir': champ_cache.dir, 'mode': cache_mode,
            'identity': champ_cache._identity,
            'hits': results['_meta']['cache_hits']},
        'hashes': {
            'weights': settings['_hashes'],
            'candidate_checkpoints': {n: cand_meta[n]['sha256']
                                      for n in cand_names},
            'config': {'file': settings['_hashes'].get('config'),
                       'text_sha256': hashlib.sha256(
                           settings['_cfg'].dump().encode('utf-8')).hexdigest()},
            'split_manifest_sha256': split_manifest_sha,
        },
        'split_stems': {args.split: stems},
        'results': {n: {'miou': results[n]['miou'],
                        'per_class': results[n]['per_class']}
                    for n in names},
        'locked_candidate': (args.locked_candidate
                             if args.split == 'test' else None),
        'argv': sys.argv,
    }
    rt.atomic_json(os.path.join(args.out_dir, '%s_summary.json' % args.prefix),
                   summary)
    log('DONE split=%s mious=%s' %
        (args.split, {n: round(results[n]['miou'], 6) for n in names}))


if __name__ == '__main__':
    try:
        main()
    except SystemExit:
        raise
    except BaseException:
        import traceback
        traceback.print_exc()
        sys.exit(1)
