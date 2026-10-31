# runtime.py
# -*- coding: utf-8 -*-
"""Runtime layer: repo injection, model init, caching, geometry, losses,
evaluation and small protocol helpers.  Python 3.8+, legacy mmcv/mmseg.

Shared protocol primitives (sha256, strict_initialize, configure_trainable,
freeze_bn, paired_geometry, loss_pair, normalization constants) and decision
ops (choose_steps, gate_first, choose_checkpoint) are IMPORTED from
fusion_core / ops -- no duplicates live here.
"""
import copy
import hashlib
import json
import os
import random
import sys
import time

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# single source of truth imports (no local duplicates)
# ---------------------------------------------------------------------------
from fusion_core import (sha256, strict_initialize, configure_trainable,
                         freeze_bn, paired_geometry, loss_pair,
                         IMG_MEAN, IMG_STD)
from ops import choose_steps, gate_first, choose_checkpoint  # noqa: F401


# ---------------------------------------------------------------------------
# repo injection / config / model building
# ---------------------------------------------------------------------------
def inject_repo(repo):
    repo = os.path.abspath(repo)
    if not os.path.isdir(repo):
        raise RuntimeError('repo directory does not exist: %s' % repo)
    os.chdir(repo)
    if repo not in sys.path:
        sys.path.insert(0, repo)
    return repo


def atomic_json(path, obj):
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    f = None
    os.replace(tmp, path)


def load_config(repo, config_rel):
    """Load config, register custom imports and disable pretrained weights.
    Does NOT touch the model's reference/source_checkpoint setting here --
    the driver patches cfg.model['source_checkpoint'] to an absolute B path
    BEFORE the constructor."""
    from mmcv import Config
    cfg = Config.fromfile(os.path.join(repo, config_rel))
    model = cfg.get('model', None)
    if model is not None and 'pretrained' in model:
        model['pretrained'] = None
    _register_imports(cfg)
    return cfg


def _register_imports(cfg):
    """Ensure config custom_imports are executed.  mmcv's Config.fromfile
    already runs them via import_modules_from_strings; for mmseg versions
    that ALSO expose mmseg.utils.register_custom_imports we call it, and
    otherwise fall back to a direct importlib import so a broken module
    path still fails loudly instead of silently skipping registration."""
    imports = cfg.get('custom_imports', None)
    if not imports:
        return
    mods = list(imports.get('imports', [])) if isinstance(imports, dict) else []
    if not mods:
        return
    try:
        from mmseg.utils import register_custom_imports      # newer mmseg
    except ImportError:
        register_custom_imports = None
    if register_custom_imports is not None:
        register_custom_imports(imports=mods,
                                allow_failed_imports=imports.get(
                                    'allow_failed_imports', False))
    else:
        import importlib
        for mod in mods:                     # old fork: fromfile already ran
            importlib.import_module(mod)     # them; re-import is idempotent


def build_model(cfg):
    _register_imports(cfg)
    from mmseg.models import build_segmentor
    return build_segmentor(cfg.model)


def resolve_repo_path(repo, p):
    """Resolve a settings path against the repo root (absolute passes
    through).  ONE convention shared by the driver (_abs) and every raw
    consumer (ChampionEnsemble member/head paths) -- the guard script runs
    with CWD=glm dir, NOT the repo, so bare 'weights/x.pth' strings must
    never reach torch.load unchanged."""
    return p if os.path.isabs(p) else os.path.join(repo, p)


# ---------------------------------------------------------------------------
# checkpoint loading (trusted local files; strict member loads)
# ---------------------------------------------------------------------------
def _load_state(path):
    """weights_only=False (trusted local file) with fallback for old torch
    versions that lack the kwarg."""
    try:
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location='cpu')
    if not isinstance(ckpt, dict):
        raise RuntimeError('bad checkpoint (not a dict): %s' % path)
    state = ckpt.get('state_dict', ckpt)
    return {k[7:] if k.startswith('module.') else k: v
            for k, v in state.items()}


def _strict_load_full(model, path):
    """Full-state, strict load.  Any missing/unexpected key fails closed."""
    state = _load_state(path)
    model.load_state_dict(state, strict=True)
    return model


def snapshot_state(model):
    return {n: p.detach().cpu().clone() for n, p in model.named_parameters()}


def snapshot_buffers(model):
    return {n: b.detach().cpu().clone() for n, b in model.named_buffers()}


# ---------------------------------------------------------------------------
# first optimizer step evidence (all-boolean dict; driver does all(ev.values()))
# ---------------------------------------------------------------------------
def _authorized_param(name):
    if name.startswith('reference.'):
        return False
    return name.startswith('decode_head') or name.startswith('backbone.stage4')


def first_update_evidence(model, ref_snapshot, bn_snapshot):
    """Verify after the first update:
    - every frozen parameter (reference AND any other non-allowed) unchanged
    - all BN buffers (running_mean/var, num_batches_tracked) unchanged
    - gradients present and nonzero on >=1 decode_head param AND >=1
      backbone.stage4 param, with finite total grad norm
    - at least one allowed (trainable) parameter actually changed
    Returns dict of booleans only."""
    ev = {'frozen_params_unchanged': True,
          'bn_buffers_unchanged': True,
          'grad_at_decoder': False,
          'grad_at_stage4': False,
          'grad_norm_finite': False,
          'allowed_changed': False}
    total_norm = 0.0
    for n, p in model.named_parameters():
        cur = p.detach().cpu()
        old = ref_snapshot.get(n)
        changed = old is not None and not torch.equal(cur, old)
        if _authorized_param(n):
            if changed:
                ev['allowed_changed'] = True
            if p.grad is not None:
                g = p.grad.detach().cpu()
                gn = float(torch.norm(g))
                if gn > 0:
                    total_norm += gn * gn
                    if n.startswith('decode_head'):
                        ev['grad_at_decoder'] = True
                    if n.startswith('backbone.stage4'):
                        ev['grad_at_stage4'] = True
        else:
            if changed:
                ev['frozen_params_unchanged'] = False
    ev['grad_norm_finite'] = bool(np.isfinite(total_norm) and total_norm > 0.0)
    for n, b in model.named_buffers():
        cur = b.detach().cpu()
        old = bn_snapshot.get(n)
        if old is not None and not torch.equal(cur, old):
            if ('running_mean' in n or 'running_var' in n or
                    n.endswith('num_batches_tracked')):
                ev['bn_buffers_unchanged'] = False
    return ev


# ---------------------------------------------------------------------------
# loss passthrough re-export guard (loss_pair imported from fusion_core)
# ---------------------------------------------------------------------------
__all__ = ['inject_repo', 'sha256', 'atomic_json', 'load_config', 'build_model',
           'resolve_repo_path', '_load_state', 'strict_initialize',
           'configure_trainable', 'freeze_bn', 'paired_geometry', 'loss_pair',
           'snapshot_state', 'snapshot_buffers', 'first_update_evidence',
           'ReferenceCache', 'per_image_counts', 'global_miou',
           'evaluate_split', 'CLASSES', 'CorrectionHead', 'ChampionEnsemble',
           'mix_probs', 'get_rng_state', 'set_rng_state', 'audit_split',
           'parse_deadline', 'choose_steps', 'gate_first', 'choose_checkpoint',
           'IMG_MEAN', 'IMG_STD']


# ---------------------------------------------------------------------------
# reference probability cache
# ---------------------------------------------------------------------------
def _digest_bytes(b):
    return hashlib.sha256(b).hexdigest()


def _arr_sha(a):
    return _digest_bytes(np.ascontiguousarray(a).tobytes())


def _unwrap_img(x):
    n = 0
    while hasattr(x, 'data') and n < 8:
        x = x.data
        n += 1
    while isinstance(x, (list, tuple)):
        x = x[0]
        while hasattr(x, 'data') and n < 16:
            x = x.data
            n += 1
    return x


def _unwrap_meta(x):
    """Original DataContainer unwrap."""
    n = 0
    while hasattr(x, 'data') and n < 8:
        x = x.data
        n += 1
    if isinstance(x, dict):
        return [x]
    out = []
    for e in x:
        e = e.data if hasattr(e, 'data') else e
        out.extend([e] if isinstance(e, dict) else [])
    return out


class ReferenceCache(object):
    """Reference probability cache at the 2048x2048 test-time resolution.

    The reference model sees the SAME deterministic val-style pipeline image
    (Resize keep_ratio=False to 2048x2048, Normalize) it would see during
    evaluation; probabilities are stored WITHOUT the rescale-to-original step
    so training geometry (resize->rotate->crop->flip at 2048) aligns with the
    cache with ZERO extra resampling of the reference map.
    """
    VERSION = 'refcache-v3'
    DTYPE = 'float16'              # disk dtype; fp32 in memory
    RESOLUTION = (2048, 2048)

    def __init__(self, cache_dir, settings, expected_count=None):
        self.dir = os.path.abspath(cache_dir)
        os.makedirs(self.dir, exist_ok=True)
        self.manifest_path = os.path.join(self.dir, 'manifest.json')
        self.complete_path = os.path.join(self.dir, 'COMPLETE')
        self.settings = settings
        self._expected_count = expected_count  # test-only override
        self._manifest = None

    # -- identity material ------------------------------------------------
    def _cfg_digest(self):
        cfg = self.settings['_cfg']
        return _digest_bytes(cfg.dump().encode('utf-8'))

    def _train_image_ids(self):
        data_root = self.settings['data_root']
        d = os.path.join(data_root, 'images', 'train')
        ids = [os.path.splitext(f)[0] for f in sorted(os.listdir(d))
               if f.lower().endswith(('.jpg', '.jpeg'))]
        lst = os.path.join(data_root, 'train.txt')
        if os.path.isfile(lst):                    # optional cross-check
            listed = [os.path.splitext(os.path.basename(l.strip()))[0]
                      for l in open(lst, encoding='utf-8') if l.strip()]
            if sorted(listed) != sorted(ids):
                raise RuntimeError('train.txt disagrees with images/train')
        return sorted(ids)

    def _source_images_digest(self):
        """Digest over the full train image set (name+file sha), full split
        IDs -- never only dict keys."""
        data_root = self.settings['data_root']
        d = os.path.join(data_root, 'images', 'train')
        h = hashlib.sha256()
        for f in sorted(os.listdir(d)):
            if f.lower().endswith(('.jpg', '.jpeg')):
                h.update(f.encode('utf-8'))
                h.update(sha256(os.path.join(d, f)).encode('utf-8'))
        return h.hexdigest()

    def _ref_weight_sha(self):
        return self.settings['_hashes']['reference_checkpoint']

    # -- completeness ------------------------------------------------------
    def complete(self):
        """Missing cache -> False (clean).  Present but inconsistent ->
        RuntimeError (corruption, fail closed).  Validates: version, split
        == train with exact full ID list, source image digest, reference
        weight sha (NOT student), effective config digest, normalization,
        per-entry file hash / dtype / shape / finite / range.  Entry hashes
        are validated HERE (once per protocol start); get() relies on this
        and skips per-access re-hashing for speed."""
        if not (os.path.isfile(self.manifest_path) and
                os.path.isfile(self.complete_path)):
            return False
        try:
            with open(self.manifest_path, 'r', encoding='utf-8') as f:
                man = json.load(f)
        except (IOError, ValueError):
            raise RuntimeError('reference cache manifest unreadable')
        self._fail_if(man.get('version') != self.VERSION, 'version')
        self._fail_if(man.get('split') != 'train', 'split')
        self._fail_if(man.get('dtype') != self.DTYPE, 'dtype')
        self._fail_if(list(man.get('resolution', [])) !=
                      list(self.RESOLUTION), 'resolution')
        self._fail_if(sorted(man.get('image_ids', [])) !=
                      self._train_image_ids(), 'train image IDs')
        self._fail_if(man.get('source_images_sha256') !=
                      self._source_images_digest(), 'source images')
        self._fail_if(man.get('reference_weight_sha256') !=
                      self._ref_weight_sha(), 'reference weight')
        self._fail_if(man.get('config_sha256') != self._cfg_digest(),
                      'config digest')
        self._fail_if(man.get('normalization') !=
                      {'mean': IMG_MEAN.tolist(), 'std': IMG_STD.tolist()},
                      'normalization')
        entries = man.get('entries', {})
        if not entries:
            raise RuntimeError('reference cache corrupt: no entries')
        for stem, e in sorted(entries.items()):
            npy = e.get('npy')
            self._fail_if(not npy or not os.path.isfile(npy),
                          'missing file %s' % stem)
            self._fail_if(not npy.endswith('.npy'), 'format %s' % stem)
            self._fail_if(sha256(npy) != e.get('sha256'),
                          'file hash %s' % stem)
            self._fail_if(list(e.get('shape', [])) !=
                          [self.RESOLUTION[0], self.RESOLUTION[1], 4],
                          'shape %s' % stem)
            arr = np.load(npy, mmap_mode='r')
            self._fail_if(arr.dtype != np.float16 or
                          tuple(arr.shape) != (self.RESOLUTION[0],
                                               self.RESOLUTION[1], 4),
                          'array header %s' % stem)
            a32 = np.asarray(arr, dtype=np.float32)
            self._fail_if(not np.isfinite(a32).all() or
                          a32.min() < 0.0 or a32.max() > 1.0,
                          'finite/range %s' % stem)
            del arr, a32
        self._fail_if(self._expected_count is not None and
                      len(entries) != self._expected_count, 'entry count')
        self._manifest = man
        return True

    @staticmethod
    def _fail_if(cond, what):
        if cond:
            raise RuntimeError('reference cache corrupt: %s mismatch' % what)

    # -- access ------------------------------------------------------------
    def get(self, stem):
        """Lazy load of one entry.  Header fields AND the entry sha256 are
        re-verified on EVERY access (tamper/bit-rot fail-closed; ~60-100ms
        for a 32MB entry, <8% of a training step).  Full finite/range
        validation happens once in complete()."""
        if self._manifest is None:
            if not self.complete():
                raise RuntimeError('reference cache incomplete or corrupt')
        e = self._manifest['entries'][stem]
        npy = e['npy']
        if sha256(npy) != e.get('sha256'):
            raise RuntimeError('cache entry hash mismatch: %s' % stem)
        arr = np.load(npy, mmap_mode='r')
        if arr.dtype != np.float16 or tuple(arr.shape) != (
                self.RESOLUTION[0], self.RESOLUTION[1], 4):
            raise RuntimeError('cache entry header mismatch: %s' % stem)
        return np.asarray(arr, dtype=np.float32)

    def __getitem__(self, stem):
        return self.get(stem)

    def __contains__(self, stem):
        if self._manifest is None:
            if not os.path.isfile(self.manifest_path):
                return False
            with open(self.manifest_path, 'r', encoding='utf-8') as f:
                self._manifest = json.load(f)
        return stem in self._manifest.get('entries', {})

    # -- build --------------------------------------------------------------
    def build(self, model, image_ids, deadline, log):
        """Reference probabilities at the 2048x2048 test resolution via the
        model's own reference pipeline.  The dataset is built from the
        DETERMINISTIC val/test pipeline template (cfg.data.val -- Resize
        keep_ratio=False to 2048, Normalize; MultiScaleFlipAug flip=False),
        NEVER from the augmenting train pipeline: test_mode=True does not
        disable RandomRotate/ForegroundMixedCrop/RandomFlip.  Only the train
        split is cached; labels are never read."""
        from mmseg.datasets import build_dataset
        cfg = self.settings['_cfg']
        d = copy.deepcopy(cfg.data.val)      # deterministic test pipeline
        d.test_mode = True
        d.pop('split', None)                 # membership = img_dir listing
        d.img_dir = os.path.join(self.settings['data_root'], 'images', 'train')
        d.ann_dir = os.path.join(self.settings['data_root'], 'labels', 'train')
        ds = build_dataset(d)
        # verify built split matches expected train IDs
        built = sorted(os.path.splitext(os.path.basename(
            ds.img_infos[i]['filename']))[0] for i in range(len(ds)))
        if built != sorted(os.path.splitext(os.path.basename(x))[0]
                           for x in image_ids):
            raise RuntimeError('reference cache: dataset split != train IDs')
        expected = len(ds) if self._expected_count is None \
            else self._expected_count
        model = model.cuda().eval()
        entries = {}
        for i in range(len(ds)):
            if time.time() > deadline:
                raise RuntimeError('deadline hit during reference cache')
            s = ds[i]
            img = _unwrap_img(s['img']).unsqueeze(0).cuda()
            meta = _unwrap_meta(s['img_metas'])
            with torch.no_grad():
                # rescale=False: probabilities stay at the 2048x2048 input
                # resolution -- exactly what the reference produced on the
                # same test-pipeline image (no resample ever).
                prob = model._reference_prob(img, meta, False)
            prob = prob[0].detach().cpu().numpy()          # (4,2048,2048)
            if prob.shape != (4, self.RESOLUTION[0], self.RESOLUTION[1]):
                raise RuntimeError(
                    'reference prob shape %r != (4, %d, %d); the val pipeline '
                    'is expected to resize to 2048x2048' % (
                        prob.shape, self.RESOLUTION[0], self.RESOLUTION[1]))
            if not np.isfinite(prob).all() or prob.min() < 0.0 or prob.max() > 1.0:
                raise RuntimeError('reference prob not finite in [0,1]')
            prob = np.ascontiguousarray(
                prob.astype(np.float16).transpose(1, 2, 0))  # H,W,4 fp16
            stem = os.path.splitext(os.path.basename(
                ds.img_infos[i]['filename']))[0]
            npy = os.path.join(self.dir, stem + '.npy')
            tmp = npy + '.tmp.npy'
            np.save(tmp, prob)
            os.replace(tmp, npy)
            entries[stem] = {'npy': npy,
                             'shape': [self.RESOLUTION[0],
                                       self.RESOLUTION[1], 4],
                             'dtype': self.DTYPE,
                             'sha256': sha256(npy)}
            if (i + 1) % 25 == 0:
                log('refcache %d/%d' % (i + 1, len(ds)))
        if len(entries) != expected:
            raise RuntimeError('cache count mismatch %d != %d' %
                               (len(entries), expected))
        manifest = {'version': self.VERSION, 'split': 'train',
                    'dtype': self.DTYPE,
                    'resolution': list(self.RESOLUTION),
                    'image_ids': sorted(entries),
                    'source_images_sha256': self._source_images_digest(),
                    'reference_weight_sha256': self._ref_weight_sha(),
                    'config_sha256': self._cfg_digest(),
                    'normalization': {'mean': IMG_MEAN.tolist(),
                                      'std': IMG_STD.tolist()},
                    'entries': entries}
        atomic_json(self.manifest_path, manifest)
        tmp = self.complete_path + '.tmp'
        with open(tmp, 'w') as f:
            f.write('ok')
        os.replace(tmp, self.complete_path)
        self._manifest = manifest
        log('reference cache COMPLETE: %d entries at 2048x2048 fp16' %
            len(entries))


# ---------------------------------------------------------------------------
# evaluation: strict per-image 4-class TP/FP/FN then global IoU
# ---------------------------------------------------------------------------
CLASSES = ['EX', 'HE', 'SE', 'MA']


def per_image_counts(prob, gt):
    """prob (4,H,W) float, gt (H,W) int with 255 ignore.  Threshold 0.5
    strict >, background (class 0) not counted."""
    counts = []
    for c in range(4):
        p = prob[c] > 0.5
        g = gt == (c + 1)
        ig = gt == 255
        tp = int(np.logical_and(p, g).sum())
        fp = int(np.logical_and(p, np.logical_and(~g, ~ig)).sum())
        fn = int(np.logical_and(~p, g).sum())
        counts.append([tp, fp, fn])
    return counts


def global_miou(all_counts):
    ious = []
    per = {}
    for c, (tp, fp, fn) in enumerate(all_counts):
        union = tp + fp + fn
        if union == 0:
            raise RuntimeError('IoU union is zero for class %d' % (c + 1))
        iou = tp / union
        ious.append(iou)
        per[CLASSES[c]] = iou
    return float(np.mean(ious)), per


def evaluate_split(predict_fn, split, data_root, cfg, deadline, log,
                   dump_counts=None, expected_count=None,
                   _ds_builder=None):  # _ds_builder: test-only injection
    """Original val/test dataset pipeline; gt nearest-resized to ori_shape.
    Predictor may return (1,4,H,W) or (4,H,W) -- only the batch dim is
    squeezed; shape/finite/range validated strictly.  The caller owns
    model.eval().  Returns (miou, per_class, per_image_counts)."""
    import cv2
    if _ds_builder is None:
        from mmseg.datasets import build_dataset
        d = copy.deepcopy(cfg.data[split])
        d.test_mode = True
        d.img_dir = os.path.join(data_root, 'images', split)
        d.ann_dir = os.path.join(data_root, 'labels', split)
        ds = build_dataset(d)
    else:
        ds = _ds_builder(data_root, split)
    n = len(ds)
    expected = expected_count if expected_count is not None \
        else {'val': 76, 'test': 227}[split]
    if n != expected:
        raise RuntimeError('split %s has %d items, expected %d' %
                           (split, n, expected))
    gt_maps = ds.get_gt_seg_maps()
    total = [[0, 0, 0] for _ in range(4)]
    per_image = {}
    t0 = time.time()
    for i in range(n):
        if time.time() > deadline:
            raise RuntimeError('deadline hit during %s eval' % split)
        s = ds[i]
        img = _unwrap_img(s['img']).unsqueeze(0).cuda()
        meta = _unwrap_meta(s['img_metas'])
        prob = predict_fn(img, meta)
        if torch.is_tensor(prob):
            prob = prob.detach().cpu().numpy()
        prob = np.asarray(prob, dtype=np.float32)
        if prob.ndim == 4:
            if prob.shape[0] != 1:
                raise RuntimeError('predictor returned batch >1')
            prob = prob[0]                      # squeeze BATCH only
        if prob.ndim != 3 or prob.shape[0] != 4:
            raise RuntimeError('predictor must return 4HW, got %s' %
                               (prob.shape,))
        ori = tuple(meta[0]['ori_shape'][:2])
        if prob.shape[1:] != ori:
            raise RuntimeError('prob shape %s != ori_shape %s' %
                               (prob.shape[1:], ori))
        if not np.isfinite(prob).all() or prob.min() < 0.0 or prob.max() > 1.0:
            raise RuntimeError('predictor output not finite prob in [0,1]')
        gt = np.asarray(gt_maps[i]).astype(np.int64)
        if gt.shape != ori:
            gt = cv2.resize(gt.astype(np.uint8), (ori[1], ori[0]),
                            interpolation=cv2.INTER_NEAREST).astype(np.int64)
        if not np.all(np.isin(gt, (0, 1, 2, 3, 4, 255))):
            raise RuntimeError('gt contains invalid labels outside 0..4/255')
        cnt = per_image_counts(prob, gt)
        name = os.path.splitext(os.path.basename(meta[0]['ori_filename']))[0]
        per_image[name] = {CLASSES[c]: cnt[c] for c in range(4)}
        for c in range(4):
            for j in range(3):
                total[c][j] += cnt[c][j]
        if (i + 1) % 25 == 0:
            log('eval %s %d/%d (%.1fs)' % (split, i + 1, n, time.time() - t0))
    miou, per = global_miou([tuple(x) for x in total])
    if dump_counts:
        d = os.path.dirname(dump_counts)
        if d:
            os.makedirs(d, exist_ok=True)
        atomic_json(dump_counts, {'split': split, 'miou': miou,
                                  'per_class': per, 'global_counts': total,
                                  'per_image': per_image})
    return miou, per, per_image


# ---------------------------------------------------------------------------
# champion ensemble with the MA correction head
# ---------------------------------------------------------------------------
class CorrectionHead(nn.Module):
    """7 -> 16 -> 16 -> 1 conv head, output scaled by 6*tanh."""
    def __init__(self):
        super(CorrectionHead, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(7, 16, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1))

    def forward(self, f):
        return 6.0 * torch.tanh(self.net(f))


def load_flat_head(head_state, device='cpu'):
    """Load the ONLY correction head (MA) from h4_best.ck['head'], a flat
    state_dict with keys net.0.weight, net.0.bias, ...  Fail closed on
    anything else."""
    if not isinstance(head_state, dict) or 'net.0.weight' not in head_state:
        raise RuntimeError('champion head file: expected flat state dict '
                           'with net.* keys')
    h = CorrectionHead().to(device)
    h.load_state_dict(head_state, strict=True)
    h.eval()
    for p in h.parameters():
        p.requires_grad_(False)
    return h


def _logit(p, eps=1e-4):
    q = p.clamp(eps, 1 - eps)
    return torch.log(q / (1 - q))


def member_prob(model, img, meta, flip, device):
    """Exact member inference via model(return_loss=False, rescale=True).
    For flip: pass the flipped image and flip flag in img_metas -- the
    mmseg test pipeline already un-flips the output, so the returned
    probability is NOT flipped a second time here.

    Fork output structure (verified against GLM's eval_ens.py forward_prob
    and h5_ensemble_corrector.py, both of which produced the reference
    numbers): forward -> forward_test -> simple_test returns, on the
    compute_aupr=True path, a LIST [(prob(4,H,W) ndarray, use_sigmoid,
    compute_aupr)].  GLM unwraps ``m(...)[0]`` then ``seg[0]`` when the
    remainder is a tuple/list.  Mirror that EXACTLY -- single-level unwrap
    leaves the (prob, bool, bool) tuple and np.asarray dies with an
    inhomogeneous-shape error (observed in server smoke gate 4)."""
    meta_f = copy.deepcopy(meta)
    x = img
    if flip:
        meta_f[0]['flip'] = True
        meta_f[0]['flip_direction'] = 'horizontal'
        x = img.flip(-1)
    out = model(return_loss=False, rescale=True, img=[x], img_metas=[meta_f])
    out = out[0] if isinstance(out, (tuple, list)) else out
    if isinstance(out, (tuple, list)):
        out = out[0]
    p = torch.from_numpy(np.asarray(out, dtype=np.float32)).to(device)
    if p.dim() != 3 or p.shape[0] != 4:
        raise RuntimeError('member prob must be (4,H,W), got %r' %
                           (tuple(p.shape),))
    return p


class ChampionEnsemble(object):
    """EXACT champion: 4 named members D,E,B,E4 with weights 3,2,1,1,
    horizontal-flip TTA for the ensemble average, per-class MA correction
    head only (no HE/SE/MA head-set requirement), driven by ps/pr from the
    same D member without flip."""

    MEMBER_WEIGHTS = {'D': 3.0, 'E': 2.0, 'B': 1.0, 'E4': 1.0}

    def __init__(self, settings, champion_cfg, device='cuda'):
        from mmseg.models import build_segmentor
        self.device = device
        repo = settings['repo']
        ref_abs = resolve_repo_path(repo, settings['reference_checkpoint'])
        members = champion_cfg['members']
        if len(members) != 4:
            raise RuntimeError('champion must have exactly 4 members')
        names = sorted(str(s['name']) for s in members)
        if names != sorted(self.MEMBER_WEIGHTS):
            raise RuntimeError('champion members must be D,E,B,E4; got %s' %
                               names)
        self.members = []
        self.d_model = None
        for spec in members:
            name = spec['name']
            c = load_config(repo, spec['config'])
            # explicit source_checkpoint: settings reference path (B) unless
            # the spec explicitly provides one (e.g. E4 base read source).
            mtype = str(c.model.get('type', ''))
            if 'source_checkpoint' in c.model or mtype == 'HRPlusRefinement':
                src = spec.get('source_checkpoint')
                c.model['source_checkpoint'] = (
                    resolve_repo_path(repo, src) if src else ref_abs)
            elif 'source_checkpoint' in spec:
                raise RuntimeError(
                    'champion member %s: source_checkpoint given but config '
                    'model has no such field (not HRPlusRefinement)' % name)
            m = build_segmentor(c.model)
            ckpt_abs = resolve_repo_path(repo, spec['checkpoint'])
            mode = spec.get('load', 'full')
            if mode == 'student_only':
                # HRPlusRefinement-family member (e.g. champion D / E):
                # load the member's student weights and take the reference
                # from checkpoint B (cross-checked against any embedded
                # reference inside the member checkpoint).  This reproduces
                # the historical h5 state exactly: h5 built each member with
                # source_checkpoint=B and load_checkpoint(strict=False),
                # which overwrote the student with the member weights and
                # left/overwrote the reference with the stored (== B) one.
                strict_initialize(m, ckpt_abs, ref_abs)
            elif mode == 'full':
                # Non-HRPlus member (e.g. champion B on hr100_b.py and E4 on
                # e4_detail_split613.py): strict COMPLETE load with no
                # reference injection; any key mismatch fails closed.
                strict_initialize(m, ckpt_abs, None)
            else:
                raise RuntimeError('unknown load mode %r for %s' %
                                   (mode, name))
            m = m.to(device).eval()
            for p in m.parameters():
                p.requires_grad_(False)
            self.members.append((m, self.MEMBER_WEIGHTS[name], spec))
            if name == 'D':
                self.d_model = m
        if self.d_model is None:
            raise RuntimeError('champion requires a D member for ps/pr')
        # exactly one head: MA, loaded from h4_best.ck['head'] flat state.
        head_abs = resolve_repo_path(repo, champion_cfg['head'])
        try:
            ck = torch.load(head_abs, map_location='cpu',
                            weights_only=False)
        except TypeError:
            ck = torch.load(head_abs, map_location='cpu')
        head_state = ck['head'] if isinstance(ck, dict) and 'head' in ck else ck
        self.heads = {'MA': load_flat_head(head_state, device)}

    def pre_head_probs(self, img, meta):
        """(pf, ps, pr) each (4,H,W) at ori resolution -- the EXACT tensors
        predict() builds BEFORE applying any correction head.  Single source
        of truth for the correction-head feature construction (round-6
        caches these during training-feature extraction, so training and
        inference provably use the same numbers)."""
        with torch.no_grad():
            acc = None
            tot = 0.0
            for m, w, spec in self.members:
                p = member_prob(m, img, meta, False, self.device)
                p2 = member_prob(m, img, meta, True, self.device)
                acc = (p + p2) * w if acc is None else acc + (p + p2) * w
                tot += 2.0 * w
            pf = acc / tot
            # same D member for ps/pr, no flip
            ps = self._raw_prob(self.d_model, img, meta, 'student')
            pr = self._raw_prob(self.d_model, img, meta, 'reference')
        return pf, ps, pr

    def corrected_class_prob(self, c, pf, ps, pr, img_f, head):
        """Corrected probability for class index c: sigmoid(logit(pf_c) +
        head(f7)) with f7 = [img_f, logit(pf_c), logit(ps_c), logit(pr_c),
        ps_c - pr_c].  The ONLY place f7 is built (predict() calls it too).

        A head may declare ``r6_round_inputs = True`` to have its inputs
        rounded to fp16 first (round-6 heads are trained on fp16-stored
        features, so inference must round identically).  Heads without the
        flag (e.g. the champion's MA head) keep the raw fp32 path, so the
        verified champion numbers are untouched."""
        if getattr(head, 'r6_round_inputs', False):
            img_f = img_f.half().float()
            pf = pf.half().float()
            ps = ps.half().float()
            pr = pr.half().float()
        f7 = torch.cat([img_f, _logit(pf[c:c + 1]), _logit(ps[c:c + 1]),
                        _logit(pr[c:c + 1]), ps[c:c + 1] - pr[c:c + 1]],
                       dim=0).unsqueeze(0)
        return torch.sigmoid(_logit(pf[c:c + 1][0]) + head(f7)[0, 0])

    def predict(self, img, meta):
        """Returns (4,H,W) corrected champion probability at ori resolution."""
        pf, ps, pr = self.pre_head_probs(img, meta)
        with torch.no_grad():
            ori = tuple(meta[0]['ori_shape'][:2])
            img_f = F.interpolate(img, size=ori, mode='bilinear',
                                  align_corners=False)[0]
            chans = []
            for c in range(4):
                cname = CLASSES[c]
                if cname in self.heads:
                    chans.append(self.corrected_class_prob(
                        c, pf, ps, pr, img_f, self.heads[cname]
                    ).unsqueeze(0))
                else:
                    chans.append(pf[c:c + 1])
            return torch.cat(chans, dim=0)

    def _raw_prob(self, m, img, meta, which):
        fn = getattr(m, '_' + which + '_prob', None)
        if fn is None:
            raise RuntimeError('D member lacks _%s_prob' % which)
        return fn(img, meta, True)[0].to(self.device)


def mix_probs(p_champ, p_new):
    return 0.5 * p_champ + 0.5 * p_new


# ---------------------------------------------------------------------------
# RNG state save/restore
# ---------------------------------------------------------------------------
def get_rng_state():
    return {'py': random.getstate(), 'np': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all()}


def set_rng_state(st):
    random.setstate(st['py'])
    np.random.set_state(st['np'])
    torch.set_rng_state(st['torch'].cpu())
    torch.cuda.set_rng_state_all([s.cpu() for s in st['cuda']])


# ---------------------------------------------------------------------------
# dataset split audit
# ---------------------------------------------------------------------------
def audit_split(data_root, expected=(454, 76, 227)):
    """Verify per-split image counts, no dup/intersection, txt IDs agree
    with image dirs AND png masks directly, and return the real assignment
    dict filename -> {'split', 'stem'} covering the full ID set."""
    assignment = {}
    seen_stems = set()
    for si, split in enumerate(('train', 'val', 'test')):
        img_dir = os.path.join(data_root, 'images', split)
        lab_dir = os.path.join(data_root, 'labels', split)
        list_file = os.path.join(data_root, split + '.txt')
        imgs = sorted(f for f in os.listdir(img_dir)
                      if f.lower().endswith(('.jpg', '.jpeg')))
        if len(imgs) != expected[si]:
            raise RuntimeError('split %s has %d images, expected %d' %
                               (split, len(imgs), expected[si]))
        if os.path.isfile(list_file):
            listed = [l.strip() for l in open(list_file, encoding='utf-8')
                      if l.strip()]
            if sorted(os.path.splitext(x)[0] for x in listed) != \
                    sorted(os.path.splitext(x)[0] for x in imgs):
                raise RuntimeError('%s does not match %s images' %
                                   (list_file, split))
        # no split list file: img_dir listing is the source of truth (the
        # baseline configs enumerate img_dir directly); counts and overlap
        # checks below still apply.
        stems = set()
        for f in imgs:
            stem = os.path.splitext(f)[0]
            if stem in seen_stems:
                raise RuntimeError('duplicate stem: %s' % stem)
            seen_stems.add(stem)
            stems.add(stem)
            assignment[f] = {'split': split, 'stem': stem}
        labs = set(os.path.splitext(x)[0] for x in os.listdir(lab_dir)
                   if x.endswith('.png'))
        if labs != stems:
            raise RuntimeError('labels/%s mismatch' % split)
    source = os.path.join(data_root, 'split_assignment.json')
    if os.path.isfile(source):
        with open(source, encoding='utf-8') as f:
            declared = json.load(f)
        declared = declared.get('images', declared)
        if set(declared) != set(assignment) or any(declared[k]['split'] != v['split'] for k, v in assignment.items()):
            raise RuntimeError('split assignment differs from actual image lists')
    return assignment


def load_image_label(data_root, split, stem):
    if split != 'train':
        raise ValueError('Training image/label loader is train-only')
    folder = os.path.join(data_root, 'images', split)
    matches = [os.path.join(folder, f) for f in os.listdir(folder)
               if os.path.splitext(f)[0] == stem and f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if len(matches) != 1:
        raise ValueError('Expected exactly one image for '+stem)
    img = cv2.imread(matches[0], cv2.IMREAD_COLOR)
    lab = cv2.imread(os.path.join(data_root, 'labels', split, stem+'.png'), cv2.IMREAD_GRAYSCALE)
    if img is None or lab is None or img.shape[:2] != lab.shape or not np.isin(lab, [0,1,2,3,4,255]).all():
        raise ValueError('Invalid training image/label pair '+stem)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB), lab


# ---------------------------------------------------------------------------
# deadline
# ---------------------------------------------------------------------------
def parse_deadline(raw):
    """Accepts ISO-8601 string or epoch seconds; anything else rejected."""
    if isinstance(raw, (int, float)):
        return float(raw)
    if isinstance(raw, str):
        try:
            from datetime import datetime
            d = datetime.fromisoformat(raw.replace('Z', '+00:00'))
            if d.tzinfo is None:
                d = d.astimezone()
            return d.timestamp()
        except ValueError:
            pass
    raise ValueError('unknown deadline format (need ISO-8601 or epoch): %r'
                     % raw)
