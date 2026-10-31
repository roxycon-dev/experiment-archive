# -*- coding: utf-8 -*-
"""Round-6 P1 feature cache (memory-frugal, parity-exact).

Per image and class we store TWO files at ORIGINAL resolution, both fp16:
    <stem>.img.npy : float16 (3,H,W) -- img_f (the bilinear-upsampled
                     *normalised* image ChampionEnsemble.predict feeds the
                     correction head)
    <stem>.f.npy   : float16 (3,H,W) -- [pf_c, ps_c, pr_c]

The ONLY train/inference parity mechanism is fp16 rounding of the four
input groups (img + 3 probability channels): training rounds the stored
values, and inference rounds the live fp32 tensors identically
(runtime.ChampionEnsemble.corrected_class_prob applies the declared
``r6_round_inputs`` flag; heads without the flag keep the raw fp32 path).

Budget: 30.2MB/image -> ~13.7GB for the 454 train images and ~2.3GB for the
76 val images, one class at a time.

Resumable: both files are validated (header + finite sample) and skipped on
restart; manifest + COMPLETE marker fail closed on identity mismatch.
"""
import os
import time

import numpy as np
import torch

import r6_lib
import runtime as rt

IMG_CH = 3
FIELD_CH = 3                # pf_c, ps_c, pr_c


def cache_dir(run_dir, cls, split):
    return os.path.join(run_dir, 'featcache_r6', cls, split)


def _manifest_path(d):
    return os.path.join(d, 'manifest.json')


def _complete_path(d):
    return os.path.join(d, 'COMPLETE')


def _digest(settings):
    import hashlib
    return hashlib.sha256(
        settings['_cfg'].dump().encode('utf-8')).hexdigest()


def _entry_valid(ip, fp):
    try:
        a = np.load(ip, mmap_mode='r')
        b = np.load(fp, mmap_mode='r')
        if a.dtype != np.float16 or a.ndim != 3 or a.shape[0] != IMG_CH:
            return False
        if b.dtype != np.float16 or b.ndim != 3 or b.shape[0] != FIELD_CH:
            return False
        if tuple(a.shape[1:]) != tuple(b.shape[1:]):
            return False
        sample = np.asarray(b[:, ::64, ::64], dtype=np.float32)
        return bool(np.isfinite(sample).all())
    except Exception:
        return False


def cache_complete(run_dir, cls, split, expected_count, cfg_digest, ref_sha):
    d = cache_dir(run_dir, cls, split)
    if not (os.path.isfile(_manifest_path(d)) and
            os.path.isfile(_complete_path(d))):
        return False
    import json
    with open(_manifest_path(d), 'r', encoding='utf-8') as f:
        man = json.load(f)
    if not (man.get('cls') == cls and man.get('split') == split
            and man.get('count') == expected_count
            and man.get('cfg_digest') == cfg_digest
            and man.get('ref_sha') == ref_sha):
        raise RuntimeError('feature cache present but inconsistent: %s' % d)
    return True


def cache_needed(run_dir, cls, split, expected_count, settings):
    """True if this cache must be (re)built -- the caller only has to reserve
    disk space in that case (a resume over an existing complete cache needs
    ZERO extra space, which is exactly what the 14:30 recovery turned on)."""
    return not cache_complete(run_dir, cls, split, expected_count,
                              _digest(settings),
                              settings['_hashes']['reference_checkpoint'])


def build_cache(settings, run_dir, cls, split, champion, log, deadline,
                expected_count, max_images=None):
    """Builds (or reuses) the per-class feature cache for one split."""
    import copy
    import json
    from mmseg.datasets import build_dataset
    c = rt.CLASSES.index(cls)
    d = cache_dir(run_dir, cls, split)
    cfg_digest = _digest(settings)
    ref_sha = settings['_hashes']['reference_checkpoint']
    if cache_complete(run_dir, cls, split, expected_count, cfg_digest,
                      ref_sha):
        log('feature cache present: %s (%d entries)' % (d, expected_count))
        return d
    os.makedirs(d, exist_ok=True)
    # ALWAYS the deterministic val/test pipeline template (Resize
    # keep_ratio=False to 2048 + Normalize), never cfg.data['train'] whose
    # augmenting pipeline both needs ann_info and would break inference
    # parity.  Identical to runtime.ReferenceCache.build.
    ds_cfg = copy.deepcopy(settings['_cfg'].data['val'])
    ds_cfg.test_mode = True
    ds_cfg.pop('split', None)
    ds_cfg.img_dir = os.path.join(settings['data_root'], 'images', split)
    ds_cfg.ann_dir = os.path.join(settings['data_root'], 'labels', split)
    ds = build_dataset(ds_cfg)
    n_full = len(ds)
    n = n_full if max_images is None else min(n_full, int(max_images))
    if n != expected_count:
        raise RuntimeError('%s has %d items (capped %s), expected %d'
                           % (split, n_full, max_images, expected_count))
    t0, entries = time.time(), {}
    for i in range(n):
        if time.time() > deadline:
            raise r6_lib.PhaseTimeout('feature cache %s/%s (phase budget)'
                                      % (cls, split))
        s = ds[i]
        img = rt._unwrap_img(s['img']).unsqueeze(0).cuda()
        meta = rt._unwrap_meta(s['img_metas'])
        stem = os.path.splitext(os.path.basename(
            meta[0]['ori_filename']))[0]
        ip = os.path.join(d, stem + '.img.npy')
        fp = os.path.join(d, stem + '.f.npy')
        if os.path.isfile(ip) and os.path.isfile(fp) and _entry_valid(ip, fp):
            entries[stem] = {'img': ip, 'f': fp,
                             'img_sha': r6_lib.sha256_file(ip),
                             'f_sha': r6_lib.sha256_file(fp)}
            continue                                  # resume: skip
        pf, ps, pr = champion.pre_head_probs(img, meta)
        ori = tuple(meta[0]['ori_shape'][:2])
        img_f = torch.nn.functional.interpolate(
            img, size=ori, mode='bilinear', align_corners=False)[0]
        if not torch.isfinite(img_f).all():
            raise RuntimeError('non-finite img_f at %s' % stem)
        fields = torch.cat([pf[c:c + 1], ps[c:c + 1], pr[c:c + 1]], dim=0)
        if fields.shape[0] != FIELD_CH or tuple(fields.shape[1:]) != ori:
            raise RuntimeError('bad field shape %r' % (tuple(fields.shape),))
        if not torch.isfinite(fields).all():
            raise RuntimeError('non-finite fields at %s' % stem)
        np.save(ip + '.tmp.npy', img_f.to(torch.float16).cpu().numpy())
        os.replace(ip + '.tmp.npy', ip)
        np.save(fp + '.tmp.npy', fields.to(torch.float16).cpu().numpy())
        os.replace(fp + '.tmp.npy', fp)
        entries[stem] = {'img': ip, 'f': fp,
                         'img_sha': r6_lib.sha256_file(ip),
                         'f_sha': r6_lib.sha256_file(fp)}
        if (i + 1) % 25 == 0:
            log('featcache %s/%s %d/%d (%.0fs)'
                % (cls, split, i + 1, n, time.time() - t0))
        del img, pf, ps, pr, img_f, fields
        torch.cuda.empty_cache()
    r6_lib.atomic_json(_manifest_path(d),
                       {'cls': cls, 'split': split, 'count': n,
                        'cfg_digest': cfg_digest, 'ref_sha': ref_sha,
                        'version': 'r6-feat-v3', 'entries': entries})
    with open(_complete_path(d), 'w') as f:
        f.write('ok')
    log('feature cache COMPLETE %s/%s: %d entries (%.0fs)'
        % (cls, split, n, time.time() - t0))
    return d


def load_entry(cache_dir_, stem):
    """(img_u8 (3,H,W), fields_f16 (3,H,W)) -- validated headers."""
    ip = os.path.join(cache_dir_, stem + '.img.npy')
    fp = os.path.join(cache_dir_, stem + '.f.npy')
    a = np.load(ip, mmap_mode='r')
    b = np.load(fp, mmap_mode='r')
    if a.dtype != np.float16 or a.shape[0] != IMG_CH:
        raise RuntimeError('bad img entry %s: %r %s' % (ip, a.shape, a.dtype))
    if b.dtype != np.float16 or b.shape[0] != FIELD_CH:
        raise RuntimeError('bad field entry %s: %r %s'
                           % (fp, b.shape, b.dtype))
    return a, b


def stems_of(cache_dir_):
    return sorted(f[:-8] for f in os.listdir(cache_dir_)
                  if f.endswith('.img.npy'))


def delete_cache(run_dir, cls, split, log):
    import shutil
    d = cache_dir(run_dir, cls, split)
    if os.path.isdir(d):
        size = sum(os.path.getsize(os.path.join(dp, f))
                   for dp, _, fs in os.walk(d) for f in fs)
        shutil.rmtree(d, ignore_errors=True)
        log('deleted feature cache %s (%.2fG)' % (d, size / 1024.0 ** 3))
