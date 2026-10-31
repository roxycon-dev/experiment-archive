#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""Round 5: ensemble expansion (E0) + fusion-supervision variants (E1/E2)
+ final ensemble lock (E3).  See NEXT_ROUND_PLAN_20260918.md.

Design invariants:
  - Reuses run_protocol helpers verbatim (train_arm with loss_fn/val_points,
    build_student, resume, checkpoints, evaluate_split) -- no copied loops.
  - Every phase is idempotent on restart via round5_state.json.
  - All gates are the pre-registered ones (G0/G1/G3); no post-hoc edits.
  - The ledger deadline (runs/r5/budget_ledger.json) is the money-critical
    hard stop; guard EXIT trap + watchdog handle backup + shutdown.

Usage:
  python3 run_round5.py --settings settings.json --run-dir runs/r5
  python3 run_round5.py --settings settings.json --run-dir runs/r5 --smoke
"""
import argparse
import hashlib
import json
import os
import sys
import time

import torch

import ops
import run_protocol as rp
import fusion_core as fc

T5 = 3000                       # pre-registered training length
VAL_POINTS = [1000, 2000, 3000]
G0_MARGIN = 0.002               # E0 gate: best >= champion + margin
G1_FLOOR = 0.4185               # E1 gate: variant max val floor
G3_MARGIN = 0.002               # E3 lock gate
BNEW_CKPT = {                   # round-4 fusion-supervised student (data disk)
    3000: '/root/autodl-tmp/hrplus/glm/runs/r1/ckpt_B_s42/iter_3000.pth',
    4000: '/root/autodl-tmp/hrplus/glm/runs/r1/ckpt_B_s42/iter_4000.pth',
}
E0_CONFIGS = [('bnew3000_w1', 3000, 1.0),
              ('bnew3000_w2', 3000, 2.0),
              ('bnew4000_w1', 4000, 1.0)]
E3_RESERVE = 1.5 * 3600         # keep for final scan + test
ARM_SECONDS = 70 * 60           # one T=3000 arm incl. 3 vals (r1-measured)


def loss_pair_w(logits, label, ref, w):
    """Weighted fusion loss: (1-w)*Dice(p) + w*Dice(0.5p+0.5ref).
    w=0.5 reproduces rt.loss_pair(..., 'B') EXACTLY; w=0 -> arm A; w=1 ->
    pure fused supervision (V1).  ref validated and detached."""
    probs = torch.sigmoid(logits)
    ref = torch.as_tensor(ref)
    fc._validate_ref(ref, probs)
    ref = ref.detach()
    if w <= 0.0:
        return fc.dice_prob(probs, label)
    fused = 0.5 * probs + 0.5 * ref
    if w >= 1.0:
        return fc.dice_prob(fused, label)
    return (1.0 - w) * fc.dice_prob(probs, label) + w * fc.dice_prob(fused,
                                                                     label)


def w_v1(_it):
    return 1.0


def w_v2(it):
    """Curriculum: student-only -> 0.5 -> pure fused (by train iter)."""
    if it < 1000:
        return 0.0
    if it < 2000:
        return 0.5
    return 1.0


class State(object):
    """round5_state.json: append-only results, idempotent phases."""

    def __init__(self, run_dir):
        self.path = os.path.join(run_dir, 'round5_state.json')
        if os.path.isfile(self.path):
            with open(self.path, 'r', encoding='utf-8') as f:
                self.d = json.load(f)
        else:
            self.d = {}

    def save(self):
        rp.rt_like_atomic(self.path, self.d)


def deviation(run_dir, kind, detail):
    entry = {'time': time.strftime('%Y-%m-%dT%H:%M:%S'), 'kind': kind,
             'detail': detail}
    with open(os.path.join(run_dir, 'DEVIATIONS.jsonl'), 'a',
              encoding='utf-8') as f:
        f.write(json.dumps(entry, sort_keys=True) + '\n')
    print('[DEVIATION] %s %s' % (kind, json.dumps(detail, sort_keys=True)),
          flush=True)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def build_extra_member(settings, ck_abs, weight):
    """5th champion member from a round-4 FULL checkpoint payload.
    Construction path mirrors ChampionEnsemble's 'student_only' member
    build exactly (same config family, source_checkpoint=B, strict load
    with embedded-reference cross-check against B)."""
    import runtime as rt
    from mmseg.models import build_segmentor
    repo = settings['repo']
    ref_abs = rt.resolve_repo_path(repo, settings['reference_checkpoint'])
    c = rt.load_config(repo, settings['config'])
    c.model['source_checkpoint'] = ref_abs
    m = build_segmentor(c.model)
    rt.strict_initialize(m, ck_abs, ref_abs)   # raises if ref != B
    m = m.to('cuda').eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return (m, float(weight), {'name': 'BNEW', 'checkpoint': ck_abs,
                               'load': 'student_only',
                               'weight': float(weight)})


def eval_ens(ens, split, settings, log, deadline, dump_counts=None):
    return rp.evaluate_split_local(lambda i, m: ens.predict(i, m), split,
                                   settings, log, deadline,
                                   dump_counts=dump_counts)


def with_extra(base_ens, base_members, extra):
    base_ens.members = base_members + [extra]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--settings', required=True)
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()

    settings = rp.validate_settings(args.settings)
    if settings['_pending']:
        raise RuntimeError('missing assets: %s' % settings['_pending'])
    os.makedirs(args.run_dir, exist_ok=True)
    log = rp.Log(args.run_dir)

    ledger = ops.load_ledger(args.run_dir)
    global_deadline = ops.deadline_of(ledger).timestamp()
    log('round5 global deadline: %s (%.2fh remaining)' %
        (ops.deadline_of(ledger).isoformat(),
         (global_deadline - time.time()) / 3600.0))

    import runtime as rt
    rt.inject_repo(settings['repo'])
    settings['_cfg'] = rt.load_config(settings['repo'], settings['config'])
    settings['_cfg'].model['source_checkpoint'] = \
        settings['_abs']['reference_checkpoint']

    # durable prep reuse: reference cache from round 4 (data disk).
    # LATENT-BUG GUARD (found 09-18): the fork's segmentor constructor
    # pops scale_ratio/fuse_mode/crop_num from cfg.model.hr_settings IN
    # PLACE, and the r1 manifest digest was computed AFTER build_student
    # mutated cfg in the same process -- so a fresh process must run
    # build_student once BEFORE cache.complete() to reach the same cfg
    # state (any r1 restart would have hit this too).
    _probe, _ = rp.build_student(settings)   # mutates settings['_cfg']
    del _probe
    torch.cuda.empty_cache()
    cache = rt.ReferenceCache(settings['cache_dir'], settings)
    if not cache.complete():
        raise RuntimeError('reference cache incomplete; round 5 assumes the '
                           'round-4 cache is durable')
    data_root = settings['data_root']
    assignment = rt.audit_split(data_root)
    log('split audit OK (454/76/227 expected)')
    rp.audit_labels(data_root, 'train',
                   [f for f, v in assignment.items() if v['split'] == 'train'],
                   log)
    split_hashes = {
        'config': settings['_hashes']['config'],
        'split': hashlib.sha256(
            json.dumps(assignment, sort_keys=True).encode()).hexdigest(),
        'assets': settings['_hashes'],
        'code': rp.config_fingerprint(settings)}
    # BNEW checkpoints are outside the settings hash regime: record them
    st = State(args.run_dir)
    if 'bnew_sha' not in st.d:
        st.d['bnew_sha'] = {str(k): sha256(v) for k, v in BNEW_CKPT.items()}
        st.save()
        for k, v in st.d['bnew_sha'].items():
            log('bnew ckpt iter%s sha256 %s' % (k, v[:16]))

    # ------------------------------------------------------------------
    if args.smoke:
        smoke(settings, st, cache, log, global_deadline, split_hashes)
        return

    # ================= E0: zero-training ensemble probe =================
    if 'e0' not in st.d:
        e0_deadline = global_deadline - E3_RESERVE - 2 * ARM_SECONDS
        base_ens = rt.ChampionEnsemble(settings, settings['champion'])
        base_members = list(base_ens.members)
        t0 = time.time()
        champ_val, _, _ = eval_ens(base_ens, 'val', settings, log, e0_deadline,
                                   dump_counts=os.path.join(
                                       args.run_dir, 'e0_champ_val_counts.json'))
        log('E0 baseline champion val %.4f (%.0fs; r1 reference 0.4267)' %
            (champ_val, time.time() - t0))
        e0 = {'champ_val': champ_val, 'configs': {}}
        for name, it, w in E0_CONFIGS:
            extra = build_extra_member(settings, BNEW_CKPT[it], w)
            with_extra(base_ens, base_members, extra)
            t0 = time.time()
            miou, _, _ = eval_ens(base_ens, 'val', settings, log, e0_deadline,
                                  dump_counts=os.path.join(
                                      args.run_dir,
                                      'e0_%s_counts.json' % name))
            e0['configs'][name] = miou
            log('E0 %s (bnew@%d w=%.1f) val %.4f (%.0fs)' %
                (name, it, w, miou, time.time() - t0))
            base_ens.members = base_members          # detach extra
            del extra
            torch.cuda.empty_cache()
        best_name = max(e0['configs'], key=e0['configs'].get)
        e0['best'] = best_name
        e0['best_val'] = e0['configs'][best_name]
        e0['g0_pass'] = e0['best_val'] >= champ_val + G0_MARGIN
        st.d['e0'] = e0
        st.save()
        log('E0 best=%s %.4f (champ %.4f) G0 pass=%s' %
            (best_name, e0['best_val'], champ_val, e0['g0_pass']))
        if e0['g0_pass']:
            _it = 3000 if '3000' in best_name else 4000
            _w = 2.0 if best_name.endswith('w2') else 1.0
            extra = build_extra_member(settings, BNEW_CKPT[_it], _w)
            with_extra(base_ens, base_members, extra)
            t_val = e0['best_val']
            t0 = time.time()
            t_miou, _, _ = eval_ens(base_ens, 'test', settings, log,
                                    global_deadline,
                                    dump_counts=os.path.join(
                                        args.run_dir,
                                        'e0_test_counts.json'))
            e0['test'] = t_miou
            st.save()
            log('E0 test confirm: %.4f vs champion test 0.4421 (%.0fs)' %
                (t_miou, time.time() - t0))
            base_ens.members = base_members
            del extra
            torch.cuda.empty_cache()
        del base_ens
        torch.cuda.empty_cache()
    e0 = st.d['e0']
    champ_val = e0['champ_val']

    # ================= E1: fusion-supervision variants ==================
    if 'e1' not in st.d:
        e1_deadline = global_deadline - E3_RESERVE
        remaining = e1_deadline - time.time()
        if remaining < 2 * ARM_SECONDS + 20 * 60:
            deviation(args.run_dir, 'e1_reduced',
                      {'reason': 'budget', 'remaining_h': remaining / 3600.0,
                       'action': 'single variant V2 only'})
            variants = [('V2', w_v2)]
        else:
            variants = [('V1', w_v1), ('V2', w_v2)]
        e1 = {}
        for name, w_fn in variants:
            loss_fn = (lambda lg, lb, rb, it, _w=w_fn:
                       loss_pair_w(lg, lb, rb, _w(it)))
            best, hist = rp.train_arm(settings, args.run_dir, name, 42, T5,
                                      e1_deadline, log, cache, split_hashes,
                                      loss_fn=loss_fn,
                                      val_points=VAL_POINTS)
            e1[name] = {'best_step': best[0], 'best_miou': best[1],
                        'hist': hist}
            st.d['e1'] = e1
            st.save()
        winner = max(e1, key=lambda k: e1[k]['best_miou'])
        e1['winner'] = winner
        e1['g1_pass'] = e1[winner]['best_miou'] >= G1_FLOOR
        st.save()
        log('E1 winner=%s (best %.4f @%d) G1 pass=%s' %
            (winner, e1[winner]['best_miou'], e1[winner]['best_step'],
             e1['g1_pass']))
    e1 = st.d['e1']

    # ================= E2: opportunistic seed43 direction check ========
    if 'e2' not in st.d:
        e2_need = ARM_SECONDS + 20 * 60
        if global_deadline - time.time() >= e2_need + E3_RESERVE:
            winner = e1['winner']
            w_fn = {'V1': w_v1, 'V2': w_v2}[winner]
            loss_fn = (lambda lg, lb, rb, it, _w=w_fn:
                       loss_pair_w(lg, lb, rb, _w(it)))
            best, hist = rp.train_arm(settings, args.run_dir, winner, 43,
                                      T5, global_deadline - E3_RESERVE, log,
                                      cache, split_hashes, loss_fn=loss_fn,
                                      val_points=VAL_POINTS)
            st.d['e2'] = {'arm': winner, 'seed': 43,
                          'best_step': best[0], 'best_miou': best[1],
                          'hist': hist}
            st.save()
            log('E2 %s s43: best %.4f @%d (s42-A reference 0.4167)' %
                (winner, best[1], best[0]))
        else:
            st.d['e2'] = {'skipped': 'budget'}
            st.save()
            log('E2 skipped (budget)')

    # ================= E3: final ensemble scan + lock ===================
    if 'e3' not in st.d:
        e3 = {'configs': {}}
        base_ens = rt.ChampionEnsemble(settings, settings['champion'])
        base_members = list(base_ens.members)
        cands = []
        if e0['g0_pass']:
            _it = 3000 if '3000' in e0['best'] else 4000
            _w = 2.0 if e0['best'].endswith('w2') else 1.0
            cands.append(('e0_best', BNEW_CKPT[_it], _w))
        if e1['g1_pass']:
            ck = os.path.join(args.run_dir, 'ckpt_%s_s42' % e1['winner'],
                              'iter_%d.pth' % e1[e1['winner']]['best_step'])
            cands.append(('e1_w1', ck, 1.0))
            cands.append(('e1_w2', ck, 2.0))
            if e0['g0_pass']:
                cands.append(('e0_plus_e1', ck, 1.0))  # with e0 extra too
        for name, ck, w in cands:
            extra = build_extra_member(settings, ck, w)
            with_extra(base_ens, base_members, extra)
            if name == 'e0_plus_e1':
                _it = 3000 if '3000' in e0['best'] else 4000
                _w0 = 2.0 if e0['best'].endswith('w2') else 1.0
                extra2 = build_extra_member(settings, BNEW_CKPT[_it], _w0)
                base_ens.members = base_members + [extra, extra2]
            t0 = time.time()
            miou, _, _ = eval_ens(base_ens, 'val', settings, log,
                                  global_deadline)
            e3['configs'][name] = miou
            log('E3 %s val %.4f (%.0fs)' % (name, miou, time.time() - t0))
            st.d['e3'] = e3
            st.save()
            base_ens.members = base_members
            del extra
            torch.cuda.empty_cache()
        best_name = max(e3['configs'], key=e3['configs'].get) \
            if e3['configs'] else None
        e3['best'] = best_name
        e3['best_val'] = e3['configs'].get(best_name)
        e3['g3_pass'] = bool(best_name and
                             e3['best_val'] >= champ_val + G3_MARGIN)
        st.d['e3'] = e3
        st.save()
        log('E3 best=%s %.4f G3 pass=%s' %
            (best_name, e3.get('best_val'), e3['g3_pass']))

        if e3['g3_pass']:
            # rebuild the winning ensemble and TEST-confirm (val locked it)
            spec = {n: (c, w) for n, c, w in cands}[best_name]
            extra = build_extra_member(settings, spec[0], spec[1])
            with_extra(base_ens, base_members, extra)
            if best_name == 'e0_plus_e1':
                _it = 3000 if '3000' in e0['best'] else 4000
                _w0 = 2.0 if e0['best'].endswith('w2') else 1.0
                extra2 = build_extra_member(settings, BNEW_CKPT[_it], _w0)
                base_ens.members = base_members + [extra, extra2]
            t0 = time.time()
            t_miou, _, _ = eval_ens(base_ens, 'test', settings, log,
                                    global_deadline,
                                    dump_counts=os.path.join(
                                        args.run_dir, 'e3_test_counts.json'))
            e3['test'] = t_miou
            st.save()
            log('E3 locked=%s test %.4f (champion test 0.4421) (%.0fs)' %
                (best_name, t_miou, time.time() - t0))
        del base_ens
        torch.cuda.empty_cache()

    # ================= summary ==========================================
    e3 = st.d['e3']
    summary = {
        'round': 5,
        'champion_val': champ_val,
        'e0': {'best': e0['best'], 'best_val': e0['best_val'],
               'g0_pass': e0['g0_pass'], 'configs': e0['configs'],
               'test': e0.get('test')},
        'e1': e1, 'e2': st.d.get('e2'),
        'e3': {'best': e3['best'], 'best_val': e3.get('best_val'),
               'g3_pass': e3['g3_pass'], 'configs': e3['configs'],
               'test': e3.get('test')},
        'target': 47.20,
        'status': ('locked' if e3.get('test') is not None else
                   ('e0_confirmed' if e0.get('test') is not None else
                    'no_gain')),
        'created': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }
    rp.rt_like_atomic(os.path.join(args.run_dir, 'round5_summary.json'),
                      summary)
    log('ROUND5 COMPLETE status=%s' % summary['status'])


def smoke(settings, st, cache, log, global_deadline, split_hashes):
    """Fast GPU sanity: champion+BNEW predict, one V1/V2 loss step."""
    import runtime as rt
    log('smoke: build base champion')
    base_ens = rt.ChampionEnsemble(settings, settings['champion'])
    base_members = list(base_ens.members)
    extra = build_extra_member(settings, BNEW_CKPT[3000], 1.0)
    with_extra(base_ens, base_members, extra)
    # predict ONE val image (dataset built EXACTLY like evaluate_split)
    import copy
    import numpy as np
    from mmseg.datasets import build_dataset
    d = copy.deepcopy(settings['_cfg'].data['val'])
    d.test_mode = True
    d.img_dir = os.path.join(settings['data_root'], 'images', 'val')
    d.ann_dir = os.path.join(settings['data_root'], 'labels', 'val')
    ds = build_dataset(d)
    s = ds[0]
    img = rt._unwrap_img(s['img']).unsqueeze(0).cuda()
    meta = rt._unwrap_meta(s['img_metas'])
    with torch.no_grad():
        p = base_ens.predict(img, meta)
    assert tuple(p.shape) == (4,) + tuple(meta[0]['ori_shape'][:2]), \
        'champion predict shape %r' % (tuple(p.shape),)
    assert bool(torch.isfinite(p).all()) and float(p.min()) >= 0 \
        and float(p.max()) <= 1
    log('smoke gate1: 5-member predict OK %r finite in [0,1]' %
        (tuple(p.shape),))
    del base_ens, extra
    torch.cuda.empty_cache()

    # one training step with V1 and V2 losses (finite + updates student)
    model, _ = rp.build_student(settings)
    model.train()
    rt.freeze_bn(model)
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                          lr=5e-5, momentum=0.9, weight_decay=0.0005)
    rng = np.random.RandomState(42)
    files = rp.list_split(settings['data_root'], 'train')
    stems = [os.path.splitext(f)[0] for f in files]
    for tag, w_fn in (('V1', w_v1), ('V2', w_v2)):
        idxs = rng.permutation(len(stems))[:2]
        imgs, labs, refs = [], [], []
        for i in idxs:
            img, lab = rt.load_image_label(settings['data_root'], 'train',
                                           stems[i])
            im2, la2, rf2, _ = rt.paired_geometry(
                img, lab, cache[stems[i]], rng)
            imgs.append(torch.from_numpy(im2))
            labs.append(torch.from_numpy(la2))
            refs.append(torch.from_numpy(rf2))
        opt.zero_grad()
        logits = model.encode_decode(torch.stack(imgs).cuda(), [{}])
        loss = loss_pair_w(logits, torch.stack(labs).cuda(),
                           torch.stack(refs).cuda(), w_fn(2500))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 5.0)
        opt.step()
        rt.freeze_bn(model)
        assert float(loss) == float(loss) and abs(float(loss)) < 10.0, \
            'loss not finite/reasonable: %r' % float(loss)
        log('smoke gate2 %s: one step OK (w(2500)=%.1f loss %.4f)' %
            (tag, w_fn(2500), float(loss)))
    del model, opt
    torch.cuda.empty_cache()
    log('SMOKE OK (5 gates: settings/ledger/cache implied, predict, V1, V2)')


if __name__ == '__main__':
    try:
        main()
    except rp.ShutdownGraceful as e:
        print('[shutdown-guard] %s' % e, flush=True)
        sys.exit(0)
