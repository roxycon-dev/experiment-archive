#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""Round-6 orchestrator (see NEXT_ROUND_PLAN_20260918_R6.md).

P0   champion baselines: val counts (4.5 min) AND test counts (~17 min) of
     the PRISTINE champion, plus the per-class error profile -> attack order.
     The pristine test counts are the ONLY valid baseline for the final
     paired bootstrap (they also independently reproduce test 0.4421).
P1   H6: per-class correction heads on cached champion features; gate
     (per class) val class-IoU delta >= +0.005 -> install into
     champion.heads (the predict loop is class-generic).
P1T  mid test-confirmation of the H6-only stack (optional, budget-checked).
P2   M1: native-res MA expert (patch plan -> U-Net -> sliding window ->
     alpha sweep on val); gate MA val IoU delta >= +0.010.
P3   final: ONE test pass with every adopted component; paired bootstrap
     against the PRISTINE champion counts; G3 = delta >= +0.002 with CI
     excluding zero.

PRE-REGISTERED gates (constants below, never edited at runtime); every
runtime deviation is appended to DEVIATIONS.jsonl.
"""
import argparse
import copy
import json
import os
import sys
import time

import numpy as np
import torch

import ops
import r6_expert
import r6_features
import r6_head
import r6_lib
import run_protocol as rp
import runtime as rt

G1_CLASS_DELTA = 0.005      # P1: per-class val IoU gain to adopt a head
G2_MA_DELTA = 0.010         # P2: MA val IoU gain to adopt the expert fusion
G3_MIOU_DELTA = 0.002       # P3: test mIoU gain (pristine champion -> final)
ALPHAS = [0.3, 0.5, 0.7, 1.0]
PHASE_HOURS = {'p0': 0.60, 'p1': 2.9, 'p1_test': 3.4, 'p2': 4.9, 'p3': 5.7}
TRAIN_IMAGES, VAL_IMAGES, TEST_IMAGES = 454, 76, 227
CACHE_SPACE_GB = 34.0        # MEASURED: ~31.5GB per class (45-60MB/image,
                             #  6 fp16 channels at the real ori sizes)
CHAMPION_TEST_REF = 0.442115        # historical locked value (contract)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--settings', required=True)
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--rehearse', action='store_true')
    args = ap.parse_args()

    settings = rp.validate_settings(args.settings)
    if settings['_pending']:
        raise RuntimeError('missing assets: %s' % settings['_pending'])
    os.makedirs(args.run_dir, exist_ok=True)
    log = r6_lib.make_log(args.run_dir)
    state = r6_lib.State(args.run_dir)

    dl_dt = ops.deadline_of(ops.load_ledger(args.run_dir))
    global_deadline = dl_dt.timestamp()
    if 'start' not in state.d:
        state.set('start', time.time())
    T0 = float(state.d['start'])
    log('round6: ledger deadline %s (%.2fh left)'
        % (dl_dt.isoformat(), (global_deadline - time.time()) / 3600.0))

    def phase_deadline(name):
        return min(T0 + PHASE_HOURS[name] * 3600.0,
                   global_deadline - 20 * 60.0)

    rt.inject_repo(settings['repo'])
    settings['_cfg'] = rt.load_config(settings['repo'], settings['config'])
    settings['_cfg'].model['source_checkpoint'] = \
        settings['_abs']['reference_checkpoint']

    if args.smoke:
        smoke(settings, args.run_dir, log, global_deadline)
        return
    if args.rehearse:
        rehearse(settings, args.run_dir, log, global_deadline)
        return

    # ============================== P0 ==============================
    p0 = state.d.get('p0')
    if p0 is None:
        d0 = phase_deadline('p0')
        assignment = rt.audit_split(settings['data_root'])
        sizes = {s: sum(1 for v in assignment.values() if v['split'] == s)
                 for s in ('train', 'val', 'test')}
        if (sizes['train'], sizes['val'], sizes['test']) != \
                (TRAIN_IMAGES, VAL_IMAGES, TEST_IMAGES):
            raise RuntimeError('split sizes %r != (454,76,227)' % (sizes,))
        log('P0: split audit OK (454/76/227)')
        champion = rt.ChampionEnsemble(settings, settings['champion'])
        t0 = time.time()
        with torch.no_grad():
            miou, per_class, _ = rt.evaluate_split(
                champion.predict, 'val', settings['data_root'],
                settings['_cfg'], d0, log,
                dump_counts=os.path.join(args.run_dir,
                                         'p0_champ_val_counts.json'))
        log('P0: pristine champion val mIoU %.4f %s (%.0fs; r5 ref 0.4267)'
            % (miou, {k: round(v, 4) for k, v in per_class.items()},
               time.time() - t0))
        t0 = time.time()
        with torch.no_grad():
            t_miou, t_per, _ = rt.evaluate_split(
                champion.predict, 'test', settings['data_root'],
                settings['_cfg'], d0, log,
                dump_counts=os.path.join(args.run_dir,
                                         'p0_champ_test_counts.json'))
        log('P0: pristine champion test mIoU %.4f %s (%.0fs; contract ref '
            '0.4421)' % (t_miou, {k: round(v, 4) for k, v in t_per.items()},
                         time.time() - t0))
        base_val = _per_image(os.path.join(args.run_dir,
                                           'p0_champ_val_counts.json'))
        profile = {}
        for c in rt.CLASSES:
            tp, fp, fn = _sum_counts(base_val, c)
            profile[c] = {'iou': r6_lib.iou(tp, fp, fn),
                          'counts': [tp, fp, fn],
                          'fn_share': (fn / float(fn + fp)) if (fn + fp)
                          else 0.0}
        order = sorted([c for c in rt.CLASSES if c != 'MA'],
                       key=lambda c: profile[c]['iou'])[:2]
        p0 = {'champion_val_miou': miou,
              'champion_val_per_class': per_class,
              'champion_test_miou': t_miou,
              'champion_test_per_class': t_per,
              'profile': profile, 'attack_order': order}
        state.set('p0', p0)
        r6_lib.atomic_json(os.path.join(args.run_dir, 'p0_report.json'), p0)
        log('P0: attack order %s (val IoU %s; FN-share %s)'
            % (order, {c: round(profile[c]['iou'], 4) for c in order},
               {c: round(profile[c]['fn_share'], 3) for c in order}))
    else:
        champion = rt.ChampionEnsemble(settings, settings['champion'])
        log('P0: cached (attack order %s)' % p0['attack_order'])

    base_val = _per_image(os.path.join(args.run_dir,
                                       'p0_champ_val_counts.json'))

    # ============================== P1 ==============================
    adopted_heads, h6 = {}, dict(state.d.get('h6', {}))
    for cls in p0['attack_order']:
        if cls in h6:
            if h6[cls]['decision'] == 'adopt':
                head, _ck = _load_head(h6[cls]['head_path'])
                adopted_heads[cls] = head
                champion.heads[cls] = head
            log('P1[%s]: cached -> %s' % (cls, h6[cls]['decision']))
            continue
        d1 = phase_deadline('p1')
        if time.time() > d1 - 20 * 60:
            r6_lib.deviation(args.run_dir, 'p1_skipped',
                             {'cls': cls, 'reason': 'phase budget'})
            continue
        log('P1[%s]: feature caches' % cls)
        need_build = (r6_features.cache_needed(args.run_dir, cls, 'train',
                                               TRAIN_IMAGES, settings) or
                      r6_features.cache_needed(args.run_dir, cls, 'val',
                                               VAL_IMAGES, settings))
        if need_build:
            r6_lib.ensure_space('/root/autodl-tmp', CACHE_SPACE_GB,
                                args.run_dir, log)
        else:
            log('P1[%s]: caches complete on disk -> no reservation needed '
                '(%.1fG free)' % (cls, r6_lib.free_gb('/root/autodl-tmp')))
        try:
            ctr = r6_features.build_cache(settings, args.run_dir, cls, 'train',
                                          champion, log, d1, TRAIN_IMAGES)
            cva = r6_features.build_cache(settings, args.run_dir, cls, 'val',
                                          champion, log, d1, VAL_IMAGES)
            res = r6_head.train_head(settings, args.run_dir, cls, ctr, cva,
                                     log, d1)
        except r6_lib.PhaseTimeout as ex:
            # covers cache build AND head training: on a later restart the
            # partial cache resumes (per-image files are validated and
            # skipped), so nothing is wasted
            r6_lib.deviation(args.run_dir, 'p1_timeout',
                             {'cls': cls, 'detail': str(ex)})
            continue
        base_iou = _class_iou(base_val, cls)
        delta = res['val_iou'] - base_iou
        decision = 'adopt' if delta >= G1_CLASS_DELTA else 'reject'
        h6[cls] = {'val_cls_iou': res['val_iou'], 'baseline_val_iou': base_iou,
                   'delta': delta, 'decision': decision,
                   'head_path': res['path'], 'sha256': res['sha256'],
                   'best_step': res['best_step'], 'curve': res['curve']}
        state.set('h6', h6)
        log('P1[%s] GATE: val %.4f vs baseline %.4f (delta %+.4f, need %+.4f)'
            ' -> %s' % (cls, res['val_iou'], base_iou, delta, G1_CLASS_DELTA,
                        decision))
        if decision == 'adopt':
            adopted_heads[cls] = _declare_rounding(res['head'])
            champion.heads[cls] = adopted_heads[cls]
        _drop_caches(args.run_dir, cls, log)
        del res
        torch.cuda.empty_cache()

    # ============================= P1T ==============================
    if adopted_heads and 'p1_test' not in state.d:
        d1t = phase_deadline('p1_test')
        need_test_h = (TEST_IMAGES * 5.0) / 3600.0
        if time.time() < d1t - (need_test_h + 0.3) * 3600:
            log('P1T: mid test-confirmation (heads %s)'
                % sorted(adopted_heads))
            rv = _stack_pass(settings, champion, None, [], 'val', log, d1t,
                             os.path.join(args.run_dir,
                                          'p1_test_val_counts.json'))
            rt_ = _stack_pass(settings, champion, None, [], 'test', log, d1t,
                              os.path.join(args.run_dir,
                                           'p1_test_test_counts.json'))
            boot_val = r6_lib.paired_bootstrap(base_val, rv['per_image'],
                                               n=500)
            boot_test = r6_lib.paired_bootstrap(
                _per_image(os.path.join(args.run_dir,
                                        'p0_champ_test_counts.json')),
                rt_['per_image'])
            state.set('p1_test', {'val_miou': rv['miou'],
                                  'val_per_class': rv['per_class'],
                                  'test_miou': rt_['miou'],
                                  'test_per_class': rt_['per_class'],
                                  'bootstrap_val_vs_champion': boot_val,
                                  'bootstrap_test_vs_champion': boot_test})
            log('P1T: test mIoU %.4f (pristine champion %.4f, contract %.4f);'
                ' delta %+.4f CI [%+.4f,%+.4f]'
                % (rt_['miou'], p0['champion_test_miou'], CHAMPION_TEST_REF,
                   boot_test['delta'], boot_test['ci'][0],
                   boot_test['ci'][1]))
        else:
            r6_lib.deviation(args.run_dir, 'p1_test_skipped',
                             {'reason': 'phase budget'})

    # ============================== P2 ==============================
    p2 = state.d.get('p2')
    expert = None
    if p2 is None:
        d2 = phase_deadline('p2')
        rng = np.random.RandomState(r6_lib.SEED)
        log('P2: MA patch plan')
        plan = r6_expert.build_patch_plan(settings, args.run_dir, log, rng)
        try:
            tr = r6_expert.train_expert(settings, args.run_dir, plan, log, d2)
            sw = _alpha_pass(settings, champion, tr['model'], args.run_dir,
                             log, d2)
            delta_ma = sw['delta_vs_champion']
            decision = 'adopt' if delta_ma >= G2_MA_DELTA else 'reject'
            p2 = {'expert_path': tr['path'], 'sha256': tr['sha256'],
                  'val_patch_iou': tr['val_patch_iou'],
                  'best_step': tr['best_step'], 'curve': tr['curve'],
                  'alpha_sweep': sw, 'delta_ma_val': delta_ma,
                  'best_alpha': sw['best_alpha'], 'decision': decision}
            state.set('p2', p2)
            log('P2 GATE: MA val %.4f vs champion %.4f (delta %+.4f, need '
                '%+.4f) -> %s (alpha %.1f)'
                % (sw['best_ma_iou'], sw['champion_ma_val_iou'], delta_ma,
                   G2_MA_DELTA, decision, sw['best_alpha']))
            if decision == 'adopt':
                expert = tr['model']
        except r6_lib.PhaseTimeout as ex:
            r6_lib.deviation(args.run_dir, 'p2_timeout', {'detail': str(ex)})
            p2 = {'decision': 'timeout'}
            state.set('p2', p2)
    else:
        log('P2: cached -> %s' % p2.get('decision'))
        if p2.get('decision') == 'adopt':
            expert, _ck = r6_expert.load_expert(p2['expert_path'])

    # ============================== P3 ==============================
    if 'p3' not in state.d:
        d3 = phase_deadline('p3')
        alphas = [float(p2['best_alpha'])] if expert is not None else []
        log('P3: final test (heads %s, expert %s, alpha %s)'
            % (sorted(adopted_heads), expert is not None, alphas))
        stack = _stack_pass(settings, champion, expert, alphas, 'test', log,
                            d3, os.path.join(args.run_dir,
                                             'p3_stack_test_counts.json'))
        final_pi = stack['per_image']
        if expert is not None and alphas:
            a = alphas[0]
            final_pi = {s: dict(stack['per_image'][s]) for s in
                        stack['per_image']}
            for s in final_pi:
                final_pi[s]['MA'] = stack['fused'][a][s]['MA']
        base_pi = _per_image(os.path.join(args.run_dir,
                                          'p0_champ_test_counts.json'))
        boot = r6_lib.paired_bootstrap(base_pi, final_pi)
        final_miou = sum(r6_lib.iou(*_sum_counts(final_pi, c))
                         for c in rt.CLASSES) / 4.0
        p3 = {'champion_test_miou_pristine': p0['champion_test_miou'],
              'champion_test_miou_contract': CHAMPION_TEST_REF,
              'stack_test_miou': stack['miou'],
              'final_test_miou': final_miou,
              'final_per_class': {c: r6_lib.iou(*_sum_counts(final_pi, c))
                                  for c in rt.CLASSES},
              'bootstrap_vs_pristine_champion': boot,
              'adopted_heads': sorted(adopted_heads),
              'alpha': alphas[0] if alphas else None,
              'g3_pass': bool(boot['delta'] >= G3_MIOU_DELTA and
                              not boot['crosses_zero'])}
        state.set('p3', p3)
        log('P3: final test %.4f (pristine champion %.4f, contract %.4f); '
            'delta %+.4f CI [%+.4f,%+.4f] crosses0=%s -> G3 %s'
            % (final_miou, p0['champion_test_miou'], CHAMPION_TEST_REF,
               boot['delta'], boot['ci'][0], boot['ci'][1],
               boot['crosses_zero'], p3['g3_pass']))

    summary = {'round': 6,
               'gates': {'G1_class_delta': G1_CLASS_DELTA,
                         'G2_ma_delta': G2_MA_DELTA,
                         'G3_miou_delta': G3_MIOU_DELTA},
               'p0': state.d['p0'], 'h6': state.d.get('h6', {}),
               'p1_test': state.d.get('p1_test'), 'p2': state.d.get('p2'),
               'p3': state.d.get('p3'),
               'champion_test_contract': CHAMPION_TEST_REF, 'target': 47.20,
               'status': ('breakthrough'
                          if state.d.get('p3', {}).get('g3_pass')
                          else 'no_breakthrough'),
               'created': time.strftime('%Y-%m-%dT%H:%M:%S')}
    r6_lib.atomic_json(os.path.join(args.run_dir, 'r6_summary.json'), summary)
    log('ROUND6 COMPLETE status=%s' % summary['status'])


# ------------------------------------------------------------- helpers ----
def _stack_pass(settings, champion, expert_model, alphas, split, log, deadline,
                dump_counts=None):
    """Champion (+ adopted heads) pass with optional expert fusion.  When
    expert_model is None this is a plain champion evaluation (alphas must be
    empty) -- guarded explicitly."""
    if expert_model is None and alphas:
        raise ValueError('alphas given without an expert model')
    res = r6_expert.eval_split_with_expert(settings, champion, expert_model,
                                           list(alphas), split, log, deadline)
    per_image = res['champ']
    miou = res['champ_iou']['mIoU']
    if dump_counts:
        r6_lib.atomic_json(dump_counts, {
            'split': split, 'miou': miou,
            'per_class': {c: res['champ_iou'][c] for c in rt.CLASSES},
            'global_counts': res['champ_counts'], 'per_image': per_image})
    return {'miou': miou,
            'per_class': {c: res['champ_iou'][c] for c in rt.CLASSES},
            'per_image': per_image, 'fused': res.get('fused', {}),
            'raw': res}


def _alpha_pass(settings, champion, expert_model, run_dir, log, deadline):
    res = r6_expert.eval_split_with_expert(settings, champion, expert_model,
                                           ALPHAS, 'val', log, deadline)
    champ_ma = res['champ_iou']['MA']
    results = {a: {'ma_iou': res['fused_ma_iou'][str(a)],
                   'counts': res['fused_totals'][str(a)]} for a in ALPHAS}
    best_a = max(ALPHAS, key=lambda a: results[a]['ma_iou'])
    out = {'champion_ma_val_iou': champ_ma,
           'champion_val_miou': res['champ_iou']['mIoU'],
           'sweep': {str(a): results[a] for a in ALPHAS},
           'best_alpha': best_a, 'best_ma_iou': results[best_a]['ma_iou'],
           'delta_vs_champion': results[best_a]['ma_iou'] - champ_ma}
    os.makedirs(os.path.join(run_dir, 'expert'), exist_ok=True)
    r6_lib.atomic_json(os.path.join(run_dir, 'expert', 'alpha_sweep.json'), out)
    for a in ALPHAS:
        log('M1 alpha %.1f: MA val %.4f' % (a, results[a]['ma_iou']))
    return out


def _per_image(path):
    with open(path, 'r', encoding='utf-8') as f:
        d = json.load(f)
    return d['per_image']


def _sum_counts(per_image, cls):
    tp = fp = fn = 0
    for s in per_image:
        v = per_image[s][cls]
        tp += v[0]
        fp += v[1]
        fn += v[2]
    return tp, fp, fn


def _class_iou(per_image, cls):
    return r6_lib.iou(*_sum_counts(per_image, cls))


def _load_head(path):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    h = rt.CorrectionHead()
    h.load_state_dict(ck['head'], strict=True)
    h.r6_round_inputs = True        # declared parity flag (see runtime)
    return h.cuda().eval(), ck


def _declare_rounding(head):
    """All round-6 heads are trained on fp16 features; the flag makes the
    live inference path round identically (champion's MA head is untouched
    because it never carries the flag)."""
    head.r6_round_inputs = True
    return head


def _drop_caches(run_dir, cls, log):
    for split in ('train', 'val'):
        try:
            r6_features.delete_cache(run_dir, cls, split, log)
        except Exception as ex:
            log('cache cleanup warning (%s/%s): %r' % (cls, split, ex))


# -------------------------------------------------------------- smoke ----
def rehearse(settings, run_dir, log, global_deadline):
    """Zero-risk rehearsal of the RESUME path on the real caches/models:
    builds the champion, replays the P1 head-training loop (short) INCLUDING
    the validation point that crashed at 14:16, and exercises the P2/P3
    evaluator on 3 val images.  Writes nothing into the run state."""
    log('rehearse: champion')
    champion = rt.ChampionEnsemble(settings, settings['champion'])
    order = state_attack_order(run_dir) or ['HE']
    cls = order[0]        # attack_order is a LIST of classes
    tmp = os.path.join(run_dir, 'rehearse_out')
    os.makedirs(tmp, exist_ok=True)
    if r6_features.cache_needed(run_dir, cls, 'train', TRAIN_IMAGES, settings):
        raise RuntimeError('rehearse: %s train cache missing' % cls)
    ctr = r6_features.cache_dir(run_dir, cls, 'train')
    cva = r6_features.cache_dir(run_dir, cls, 'val')
    log('rehearse: head training on %s cache (40 steps, val every 20, '
        '3 val images)' % cls)
    res = r6_head.train_head(settings, run_dir, cls, ctr, cva, log,
                             global_deadline, steps=40, val_every=20,
                             val_max_images=3, out_dir_override=tmp,
                             save=False)
    log('rehearse: head val path OK (val %s IoU %.4f on 3 images)'
        % (cls, res['val_iou']))
    log('rehearse: expert + fusion evaluator on 3 val images')
    model = r6_expert.UNet().cuda()
    out = r6_expert.eval_split_with_expert(
        settings, champion, model, [0.5], 'val', log, global_deadline)
    log('rehearse: evaluator OK (champion val mIoU %.4f, fused MA %.4f)'
        % (out['champ_iou']['mIoU'], out['fused_ma_iou']['0.5']))
    del champion, model
    torch.cuda.empty_cache()
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)
    log('REHEARSE OK -- resume path validated; state untouched')


def state_attack_order(run_dir):
    try:
        with open(os.path.join(run_dir, 'r6_state.json'), 'r',
                  encoding='utf-8') as f:
            return (json.load(f).get('p0') or {}).get('attack_order')
    except Exception:
        return None


def smoke(settings, run_dir, log, global_deadline):
    """5 gates: champion features / head identity / head steps / expert steps
    / expert sliding window."""
    log('smoke: champion')
    champion = rt.ChampionEnsemble(settings, settings['champion'])
    r6_lib.ensure_space('/root/autodl-tmp', 2.0, run_dir, log,
                        allow_cleanup=False)
    cfg = copy.deepcopy(settings['_cfg'].data['val'])
    cfg.test_mode = True
    cfg.pop('split', None)
    cfg.img_dir = os.path.join(settings['data_root'], 'images', 'val')
    cfg.ann_dir = os.path.join(settings['data_root'], 'labels', 'val')
    from mmseg.datasets import build_dataset
    ds = build_dataset(cfg)
    s = ds[0]
    img = rt._unwrap_img(s['img']).unsqueeze(0).cuda()
    meta = rt._unwrap_meta(s['img_metas'])
    pf, ps, pr = champion.pre_head_probs(img, meta)
    if pf.shape[0] != 4 or tuple(pf.shape[1:]) != tuple(
            meta[0]['ori_shape'][:2]):
        raise RuntimeError('pre_head_probs shape %r' % (tuple(pf.shape),))
    log('smoke gate1: pre_head_probs OK %r' % (tuple(pf.shape),))

    smk = os.path.join(run_dir, 'smoke')
    ctr = r6_features.build_cache(settings, smk, 'HE', 'train', champion,
                                  log, global_deadline, 3, max_images=3)
    stems = r6_features.stems_of(ctr)
    if len(stems) != 3:
        raise RuntimeError('smoke: cache has %d entries' % len(stems))
    img_f16, fields = r6_features.load_entry(ctr, stems[0])
    if img_f16.dtype.name != 'float16' or fields.dtype.name != 'float16':
        raise RuntimeError('smoke: bad cache dtypes %s/%s'
                           % (img_f16.dtype, fields.dtype))
    if img_f16.shape[0] != 3 or fields.shape[0] != 3 or \
            tuple(img_f16.shape[1:]) != tuple(fields.shape[1:]):
        raise RuntimeError('smoke: bad cache shapes %r/%r'
                           % (img_f16.shape, fields.shape))
    log('smoke gate1b: feature cache entry OK img%r fields%r'
        % (img_f16.shape, fields.shape))

    head = rt.CorrectionHead().cuda()
    with torch.no_grad():
        head.net[4].weight.zero_()
        head.net[4].bias.zero_()
    a_i, a_f, _l0 = r6_head._crop(img_f16, fields,
                                  np.zeros(img_f16.shape[1:], dtype=np.int64),
                                  0, 0, 256)
    f7 = r6_head._f7_on_device(a_i, a_f, head)
    with torch.no_grad():
        d = head(f7)
    if float(d.abs().max()) > 1e-6:
        raise RuntimeError('zero-init head not identity')
    log('smoke gate2: head identity OK')
    _, lab = rt.load_image_label(settings['data_root'], 'train', stems[0])
    opt = torch.optim.Adam(head.parameters(), lr=1e-3)
    for i in range(5):
        a_i, a_f, labc = r6_head._sample_crop(
            img_f16, fields, lab, rt.CLASSES.index('HE') + 1,
            np.random.RandomState(i))
        f7 = r6_head.f7_from_arrays(a_i, a_f).cuda()
        tgt = torch.as_tensor((np.asarray(labc) == 2).astype(np.float32)
                              ).unsqueeze(0).unsqueeze(0).cuda()
        opt.zero_grad()
        loss = r6_head._dice_loss(torch.sigmoid(rt._logit(f7[:, 3:4]) +
                                                head(f7)), tgt)
        loss.backward()
        opt.step()
    log('smoke gate3: head 5 steps OK (loss %.4f)' % float(loss))

    plan = r6_expert.build_patch_plan(settings, smk, log,
                                      np.random.RandomState(r6_lib.SEED))
    model = r6_expert.UNet().cuda()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    tr_stems = [s for s, v in plan.items() if v['pos']][:2]
    if not tr_stems:
        raise RuntimeError('smoke: no MA-positive train images')
    ip, lp = r6_expert._paths(settings['data_root'], 'train', tr_stems[0])
    y, x = plan[tr_stems[0]]['pos'][0]
    for i in range(5):
        t = torch.from_numpy(r6_expert._crop_rgb(ip, y, x)
                             .transpose(2, 0, 1)).unsqueeze(0).cuda()
        g = torch.from_numpy(
            r6_expert._crop_lab(lp, y, x)[None][None]).cuda()
        opt.zero_grad()
        loss = r6_expert._loss(model(t), g)
        loss.backward()
        opt.step()
    log('smoke gate4: expert 5 steps OK (loss %.4f)' % float(loss))
    p = r6_expert.expert_full_prob(model, ip)
    if p.ndim != 2 or float(p.min()) < 0 or float(p.max()) > 1:
        raise RuntimeError('expert sliding window bad output')
    log('smoke gate5: expert sliding window OK %r' % (p.shape,))
    del champion, model
    torch.cuda.empty_cache()
    import shutil
    shutil.rmtree(smk, ignore_errors=True)
    log('SMOKE OK (6 gates; smoke cache removed)')


if __name__ == '__main__':
    try:
        main()
    except rp.ShutdownGraceful as e:
        print('[shutdown-guard] %s' % e, flush=True)
        sys.exit(0)
