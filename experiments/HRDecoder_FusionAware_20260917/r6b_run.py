#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""Round-6b orchestrator: the authorised MA-expert recipe revision.

Flow: validate settings -> build champion -> R6b patch plan (256, Tversky)
-> train 2000 steps -> val alpha sweep (shared evaluator, window pointed at
256/128) -> GATE: MA val IoU >= champion + 0.010 -> (if pass) test
confirmation with a paired bootstrap -> summary.  Then the guard's EXIT trap
backs up + shuts down, exactly like every other round.

Provenance: the revision and the user's authorisation are written to
DEVIATIONS.jsonl at start-up, before any training happens.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

import ops
import r6_expert as X
import r6_lib
import r6b_expert as B
import run_protocol as rp
import runtime as rt

ALPHAS = [0.3, 0.5, 0.7, 1.0]
G2_MA_DELTA = 0.010          # unchanged from the round-6 plan
TRAIN_IMAGES, VAL_IMAGES, TEST_IMAGES = 454, 76, 227
PHASE_HOURS = {'p0': 0.25, 'plan': 0.30, 'train': 1.10, 'sweep': 1.60,
               'test': 2.20}


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
    log = r6_lib.make_log(args.run_dir)
    state = r6_lib.State(args.run_dir)

    ledger = ops.load_ledger(args.run_dir)
    dl = ops.deadline_of(ledger)
    global_deadline = dl.timestamp()
    if 'start' not in state.d:
        state.set('start', time.time())
    T0 = float(state.d['start'])
    log('round6b: deadline %s (%.2fh)' % (dl.isoformat(),
                                          (global_deadline - time.time()) / 3600))

    if 'provenance' not in state.d:
        state.set('provenance', {
            'authorised_by_user_at': '2026-09-18T16:34:00+08:00',
            'reason': ('R6 P2 (512 patches, Dice+0.5*BCE) collapsed to the '
                       'null solution: loss ~1.00 and val patch IoU 0.0000 '
                       'for 1000+ steps; the BCE gradient is dominated by '
                       '~1e5 negative pixels per patch'),
            'changed': {'patch': '512 -> 256',
                        'loss': 'Dice + 0.5*BCE -> Tversky(0.3, 0.7)',
                        'steps': '3000 -> 2000'},
            'unchanged': ['U-Net topology', 'native-resolution patch plan '
                          'with a positive-pixel floor',
                          'alpha fusion sweep {0.3,0.5,0.7,1.0}',
                          'gate: MA val IoU >= champion + 0.010 -> test'],
        })
        r6_lib.deviation(args.run_dir, 'r6b_authorised_recipe_change',
                         state.d['provenance'])
    log('R6b provenance recorded (authorised recipe revision)')

    def phase_deadline(name):
        return min(T0 + PHASE_HOURS[name] * 3600.0,
                   global_deadline - 15 * 60.0)

    rt.inject_repo(settings['repo'])
    settings['_cfg'] = rt.load_config(settings['repo'], settings['config'])
    settings['_cfg'].model['source_checkpoint'] = \
        settings['_abs']['reference_checkpoint']

    if args.smoke:
        smoke(settings, args.run_dir, log, global_deadline)
        return

    # ---------------- champion + baseline ----------------
    log('R6b: champion')
    champion = rt.ChampionEnsemble(settings, settings['champion'])

    # ---------------- patch plan + training ----------------
    plan = state.d.get('plan')
    if plan is None:
        rng = np.random.RandomState(r6_lib.SEED)
        log('R6b: patch plan (%d px windows, >=%d MA px)'
            % (B.PATCH, B.MIN_POS_PX))
        plan = B.build_plan(settings, args.run_dir, log, rng)
        state.set('plan', {'n_images': len(plan)})
    tr = state.d.get('train')
    if tr is None:
        out = B.train(settings, args.run_dir, plan, log,
                      phase_deadline('train'))
        tr = {'path': out['path'], 'sha256': out['sha256'],
              'best_step': out['best_step'],
              'val_patch_iou': out['val_patch_iou'], 'curve': out['curve']}
        state.set('train', tr)
        model = out['model']
    else:
        model, _ck = X.load_expert(tr['path'])

    # ---------------- val alpha sweep (shared evaluator, 256 windows) ----
    X.PATCH, X.STRIDE = B.PATCH, B.STRIDE      # point the shared evaluator
    if 'sweep' not in state.d:
        d_sweep = phase_deadline('sweep')
        log('R6b: val alpha sweep (expert windows %d/%d)'
            % (B.PATCH, B.STRIDE))
        res = X.eval_split_with_expert(settings, champion, model, ALPHAS,
                                       'val', log, d_sweep)
        champ_ma = res['champ_iou']['MA']
        results = {a: {'ma_iou': res['fused_ma_iou'][str(a)],
                       'counts': res['fused_totals'][str(a)]}
                   for a in ALPHAS}
        best_a = max(ALPHAS, key=lambda a: results[a]['ma_iou'])
        sw = {'champion_ma_val_iou': champ_ma,
              'champion_val_miou': res['champ_iou']['mIoU'],
              'sweep': {str(a): results[a] for a in ALPHAS},
              'best_alpha': best_a, 'best_ma_iou': results[best_a]['ma_iou'],
              'delta_vs_champion': results[best_a]['ma_iou'] - champ_ma}
        for a in ALPHAS:
            log('R6b alpha %.1f: MA val %.4f' % (a, results[a]['ma_iou']))
        sw['decision'] = ('adopt' if sw['delta_vs_champion'] >= G2_MA_DELTA
                          else 'reject')
        state.set('sweep', sw)
        log('R6b GATE: MA val %.4f vs champion %.4f (delta %+.4f, need %+.4f)'
            ' -> %s (alpha %.1f)'
            % (sw['best_ma_iou'], champ_ma, sw['delta_vs_champion'],
               G2_MA_DELTA, sw['decision'], best_a))
    sw = state.d['sweep']

    # ---------------- test confirmation ----------------
    if sw.get('decision') == 'adopt' and 'test' not in state.d:
        d_test = phase_deadline('test')
        a = float(sw['best_alpha'])
        log('R6b: TEST confirmation at alpha %.1f' % a)
        res = X.eval_split_with_expert(
            settings, champion, model, [a], 'test', log, d_test,
            dump_counts=os.path.join(args.run_dir, 'r6b_test_stack_counts.json'))
        fused = {s: dict(res['champ'][s]) for s in res['champ']}
        for s in fused:
            fused[s]['MA'] = res['fused'][a][s]['MA']
        boot = r6_lib.paired_bootstrap(res['champ'], fused)
        test_miou = sum(r6_lib.iou(*_sum(fused, c)) for c in rt.CLASSES) / 4.0
        champ_miou = res['champ_iou']['mIoU']
        state.set('test', {
            'alpha': a, 'test_miou_champion': champ_miou,
            'test_miou_with_expert': test_miou,
            'test_per_class_champion': {c: res['champ_iou'][c]
                                        for c in rt.CLASSES},
            'test_per_class_with_expert': {c: r6_lib.iou(*_sum(fused, c))
                                           for c in rt.CLASSES},
            'test_ma_champion': res['champ_iou']['MA'],
            'test_ma_with_expert': r6_lib.iou(*_sum(fused, 'MA')),
            'bootstrap_vs_champion_stack': boot})
        log('R6b TEST: mIoU %.4f -> %.4f (champion stack %.4f); MA %.4f -> '
            '%.4f; bootstrap delta %+.4f CI [%+.4f,%+.4f]'
            % (champ_miou, test_miou, champ_miou, res['champ_iou']['MA'],
               r6_lib.iou(*_sum(fused, 'MA')), boot['delta'], boot['ci'][0],
               boot['ci'][1]))
    elif sw.get('decision') != 'adopt':
        log('R6b: gate rejected -> no test confirmation (pre-registered)')

    summary = {'round': '6b', 'recipe_revision': state.d['provenance'],
               'train': state.d['train'], 'sweep': sw,
               'test': state.d.get('test'),
               'gate': {'G2_ma_delta': G2_MA_DELTA},
               'status': ('adopted' if sw.get('decision') == 'adopt'
                          else 'rejected'),
               'created': time.strftime('%Y-%m-%dT%H:%M:%S')}
    r6_lib.atomic_json(os.path.join(args.run_dir, 'r6b_summary.json'), summary)
    log('ROUND6B COMPLETE status=%s' % summary['status'])


def _sum(per_image, cls):
    tp = fp = fn = 0
    for s in per_image:
        v = per_image[s][cls]
        tp += v[0]
        fp += v[1]
        fn += v[2]
    return tp, fp, fn


def smoke(settings, run_dir, log, global_deadline):
    """Short smoke: champion + 20 training steps + val patch IoU + one
    full-image sliding-window pass."""
    champion = rt.ChampionEnsemble(settings, settings['champion'])
    rng = np.random.RandomState(r6_lib.SEED)
    smk = os.path.join(run_dir, 'smoke')
    os.makedirs(smk, exist_ok=True)          # build_plan writes the plan there
    plan = B.build_plan(settings, smk, log, rng)
    stems = [s for s, v in plan.items() if v['pos']][:4]
    if not stems:
        raise RuntimeError('smoke: no positive windows found')
    model = X.UNet().cuda()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for i in range(20):
        t, g = B._batch(plan, stems, rng, settings['data_root'])
        if t is None:
            continue
        opt.zero_grad()
        loss = B.tversky_loss(model(t), g)
        loss.backward()
        opt.step()
    log('smoke gate1: 20 Tversky steps OK (loss %.4f)' % float(loss))
    pset = B.val_patch_set(settings, rng, per_image=1, n_images=3)
    vi = B.val_patch_iou(model, pset, settings)
    log('smoke gate2: val patch IoU path OK (%.4f on %d patches)'
        % (vi, len(pset)))
    ip, _lp = X._paths(settings['data_root'], 'val', pset[0][0])
    full = B.full_prob(model, ip)
    if full.ndim != 2 or float(full.min()) < 0 or float(full.max()) > 1:
        raise RuntimeError('smoke: bad full-image probabilities')
    log('smoke gate3: 256/128 sliding window OK %r' % (full.shape,))
    del champion, model
    torch.cuda.empty_cache()
    import shutil
    shutil.rmtree(os.path.join(run_dir, 'smoke'), ignore_errors=True)
    log('SMOKE OK (3 gates)')


if __name__ == '__main__':
    try:
        main()
    except rp.ShutdownGraceful as e:
        print('[shutdown-guard] %s' % e, flush=True)
        sys.exit(0)
