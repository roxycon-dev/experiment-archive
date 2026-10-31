# -*- coding: utf-8 -*-
"""Decide whether an audit checkpoint can stand in for the lost D student.

Method (functional, not bitwise -- a re-serialised copy has different file
bytes but identical tensors):
  1. print the candidate's structure (top keys, reference.* count, meta);
  2. build the champion with member D loaded from the candidate;
  3. run the val split and compare PER-IMAGE PER-CLASS counts against the
     verified P0 baseline (p0_champ_val_counts.json);
  4. verdict: identical counts on all 76 images -> functionally the same
     weights -> the lost file is recovered.
Nothing in the run state is modified (only a *_probe_*.json report).
"""
import argparse
import json
import os
import sys

sys.path.insert(0, '/root/autodl-tmp/hrplus/glm')
os.chdir('/root/autodl-tmp/hrplus/repo')

import numpy as np                      # noqa: E402
import torch                            # noqa: E402

import runtime as rt                    # noqa: E402
import run_protocol as rp               # noqa: E402

GLM = '/root/autodl-tmp/hrplus/glm'
RUN = GLM + '/runs/r6'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--candidate', required=True)
    ap.add_argument('--settings', default=GLM + '/settings.json')
    ap.add_argument('--run-dir', default=RUN)
    ap.add_argument('--tag', default='c2')
    ap.add_argument('--strip-unexpected', action='store_true',
                    help='drop candidate tensors the D model does not have '
                         '(e.g. an ma_refiner branch) before loading')
    args = ap.parse_args()

    ck = torch.load(args.candidate, map_location='cpu', weights_only=False)
    sd = ck.get('state_dict', ck)
    ref_keys = [k for k in sd if str(k).startswith('reference.')]
    print('candidate : %s' % args.candidate)
    print('top keys  : %s' % list(ck.keys())[:8])
    print('n tensors : %d (reference.* %d)' % (len(sd), len(ref_keys)))
    print('meta      : %s' % json.dumps(ck.get('meta', {}), default=str)[:400])

    settings = rp.validate_settings(args.settings)
    rt.inject_repo(settings['repo'])       # mmseg becomes importable here
    settings['_cfg'] = rt.load_config(settings['repo'], settings['config'])
    settings['_cfg'].model['source_checkpoint'] = \
        settings['_abs']['reference_checkpoint']
    print('settings : validate_settings PASSED (other assets intact)')

    cand_path = args.candidate
    if args.strip_unexpected:
        import copy
        from mmseg.models import build_segmentor
        c = rt.load_config(settings['repo'], settings['config'])
        c.model['source_checkpoint'] = settings['_abs']['reference_checkpoint']
        ref_model = build_segmentor(c.model)
        expected = set(ref_model.state_dict().keys())
        keep = {k: v for k, v in sd.items() if k in expected}
        dropped = sorted(set(sd) - expected)
        missing = sorted(expected - set(sd))
        print('strip    : dropped %d unexpected keys, missing %d'
              % (len(dropped), len(missing)))
        print('strip    : dropped sample %s' % dropped[:8])
        if missing:
            raise RuntimeError('candidate lacks %d expected keys' % len(missing))
        cand_path = '/tmp/r6_candidate_stripped.pth'
        torch.save({'state_dict': keep, 'meta': ck.get('meta', {})}, cand_path)
        print('strip    : wrote %s' % cand_path)

    spec = dict(settings['champion'])
    members = []
    for m in spec['members']:
        m = dict(m)
        if m['name'] == 'D':
            m['checkpoint'] = cand_path
            print('member D  : override -> %s' % cand_path)
        members.append(m)
    spec['members'] = members
    settings['champion'] = spec

    log = lambda msg: print('[probe] %s' % msg, flush=True)
    ens = rt.ChampionEnsemble(settings, settings['champion'])
    out = os.path.join(args.run_dir, '%s_probe_val_counts.json' % args.tag)
    with torch.no_grad():
        miou, per_class, _ = rt.evaluate_split(
            ens.predict, 'val', settings['data_root'], settings['_cfg'],
            1e18, log, dump_counts=out)
    print('candidate champion val mIoU %.6f  %s'
          % (miou, {k: round(v, 6) for k, v in per_class.items()}))

    base_path = os.path.join(args.run_dir, 'p0_champ_val_counts.json')
    with open(base_path, 'r', encoding='utf-8') as f:
        base = json.load(f)
    with open(out, 'r', encoding='utf-8') as f:
        cand = json.load(f)
    print('verified baseline val mIoU %.6f  %s'
          % (base['miou'], {k: round(v, 6) for k, v in base['per_class'].items()}))

    diff_imgs, diff_cells = [], 0
    for stem in sorted(base['per_image']):
        b = base['per_image'][stem]
        c = cand['per_image'].get(stem)
        if c is None:
            diff_imgs.append(stem)
            continue
        for cls in ('EX', 'HE', 'SE', 'MA'):
            if list(b[cls]) != list(c[cls]):
                diff_cells += 1
                if stem not in diff_imgs:
                    diff_imgs.append(stem)
    verdict = ('IDENTICAL -> candidate IS the D student (functional match)'
               if diff_imgs == [] else
               'DIFFERENT -> candidate is NOT the D student')
    report = {'candidate': args.candidate, 'candidate_miou': miou,
              'baseline_miou': base['miou'],
              'candidate_per_class': per_class,
              'baseline_per_class': base['per_class'],
              'n_images_with_any_count_difference': len(diff_imgs),
              'n_differing_cells': diff_cells,
              'first_differing_images': diff_imgs[:10],
              'verdict': verdict}
    with open(os.path.join(args.run_dir, '%s_probe_verdict.json' % args.tag),
              'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, sort_keys=True)
    print('\nVERDICT: %s' % verdict)
    print('differing images: %d/%d, differing cells: %d'
          % (len(diff_imgs), len(base['per_image']), diff_cells))


if __name__ == '__main__':
    main()
