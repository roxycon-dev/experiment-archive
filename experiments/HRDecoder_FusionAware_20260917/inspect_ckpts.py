# -*- coding: utf-8 -*-
"""Inspect local checkpoint key structures to pin down champion member load
modes (full vs student_only) before deployment.  Read-only."""
import torch

FILES = {
    'E_1500 (champion E)': r'D:\dataset\experiment\HRDecoder_FusionAware_20260917\assets\E_1500.pth',
    'D_student_only_2000 (student init)': r'D:\dataset\experiment\hrdecoder_plus3\results\D_0913_retry\backup_student_2000.pth',
    'b_weights (B / reference source)': r'D:\dataset\experiment\hrdecoder_plus3\repo\weights\b_weights.pth',
    'h4_best (MA head)': r'D:\dataset\experiment\HRDecoder_审查轮_20260916\audit_runs\audit_20260916_A\h4_best.pth',
    'e4_reference_weights (E4 fallback)': r'D:\dataset\experiment\autodl100\results\e4_reference_weights.pth',
}

def summarize(tag, path):
    print('=' * 72)
    print('%s -> %s' % (tag, path))
    try:
        ck = torch.load(path, map_location='cpu', weights_only=False)
    except Exception as e:
        print('  LOAD FAILED: %r' % e)
        return
    print('  top-level type: %s' % type(ck).__name__)
    if not isinstance(ck, dict):
        return
    top = list(ck.keys())
    print('  top-level keys (%d): %s' % (len(top), top[:12]))
    sd = ck.get('state_dict', ck)
    if not isinstance(sd, dict) or not all(isinstance(v, torch.Tensor) for v in sd.values()):
        # e.g. h4_best has nested 'head'
        for k in top:
            v = ck[k]
            if isinstance(v, dict) and v and all(isinstance(x, torch.Tensor) for x in v.values()):
                print('  nested tensor-dict %r: %d tensors, first keys %s' %
                      (k, len(v), sorted(v.keys())[:6]))
        return
    keys = sorted(sd.keys())
    n_ref = sum(1 for k in keys if k.startswith('reference.'))
    prefixes = {}
    for k in keys:
        p = k.split('.')[0]
        prefixes[p] = prefixes.get(p, 0) + 1
    print('  state_dict: %d tensors, reference.* = %d' % (len(keys), n_ref))
    print('  top prefixes: %s' % dict(sorted(prefixes.items(), key=lambda x: -x[1])))
    print('  first keys: %s' % keys[:5])
    if n_ref:
        refk = [k for k in keys if k.startswith('reference.')]
        print('  first reference keys: %s' % refk[:3])
    # any suspicious extra modules (heads etc.)
    extras = [p for p in prefixes if p not in
              ('backbone', 'decode_head', 'reference', 'detail_branch', 'detail_head',
               'auxiliary_head', 'ma_head', 'corrector', 'neck')]
    if extras:
        print('  EXTRA prefixes: %s' % extras)

for tag, p in FILES.items():
    summarize(tag, p)
