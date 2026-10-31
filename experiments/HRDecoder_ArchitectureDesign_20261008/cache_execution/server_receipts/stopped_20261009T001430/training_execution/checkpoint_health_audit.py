"""CPU-only finite-value audit; retain the observed checkpoint via a hard link."""
import gc
import json
import math
import os
from pathlib import Path
import time
import torch

torch.set_num_threads(1)
ROOT = Path('/root/autodl-tmp/HRDecoder_v3_20261008')
OUT = ROOT / 'reports/diagnostics'
OUT.mkdir(parents=True, exist_ok=True)

def inspect(path, retain=False):
    before = path.stat()
    checkpoint = torch.load(path, map_location='cpu')
    state = checkpoint['model']
    floating = 0
    bad = []
    variances = []
    output_layers = []
    for name, value in state.items():
        if not torch.is_tensor(value) or not torch.is_floating_point(value):
            continue
        floating += 1
        if not bool(torch.isfinite(value).all()):
            bad.append(name)
            continue
        if name.endswith('running_var'):
            variances.append({'name': name, 'min': float(value.min()), 'max': float(value.max()),
                              'mean': float(value.mean()), 'negative': int((value < 0).sum())})
        if name.endswith(('conv_seg.bias', 'conv_seg.weight', 'fusion.final_conv.bias', 'fusion.final_conv.weight')):
            channel = value.reshape(value.shape[0], -1)
            output_layers.append({'name': name, 'shape': list(value.shape),
                                  'per_output_min': channel.min(1).values.tolist(),
                                  'per_output_max': channel.max(1).values.tolist(),
                                  'per_output_norm': channel.norm(dim=1).tolist()})
    optimizer_bad = []
    for index, values in checkpoint.get('optimizer', {}).get('state', {}).items():
        for name, value in values.items():
            if torch.is_tensor(value) and torch.is_floating_point(value) and not bool(torch.isfinite(value).all()):
                optimizer_bad.append([index, name])
    retained = None
    if retain and path.stat().st_ino == before.st_ino:
        retained = OUT / ('checkpoint_%05d.pt' % checkpoint['completed'])
        if not retained.exists():
            os.link(str(path), str(retained))
    result = {'source': str(path), 'completed': checkpoint['completed'],
              'identity_SHA': checkpoint['identity_SHA'], 'checkpoint_bytes': before.st_size,
              'floating_model_tensors': floating, 'nonfinite_model_tensors': bad,
              'nonfinite_optimizer_tensors': optimizer_bad, 'BN_buffer_count': len(variances),
              'negative_BN_variances': [v for v in variances if v['negative']],
              'largest_BN_variances': sorted(variances, key=lambda v: v['max'], reverse=True)[:12],
              'smallest_BN_variances': sorted(variances, key=lambda v: v['min'])[:8],
              'output_layer_statistics': output_layers,
              'retained_diagnostic_checkpoint': str(retained) if retained else None}
    result['numeric_health'] = 'PASS' if not (bad or optimizer_bad or result['negative_BN_variances']) else 'FAIL'
    del checkpoint, state
    gc.collect()
    return result

last = inspect(ROOT / 'training_runs/v3/last.pt', retain=True)
best = inspect(ROOT / 'training_runs/v3/best_val.pt')
report = {'time_unix': time.time(), 'device': 'CPU only; no model forward or SGD update',
          'last': last, 'best_val': best, 'no_data_or_cache_hashes_recomputed': True,
          'training_protocol_changed': False,
          'limitation': 'Finite parameters and buffers do not prove finite inference outputs or establish a cause of validation collapse.'}
target = OUT / ('checkpoint_health_%05d.json' % last['completed'])
target.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report))
