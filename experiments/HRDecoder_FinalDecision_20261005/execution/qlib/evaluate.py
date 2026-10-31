import os
import time
from pathlib import Path

import numpy as np
import torch

from .common import sha, write_json
from .geometry import rgb_window, center_logits, tiles
from .metrics import counts, summarize


def load_probability(path, hw):
    with np.load(path, allow_pickle=False) as f:
        p = f['prob']
    if p.dtype != np.float32 or p.shape != (4, *hw) or not np.isfinite(p).all():
        raise ValueError('Historical FP32 probability contract failed')
    return p


def load_p0(runtime, sid, hw):
    if runtime.get('p0_combined_cache'):
        return load_probability(Path(runtime['p0_combined_cache'])/(sid+'.npz'), hw)
    c = load_probability(Path(runtime['p0_c_cache'])/(sid+'.npz'), hw)
    u = load_probability(Path(runtime['p0_u_cache'])/(sid+'.npz'), hw)
    return .75*c + .25*u


@torch.no_grad()
def evaluate(model, dataset, runtime, output, update, keep_probability=False, locked_formula=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if dataset.split == 'test' and not runtime.get('final_test_authorized'):
        raise ValueError('Final test gate absent')
    if dataset.split == 'test' and locked_formula not in ('single', 'fixed_fusion', 'half_fusion'):
        raise ValueError('Test requires one prelocked output formula')
    previous_training = model.training
    model.eval()
    device = next(model.parameters()).device
    dtype = {'bfloat16': torch.bfloat16, 'float16': torch.float16}[runtime['resources']['dtype']]
    rows, probability_hashes = [], []
    started = time.perf_counter()
    for sid in dataset.ids:
        rgb, g, v = dataset.load(sid)
        hw = rgb.shape[:2]
        q = np.empty((4, *hw), np.float32)
        image_started = time.perf_counter()
        for tile in tiles(*hw):
            x = torch.from_numpy(rgb_window(rgb, tile)).unsqueeze(0).to(device)
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=device.type == 'cuda'):
                z = center_logits(model(x))
            p = z.float().sigmoid()[0, :, :tile.height, :tile.width].cpu().numpy()
            q[:, tile.y:tile.y+tile.height, tile.x:tile.x+tile.width] = p
        p0 = load_p0(runtime, sid, hw)
        gt, valid = g.transpose(2, 0, 1), v.transpose(2, 0, 1)
        predictions = {'single': q, 'fixed_fusion': .75*p0+.25*q,
                       'half_fusion': .5*p0+.5*q, 'p0': p0}
        formulas = (locked_formula,) if dataset.split == 'test' else tuple(predictions)
        row = {'sid': sid, 'q_inference_seconds': time.perf_counter()-image_started}
        row.update({formula: counts(predictions[formula], gt, valid).tolist() for formula in formulas})
        rows.append(row)
        if keep_probability:
            path = output/(sid+'.npz')
            temporary = path.with_suffix('.npz.tmp')
            with temporary.open('wb') as f:
                np.savez_compressed(f, prob=q)
            os.replace(temporary, path)
            probability_hashes.append({'sid': sid, 'sha256': sha(path), 'dtype': 'float32', 'shape': list(q.shape)})
        write_json(output/'progress.json', {'completed_images': len(rows), 'expected': len(dataset.ids), 'update': update})
    result = {'update': update, 'split': dataset.split, 'n_images': len(rows), 'per_image': rows,
              'seconds': time.perf_counter()-started, 'threshold': .5, 'strict_greater': True,
              'coordinates': 'original_image', 'probability_dtype': 'float32',
              'probability_files': probability_hashes}
    for formula in formulas:
        result[formula] = summarize(np.asarray([r[formula] for r in rows]).sum(axis=0))
    if dataset.split == 'test':
        result['locked_formula'] = locked_formula
    write_json(output/'metrics.json', result)
    model.train(previous_training)
    return result
