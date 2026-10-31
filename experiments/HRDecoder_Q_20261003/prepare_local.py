"""Only experiment preparation: data identity and the mandatory P0 metric check."""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from qlib.common import CLASSES, digest, read_json, sha, utc_now, write_json
from qlib.metrics import counts, summarize


ROOT = Path(__file__).resolve().parent


def data_identity(runtime):
    root = Path(runtime['ddr_root'])
    old = Path(runtime['historical_root'])
    contract = read_json(old/'configs/val_contract.json')
    splits = {}
    for split, expected in (('train', 454), ('val', 76), ('test', 227)):
        images = sorted((root/'images'/split).glob('*.jpg'))
        if len(images) != expected:
            raise ValueError(f'Unexpected split size: {split} {len(images)}')
        rows = []
        for image in images:
            label = root/'labels'/split/(image.stem+'.png')
            with Image.open(image) as im:
                hw = [im.height, im.width]
            label_sha = sha(label)
            if split == 'val' and label_sha != contract['label_files_sha256'][image.stem]:
                raise ValueError('Historical label SHA changed: '+image.stem)
            rows.append({'sid': image.stem, 'hw': hw, 'image_sha256': sha(image), 'label_sha256': label_sha,
                         'image_bytes': image.stat().st_size, 'label_bytes': label.stat().st_size})
        if split == 'val' and [r['sid'] for r in rows] != contract['stems']:
            raise ValueError('Historical validation IDs changed')
        splits[split] = {'samples': rows, 'sha256': digest(rows), 'n_images': len(rows)}
        print(f'Identity locked: {split} {len(rows)}; test label pixels not decoded', flush=True)
    ids = [{r['sid'] for r in splits[s]['samples']} for s in ('train', 'val', 'test')]
    if any(ids[i] & ids[j] for i in range(3) for j in range(i+1, 3)):
        raise ValueError('DDR split ID overlap')
    result = {'status': 'PASS', 'classes': CLASSES, 'splits': splits, 'test_label_pixel_access': False,
              'assignment_sha256': sha(root/'split_assignment.json'), 'created_at': utc_now()}
    result['sha256'] = digest(splits)
    write_json(ROOT/'reports/data_identity.json', result)
    return result


def reproduce_p0(runtime, identity):
    old = Path(runtime['historical_root'])
    expected = read_json(old/'server_results/work/G/current_deployment/metrics.json')
    c_cache = old/'server_results/work/G/champion'
    u_cache = old/'server_results/work/G/U0_native_whole'
    rows, hashes = [], []
    for i, sample in enumerate(identity['splits']['val']['samples']):
        sid = sample['sid']
        with Image.open(Path(runtime['ddr_root'])/'labels/val'/(sid+'.png')) as im:
            lab = np.asarray(im)
        if not np.isin(lab, [0, 1, 2, 3, 4, 255]).all():
            raise ValueError('Uncontracted validation label')
        cp, up = c_cache/(sid+'.npz'), u_cache/(sid+'.npz')
        with np.load(cp, allow_pickle=False) as f:
            c = f['prob']
        with np.load(up, allow_pickle=False) as f:
            u = f['prob']
        if c.dtype != np.float32 or u.dtype != np.float32 or c.shape != (4, *lab.shape) or c.shape != u.shape:
            raise ValueError('P0 probability identity/dtype/coordinate mismatch')
        g = np.stack([lab == k for k in range(1, 5)])
        v = np.broadcast_to(lab != 255, g.shape)
        # Preserve the archived FP32 arithmetic order, with one strict >.5.
        ct = counts(.75*c + .25*u, g, v)
        rows.append({'sid': sid, 'counts': ct.tolist()})
        hashes.append({'sid': sid, 'C_sha256': sha(cp), 'U0_sha256': sha(up)})
        del c, u, g, v, lab
        if (i+1) % 10 == 0:
            print(f'P0 exact metric acceptance: {i+1}/76', flush=True)
    aggregate = np.asarray([r['counts'] for r in rows]).sum(0)
    if not np.array_equal(aggregate, expected['counts']):
        raise ValueError('P0 aggregate counts differ from archived protocol')
    original = {r['sid']: r['counts'] for r in expected['per_image']}
    if any(r['counts'] != original[r['sid']] for r in rows):
        raise ValueError('P0 per-image counts differ')
    result = {'status': 'PASS', **summarize(aggregate), 'per_image': rows, 'cache_hashes': hashes,
              'formula': '0.75*C+0.25*U0_native_whole', 'threshold': .5, 'strict_greater': True,
              'coordinates': 'original_image', 'created_at': utc_now(), 'test_read': False}
    write_json(ROOT/'reports/p0_acceptance.json', result)
    champion = read_json(old/'configs/champion_resolved.json')
    write_json(ROOT/'configs/p0_identity.json', {'champion': champion['champion'],
        'historical_lock_sha256': sha(old/'configs/original_deployment_lock.json'),
        'champion_config_sha256': sha(old/'configs/champion_resolved.json'),
        'p0': '0.75*C+0.25*U0_historical_native_whole',
        'q_fusion': '0.5625*C+0.1875*U0_historical_native_whole+0.25*Q',
        'H4_is_a_learned_residual_head_not_a_scalar_weight': True, 'p0_acceptance_sha256': sha(ROOT/'reports/p0_acceptance.json')})
    print(f'P0 PASS: {result["miou_pct"]:.10f}% with exact per-image TP/FP/FN', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--identity-only', action='store_true')
    args = parser.parse_args()
    runtime = read_json(ROOT/'configs/runtime.local.json')
    identity = data_identity(runtime)
    if not args.identity_only:
        reproduce_p0(runtime, identity)


if __name__ == '__main__':
    main()
