"""Recover ten immutable P0 validation files; no inference or model selection."""
import json
import shlex
import sys
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ops.remote import connect, command
from qlib.common import read_json, write_json, sha, utc_now
from qlib.metrics import counts


def main():
    plan = read_json(ROOT / 'reports/publication_server_evidence_plan.json')
    runtime = read_json(ROOT / 'configs/runtime.local.json')
    cases = plan['cases']
    assert len(cases) == 5 and plan['no_test_inference_or_training']
    expected = [{'sid': x['sid'], 'component': component,
                 'remote_path': prefix + '/' + x['sid'] + '.npz'}
                for x in cases for component, prefix in plan['p0_source_directories'].items()]
    code = """from pathlib import Path
import hashlib,json
rows=json.loads(%r)
for row in rows:
 p=Path(row['remote_path']);assert p.is_file()
 assert p.parent in [Path('/root/autodl-tmp/HRDecoder_Ready_20261002/cache/val/champion'),Path('/root/autodl-tmp/HRDecoder_Ready_20261002/cache/val/U0')]
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 row.update(sha256=h.hexdigest(),bytes=p.stat().st_size)
assert sum(x['bytes'] for x in rows)<600*1024**2
print(json.dumps(rows))
""" % json.dumps(expected)
    output = ROOT / 'report_evidence/saved_val_p0'
    with connect() as client:
        inventory = json.loads(command(client, shlex.quote(runtime['server_python']) + ' -c ' + shlex.quote(code), 120))
        write_json(ROOT / 'reports/publication_server_evidence_inventory.json', {'files': inventory, 'time': utc_now(), 'plan_sha256': sha(ROOT / 'reports/publication_server_evidence_plan.json')})
        with client.open_sftp() as sftp:
            for i, row in enumerate(inventory):
                target = output / row['component'] / (row['sid'] + '.npz')
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.exists() or sha(target) != row['sha256']:
                    temporary = target.with_suffix('.npz.download')
                    sftp.get(row['remote_path'], str(temporary), prefetch=True, max_concurrent_prefetch_requests=64)
                    assert sha(temporary) == row['sha256']
                    temporary.replace(target)
                row['local_path'] = str(target)
                write_json(ROOT / 'reports/publication_server_evidence_transfer.json', {'stage': 'READ_ONLY_VALIDATION_CUSTODY', 'verified_files': i+1, 'expected_files': len(inventory), 'time': utc_now()})
    metrics = read_json(ROOT / 'server_results/runs/S_seed42/val_u20000/metrics.json')
    rows = {x['sid']: x for x in metrics['per_image']}
    identity = read_json(ROOT / 'reports/data_identity.json')
    data = {x['sid']: x for x in identity['splits']['val']['samples']}
    verified = []
    for case in cases:
        sid = case['sid']
        label_path = Path(runtime['ddr_root']) / 'labels/val' / (sid + '.png')
        assert sha(label_path) == data[sid]['label_sha256']
        with Image.open(label_path) as image:
            label = np.asarray(image)
        gt = np.stack([label == c for c in range(1, 5)])
        valid = np.broadcast_to(label != 255, gt.shape)
        probabilities = []
        for component in ('champion', 'U0'):
            with np.load(output / component / (sid + '.npz'), allow_pickle=False) as f:
                prob = f['prob']
            assert prob.dtype == np.float32 and prob.shape == gt.shape and np.isfinite(prob).all()
            probabilities.append(prob)
        p0 = .75*probabilities[0] + .25*probabilities[1]
        with np.load(ROOT / 'server_results/runs/S_seed42/val_u20000' / (sid+'.npz'), allow_pickle=False) as f:
            q = f['prob']
        actual_p0 = counts(p0, gt, valid).tolist()
        actual_locked = counts(.75*p0+.25*q, gt, valid).tolist()
        assert actual_p0 == rows[sid]['p0'] and actual_locked == rows[sid]['fixed_fusion'], sid
        verified.append({'sid': sid, 'label_sha256': sha(label_path), 'exact_saved_p0_counts': actual_p0,
                         'exact_locked_validation_counts': actual_locked})
    receipt = {'stage': 'REQUIRED_SAVED_VALIDATION_EVIDENCE_SHA_AND_COUNTS_VERIFIED', 'time': utc_now(),
               'plan_sha256': sha(ROOT / 'reports/publication_server_evidence_plan.json'),
               'selection_lock_sha256': plan['lock_sha256'], 'files': inventory, 'case_count_verifications': verified,
               'no_inference_training_threshold_scan_or_selection': True, 'no_test_images_used': True,
               'all_outputs_recovered_with_SHA': True}
    write_json(ROOT / 'reports/publication_server_evidence_recovery.json', receipt)
    print(json.dumps({'stage': receipt['stage'], 'files': len(inventory), 'bytes': sum(x['bytes'] for x in inventory)}), flush=True)


if __name__ == '__main__':
    main()
