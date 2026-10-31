"""Prepare only assets required by the user's frozen final experiment protocol."""
from pathlib import Path
import hashlib
import json
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT.parents[1] / 'HRDecoder_NextPlan_20261004' / 'execution'
QROOT = ROOT.parents[1] / 'HRDecoder_Q_20261003'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    for name in ('qlib', 'configs', 'reports', 'inputs', 'tests', 'vendor', 'runs'):
        (ROOT / name).mkdir(exist_ok=True)
    copied = {}
    for name in ('__init__.py', 'common.py', 'model.py', 'data.py', 'geometry.py',
                 'loss.py', 'metrics.py', 'evaluate.py', 'checkpoint.py'):
        src = OLD / 'qlib' / name
        dst = ROOT / 'qlib' / name
        if dst.exists():
            assert sha(dst) == sha(src), 'Existing shared core differs; preserve and review'
        else:
            shutil.copyfile(src, dst)
        copied['qlib/' + name] = sha(src)
    for name in ('adaptation.py', 'performance.py'):
        dst = ROOT / ('parent_' + name)
        if not dst.exists():
            shutil.copyfile(OLD / name, dst)
        copied[dst.name] = sha(OLD / name)
    for src, dst in [
        (ROOT.parent / '01_执行协议.json', ROOT / 'configs/protocol.json'),
        (ROOT.parent / '02_空间交互与随机流接口.md', ROOT / 'configs/spatial_interface.md'),
        (OLD / 'reports/data_identity.json', ROOT / 'reports/data_identity.json'),
        (OLD / 'reports/pretraining_identity.json', ROOT / 'reports/pretraining_identity.json'),
        (OLD / 'configs/p0_identity.json', ROOT / 'configs/p0_identity.json'),
        (OLD / 'configs/initial_weights.json', ROOT / 'configs/parent_Q16k.json'),
        (OLD / 'configs/performance.json', ROOT / 'configs/parent_performance.json'),
        (OLD / 'reports/stream_F_seed2024.jsonl', ROOT / 'reports/stream_PARENT_seed2024.jsonl'),
    ]:
        if not dst.exists():
            shutil.copyfile(src, dst)
        assert sha(src) == sha(dst)
    sys.path.insert(0, str(ROOT))
    from qlib.common import read_json, write_json, utc_now
    protocol = read_json(ROOT / 'configs/protocol.json')
    start = protocol['initialization']['seed42']
    assert Path(start['path']).stat().st_size == start['bytes']
    assert sha(start['path']) == start['sha256']
    q2024 = protocol['initialization']['seed2024']
    assert sha(q2024['parent']) == q2024['parent_sha256']
    closure = read_json(OLD / 'reports/closure_receipt.json')
    assert closure['stage'] == 'COMPLETE' and closure['platform_OFF_verified'] is True
    write_json(ROOT / 'reports/inherited_core_identity.json', {
        'files': copied, 'historical_execution': str(OLD),
        'seed42_parent': start, 'seed2024_Q_parent': q2024,
        'old_recovery_complete': True, 'time': utc_now()})
    for variant in ('local', 'server'):
        runtime = read_json(OLD / ('configs/runtime.local.json' if variant == 'local'
                                  else 'server_results/configs/runtime.server.json'))
        runtime['server'].update(host=None, port=None,
                                remote_root='/root/autodl-tmp/HRDecoder_FinalDecision_20261005')
        runtime.update(execution_root=str(ROOT) if variant == 'local' else runtime['server']['remote_root'],
                       final_test_authorized=False, stage='PREPARATION_IN_PROGRESS_NO_SERVER_ASSIGNED',
                       parent_f12_execution=str(OLD) if variant == 'local' else '/root/autodl-tmp/HRDecoder_Finetune_20261004',
                       TJDR_root=str(ROOT / 'inputs/TJDR/train') if variant == 'local' else
                       '/root/autodl-tmp/HRDecoder_FinalDecision_20261005/inputs/TJDR/train')
        runtime['budget'] = {'hourly_rate_cny': None,
                             'execution_authorization': 'User explicitly requested strict execution of the 2026-10-05 final protocol; waits for preparation before opening server.'}
        runtime['resources'] = {'micro_batch': 2, 'accumulation': 4, 'dtype': 'bfloat16',
                                'workers': 8, 'source_workers': 4, 'prefetch_factor': 4,
                                'pin_memory': True, 'shared_cache_limit_bytes': 80 * 1024**3,
                                'source_worker_cache_bytes': 1024**3}
        write_json(ROOT / f'configs/runtime.{variant}.json', runtime)
    write_json(ROOT / 'reports/preparation_status.json', {
        'stage': 'IMPLEMENTATION_AND_TJDR_ACCEPTANCE_PENDING', 'time': utc_now(),
        'parent_SHA_PASS': True, 'protocol_sha256': sha(ROOT / 'configs/protocol.json'),
        'interface_sha256': sha(ROOT / 'configs/spatial_interface.md'),
        'paid_gpu_started': False, 'server_assigned': False,
        'required_gates': ['TJDR terms/map/overlay/dedup', 'zero-step identity',
                           'two-step gradients', 'geometry and RNG resume',
                           'CUDA kernel forward/backward equivalence',
                           '100 complete updates per path and full val timing',
                           '100GiB free disk reserve']})
    print('Inherited parents and evaluation core verified; new execution isolated.', flush=True)


if __name__ == '__main__':
    main()
