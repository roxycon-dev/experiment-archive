"""Separate continuation lock; reuse the immutable original inference math.

Freeze only complete paired validation evidence. An exactly previously tested
identity reuses its counts; a different eligible identity is evaluated once.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'reports'))
from qlib.common import assert_source, digest, read_json, sha, utc_now, write_json
from qlib.decision import lock_selection
import final_evaluation as original

LOCK = ROOT / 'reports/continuation_final_selection.json'
OUT = ROOT / 'reports/continuation_final_test'
STATE = ROOT / 'reports/continuation_final_evaluation_state.json'
PRIOR_LOCK = ROOT / 'reports/final_selection.json'
PRIOR_OUT = ROOT / 'reports/final_test'
RUNNER = ROOT / 'reports/run_continuation_locked_test.py'
PROTOCOL = ROOT / 'reports/qmix_continuation_protocol.json'
EXPECTED_IDENTITIES = {
    'Q_DDR_seed42': 'b3e4cea88c5eeda8923a9c3950c0f67e74f02c28057730f2dd2d19154c500396',
    'Q_DDR_seed2024': 'c25584777fd2ded929bce248f6fc0e05d954ce9ba200f62d3066dbad41cdd252',
    'Q_MIX_seed42': '9803ad79aecd9c3d7a426fda0053ae0e43f1afc663933e30d35364be38800a3e',
    'Q_MIX_seed2024': '60920d33a3d85acd05f0c6637a4ec2442f21f69a6d4ca3beff46b2cb7034adac',
}
_original_gate = original.gate
original.LOCK = LOCK
original.OUT = OUT


def test_action(selection, checkpoints, prior):
    selected = selection['selected']
    if not selected['test_eligible']:
        return 'NO_TEST_NOT_ELIGIBLE'
    previous = prior['selection']['selected']
    same = all(selected[k] == previous[k] for k in ('arm', 'update', 'formula'))
    same = same and all(
        checkpoints[str(seed)]['sha256'] == prior['checkpoints'][str(seed)]['sha256']
        and checkpoints[str(seed)]['identity'] == prior['checkpoints'][str(seed)]['identity']
        for seed in (42, 2024))
    return 'REUSE_EXACT_PRIOR_TEST_COUNTS' if same else 'EVALUATE_NEW_IDENTITY_ONCE'


def contract():
    protocol = read_json(PROTOCOL)
    assert sha(PRIOR_LOCK) == protocol['existing_DDR_final_lock_sha256']
    prior = read_json(PRIOR_LOCK)
    assert sha(ROOT / 'reports/final_evaluation.py') == prior['driver_sha256']
    assert assert_source(ROOT)['sha256'] == protocol['source_sha256']
    assert sha(ROOT / 'configs/runtime.qmix.server.json') == protocol['runtime_qmix_sha256']
    assert sha(ROOT / 'reports/external_source_user_override.json') == protocol['source_override_sha256']
    a = read_json(ROOT / 'configs/runtime.server.json')
    b = read_json(ROOT / 'configs/runtime.qmix.server.json')
    for key in ('resources', 'ddr_root', 'dinov3_repo', 'pretrained_checkpoint'):
        assert a[key] == b[key], 'Paired inference resources changed: ' + key
    assert sha(b['external_manifest']) == protocol['external_manifest_server_sha256']
    return prior, protocol


def complete_runs():
    for name, identity in EXPECTED_IDENTITIES.items():
        run = ROOT / 'runs' / name
        status = read_json(run / 'status.json')
        assert status['stage'] == 'COMPLETE' and status['completed'] == status['endpoint'] == 24000
        assert status['identity_sha256'] == identity == digest(read_json(run / 'identity.json'))
    pid_file = ROOT / 'reports/formal.pid'
    if pid_file.exists():
        command = Path('/proc') / pid_file.read_text().strip() / 'cmdline'
        assert not command.exists() or b'train_run.py' not in command.read_bytes(), 'Formal training has not exited'


def freeze():
    if LOCK.exists():
        gate()
        return read_json(LOCK)
    prior, protocol = contract()
    complete_runs()
    campaign = read_json(ROOT / 'reports/campaign_state.json')
    assert campaign['arms'] == ['Q_DDR', 'Q_MIX'] and campaign['common_endpoint'] == 24000
    assert campaign['review_seed_authorized'] and not campaign['architecture_stopped']
    results, metrics = {}, {}
    for arm in campaign['arms']:
        results[arm] = {}
        for seed in (42, 2024):
            run = ROOT / 'runs' / ('%s_seed%s' % (arm, seed))
            results[arm][seed] = {}
            for path in run.glob('val_u*/metrics.json'):
                update = int(path.parent.name[5:])
                node = read_json(path)
                assert node['update'] == update and node['n_images'] == 76
                results[arm][seed][update] = node
                if 12000 <= update <= 24000:
                    metrics[str(path)] = sha(path)
            assert {12000, 16000, 20000, 24000}.issubset(results[arm][seed]), 'Missing required formal late node'
    selection = lock_selection(results, 24000)
    assert selection == campaign['selection'], 'Original decision writer and freeze disagree'
    selected = selection['selected']
    checkpoints = {}
    for seed in (42, 2024):
        run = ROOT / 'runs' / ('%s_seed%s' % (selected['arm'], seed))
        path = run / ('checkpoint_u%s.pth' % selected['update'])
        checkpoints[str(seed)] = dict(path=str(path), sha256=sha(path), identity=read_json(run / 'identity.json'))
    action = test_action(selection, checkpoints, prior)
    payload = copy.deepcopy(prior)
    payload.update(status='LOCKED_TEST_AUTHORIZED' if selected['test_eligible'] else 'LOCKED_TEST_NOT_ELIGIBLE',
        selection=selection, checkpoints=checkpoints, candidate_metrics_sha256=metrics,
        created_at=utc_now(), adapter_sha256=sha(__file__), runner_sha256=sha(RUNNER),
        continuation_protocol_sha256=sha(PROTOCOL), runtime_qmix_sha256=sha(ROOT / 'configs/runtime.qmix.server.json'),
        source_override_sha256=protocol['source_override_sha256'],
        external_manifest_sha256=protocol['external_manifest_server_sha256'],
        prior_final_selection_sha256=sha(PRIOR_LOCK), test_action=action,
        original_inference_driver_unchanged=True, inference_runtime='configs/runtime.server.json',
        selected_training_runtime='configs/runtime.qmix.server.json' if selected['arm'] == 'Q_MIX' else 'configs/runtime.server.json',
        test_evidence_boundary=protocol['test_evidence_boundary'],
        historical_limitation='Exploratory continuation after historical DDR val/test use and an already completed DDR16k test. Choice uses paired validation only; test never changes it.')
    # Exclusive creation prevents another process from overwriting the choice.
    with os.fdopen(os.open(str(LOCK), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, allow_nan=False)
    gate()
    return payload


def gate():
    prior, protocol = contract()
    lock = read_json(LOCK)
    assert lock['adapter_sha256'] == sha(__file__) and lock['runner_sha256'] == sha(RUNNER)
    assert lock['continuation_protocol_sha256'] == sha(PROTOCOL)
    assert lock['prior_final_selection_sha256'] == sha(PRIOR_LOCK)
    assert lock['runtime_qmix_sha256'] == protocol['runtime_qmix_sha256']
    assert lock['test_action'] == test_action(lock['selection'], lock['checkpoints'], prior)
    complete_runs()
    if lock['test_action'] != 'NO_TEST_NOT_ELIGIBLE':
        return _original_gate()
    assert lock['status'] == 'LOCKED_TEST_NOT_ELIGIBLE' and not lock['selection']['selected']['test_eligible']
    for path, expected in lock['candidate_metrics_sha256'].items():
        assert sha(path) == expected
    for spec in lock['checkpoints'].values():
        assert sha(spec['path']) == spec['sha256']
    return lock, sha(LOCK)


original.gate = gate


def reuse_prior_counts():
    lock, lock_sha = gate()
    assert lock['test_action'] == 'REUSE_EXACT_PRIOR_TEST_COUNTS'
    receipt = read_json(ROOT / 'reports/final_evidence_backup_receipt.json')
    assert receipt['stage'] == 'COMPLETE' and receipt['local_backup_state'] == 'SHA_VERIFIED'
    assert receipt['lock_sha256'] == lock['prior_final_selection_sha256']
    expected = {row['path']: row for row in receipt['files']}
    sources = {}
    for name in ('seed42.json', 'seed2024.json', 'summary.json'):
        path = PRIOR_OUT / name
        row = expected[path.relative_to(ROOT).as_posix()]
        assert path.stat().st_size == row['bytes'] and sha(path) == row['sha256']
        data = read_json(path)
        assert data['status'] == 'COMPLETE' and data['lock_sha256'] == receipt['lock_sha256']
        if name.startswith('seed'):
            assert data['n_images'] == len(data['images']) == 227
        sources[name] = dict(path=str(path), sha256=row['sha256'])
        data.update(lock_sha256=lock_sha, reused_from_prior_lock_sha256=receipt['lock_sha256'],
                    reused_source_sha256=row['sha256'], test_action=lock['test_action'], no_new_test_inference=True)
        if name == 'summary.json':
            data.update(arm=lock['selection']['selected']['arm'], common_endpoint=24000,
                        selected_mean_val_pct=lock['selection']['selected']['mean_val_pct'],
                        historical_limitation=lock['historical_limitation'])
        write_json(OUT / name, data)
    write_json(OUT / 'reuse_receipt.json', dict(status='COMPLETE', lock_sha256=lock_sha,
        original_test_sources=sources, no_new_test_inference=True, completed_at=utc_now()))


def prepare_p0_reuse(lock, lock_sha):
    target = OUT / 'p0_manifest.json'
    if target.exists():
        return
    path = PRIOR_OUT / 'p0_manifest.json'
    if not path.exists():
        return
    receipt = read_json(ROOT / 'reports/final_evidence_backup_receipt.json')
    expected = {row['path']: row for row in receipt['files']}
    assert receipt['lock_sha256'] == lock['prior_final_selection_sha256']
    assert sha(path) == expected[path.relative_to(ROOT).as_posix()]['sha256']
    old = read_json(path)
    assert old['lock_sha256'] == lock['prior_final_selection_sha256'] and old['status'] == 'COMPLETE'
    images = {}
    for sid, row in old['images'].items():
        probability = Path(lock['p0_cache']) / (sid + '.npz')
        if probability.is_file() and sha(probability) == row['sha256']:
            images[sid] = row
    manifest = copy.deepcopy(old)
    manifest.update(lock_sha256=lock_sha, images=images,
        status='COMPLETE' if len(images) == 227 else 'RUNNING',
        reused_historical_P0_manifest_sha256=sha(path), reused_existing_FP32_files=len(images))
    write_json(target, manifest)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', required=True, choices=('freeze', 'p0', 'q'))
    phase = parser.parse_args().phase
    if phase == 'freeze':
        print(json.dumps(dict(test_action=freeze()['test_action'], lock_sha256=sha(LOCK))))
        return
    lock, lock_sha = gate()
    assert lock['test_action'] == 'EVALUATE_NEW_IDENTITY_ONCE', 'A previously tested identity must only reuse its evidence'
    ledger = read_json(STATE)
    assert ledger['stage'] == 'RUNNING' and ledger['lock_sha256'] == lock_sha
    assert ledger['phase'] == phase.upper() and ledger['runner_sha256'] == sha(RUNNER)
    if phase == 'p0':
        prepare_p0_reuse(lock, lock_sha)
        original.p0_phase()
    else:
        original.q_phase()


if __name__ == '__main__':
    main()
