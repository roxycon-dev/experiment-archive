"""One-shot evidence gate; original attendants retain full backup/shutdown."""
import base64
import json
import os
import shlex
import sys
import time
import traceback
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ops.remote import connect, command
from qlib.common import read_json, write_json, sha, utc_now, digest
from qlib.metrics import summarize

STATE = ROOT / 'reports/publication_server_release_worker_state.json'
LOCK = '3146ca1edafdffdae7bf4987524bddc502209929e5d6d5b0e7ba49f12b6bd66e'
SCOPE = '9a936d4ea5d2f21d7233c469d1024a7e2bde44902d0c5c7853976e430dda8e6d'
SCIENCE = '937565a59a9d65f46d891cdf456a8bc75141434e2815bb875dce316b5d5d8629'


def state(stage, **values):
    write_json(STATE, {'stage': stage, 'pid': os.getpid(), 'time': utc_now(),
                      'no_new_training_inference_or_selection': True, **values})


def main():
    runtime = read_json(ROOT / 'configs/runtime.local.json')
    evidence = read_json(ROOT / 'reports/publication_server_evidence_recovery.json')
    assert evidence['stage'] == 'REQUIRED_SAVED_VALIDATION_EVIDENCE_SHA_AND_COUNTS_VERIFIED'
    assert evidence['all_outputs_recovered_with_SHA'] and evidence['selection_lock_sha256'] == LOCK
    assert sha(ROOT / 'reports/publication_report_scope.json') == SCOPE
    for row in evidence['files']:
        assert sha(ROOT / 'server_results' / row['path']) == row['sha256']
    # Seal already verified scientific custody. The attendants still perform
    # the final SHA-union check of every raw/EMA/optimizer/RNG artifact.
    custody = []
    for arm in ('C', 'S', 'D', 'SD'):
        for update in (20000, 32000):
            path = ROOT / f'reports/storage_offload/{arm}_seed42_u{update}.json'
            receipt = read_json(path)
            assert receipt['stage'] == 'LOCAL_SHA_VERIFIED_REMOTE_RELEASED' and len(receipt['files']) == 77
            for row in receipt['files']:
                p = ROOT / 'server_results' / row['path']
                assert p.is_file() and p.stat().st_size == row['bytes']
            custody.append({'receipt': str(path), 'sha256': sha(path), 'run': arm, 'update': update})
    required = [
        'reports/final_selection_lock.json', 'reports/decision.json',
        'reports/final_test.json', 'reports/final_test_state.json',
        'reports/最终实验报告.md', 'reports/parallel_global_final_report_mirrored.json',
        'reports/test_p0/p0_manifest.json', 'runs/LOCKED_TEST_seed42/test_identity.json',
        'runs/LOCKED_TEST_seed42/metrics.json', 'reports/source_identity.json',
        'reports/publication_report_scope.json', 'configs/runtime.server.json',
        'reports/publication_server_evidence/derivation.json',
    ] + [x['path'] for x in evidence['files']]
    code = """from pathlib import Path
import hashlib,json
r=Path(%r);driver=json.loads((r/'reports/driver_state.json').read_text());out={'driver':driver}
if driver['stage']=='POSTTRAIN_REPORT_WORK_PRIMARY_HELD':
 def h(p):
  d=hashlib.sha256()
  with p.open('rb') as f:
   for b in iter(lambda:f.read(8*1024**2),b''):d.update(b)
  return d.hexdigest()
 paths=json.loads(%r)
 out['files']=[{'path':x,'bytes':(r/x).stat().st_size,'sha256':h(r/x)} for x in paths]
 out['run_stages']={a:json.loads((r/f'runs/{a}_seed42/status.json').read_text())['stage'] for a in ['C','S','D','SD']}
 out['final_test_state']=json.loads((r/'reports/final_test_state.json').read_text())
 out['source']=json.loads((r/'reports/source_identity.json').read_text())
 out['scope_sha256']=h(r/'reports/publication_report_scope.json')
print(json.dumps(out))
""" % (runtime['server']['remote_root'], json.dumps(required))
    started = time.monotonic()
    state('WAITING_FOR_ORIGINAL_LOCKED_TEST_AND_GLOBAL_ACK', writing_report_after_servers_closed=True)
    while True:
        assert time.monotonic() - started < 12*3600, 'Diagnose the original finalizer; never bypass its gates'
        with connect() as client:
            remote = json.loads(command(client, shlex.quote(runtime['server_python'])+' -c '+shlex.quote(code), 120))
        stage = remote['driver']['stage']
        assert stage != 'TECHNICAL_PAUSE', remote['driver']
        if stage != 'POSTTRAIN_REPORT_WORK_PRIMARY_HELD':
            time.sleep(30)
            continue
        assert all(x == 'COMPLETE' for x in remote['run_stages'].values())
        assert remote['source']['sha256'] == SCIENCE and remote['scope_sha256'] == SCOPE
        assert remote['final_test_state']['stage'] == 'COMPLETE' and remote['final_test_state']['lock_sha256'] == LOCK
        missing = [x['path'] for x in remote['files'] if not (ROOT/'server_results'/x['path']).is_file()
                   or sha(ROOT/'server_results'/x['path']) != x['sha256']]
        if missing:
            state('WAITING_FOR_EXISTING_RECEIVER_SHA_RECOVERY', missing_paths=missing,
                  no_duplicate_mirror_or_download=True)
            time.sleep(30)
            continue
        break
    local = ROOT / 'server_results'
    lock = read_json(local/'reports/final_selection_lock.json')
    assert lock['lock_sha256'] == LOCK and digest({k:v for k,v in lock.items() if k != 'lock_sha256'}) == LOCK
    assert (lock['arm'], lock['update'], lock['formula'], lock['seed_scope']) == ('S', 20000, 'fixed_fusion', [42])
    selected = lock['checkpoints']['42']
    assert sha(local/selected['path']) == selected['sha256']
    metrics = read_json(local/'runs/LOCKED_TEST_seed42/metrics.json')
    final = read_json(local/'reports/final_test.json')
    reference = read_json(ROOT/'configs/R_test_counts_seed42.json')
    assert metrics['split'] == 'test' and metrics['n_images'] == 227
    assert metrics['threshold'] == .5 and metrics['strict_greater'] and metrics['locked_formula'] == 'fixed_fusion'
    assert [r['sid'] for r in metrics['per_image']] == [r['sid'] for r in reference['per_image']]
    counts = np.asarray([r['fixed_fusion'] for r in metrics['per_image']], dtype=np.int64)
    assert counts.shape == (227, 4, 3) and (counts >= 0).all()
    score = summarize(counts.sum(axis=0))['miou_pct']
    assert abs(score - metrics['fixed_fusion']['miou_pct']) < 1e-10
    assert final['stage'] == 'COMPLETE' and final['lock_sha256'] == LOCK
    assert abs(score - final['seeds']['42']['miou_pct']) < 1e-10
    ack = read_json(local/'reports/parallel_global_final_report_mirrored.json')
    assert ack['stage'] == 'BOTH_NODES_GLOBAL_REPORT_SHA_VERIFIED'
    assert ack['report_sha256'] == sha(local/'reports/最终实验报告.md')
    p0 = read_json(local/'reports/test_p0/p0_manifest.json')
    assert p0['status'] == 'COMPLETE' and len(p0['images']) == 227
    proof = {'stage': 'SERVER_NONTRAINING_EVIDENCE_WORK_COMPLETE', 'time': utc_now(),
             'scope_sha256': SCOPE, 'all_outputs_recovered_with_SHA': True,
             'no_new_training_or_selection_or_threshold_scan': True,
             'selection_lock_sha256': LOCK, 'selected_checkpoint_sha256': selected['sha256'],
             'exact_locked_test_count_recomputation_miou_pct': score,
             'recovered_evidence_files': remote['files'], 'sealed_endpoint_custody': custody,
             'saved_validation_evidence_recovery_sha256': sha(ROOT/'reports/publication_server_evidence_recovery.json'),
             'write_and_render_report_locally_after_servers_closed': True,
             'existing_attendants_still_own_final_full_SHA_union_backup_and_exactly_once_shutdown': True}
    write_json(ROOT/'reports/publication_report_server_release.json', proof)
    payload = base64.b64encode((ROOT/'reports/publication_report_server_release.json').read_bytes()).decode('ascii')
    release_code = """from pathlib import Path
import base64,json,hashlib,os
r=Path(%r);raw=base64.b64decode(%r);proof=json.loads(raw)
assert json.loads((r/'reports/driver_state.json').read_text())['stage']=='POSTTRAIN_REPORT_WORK_PRIMARY_HELD'
assert hashlib.sha256((r/'reports/publication_report_scope.json').read_bytes()).hexdigest()==proof['scope_sha256']
assert json.loads((r/'reports/final_test_state.json').read_text())['stage']=='COMPLETE'
assert json.loads((r/'reports/final_selection_lock.json').read_text())['lock_sha256']==proof['selection_lock_sha256']
for row in proof['recovered_evidence_files']:
 assert hashlib.sha256((r/row['path']).read_bytes()).hexdigest()==row['sha256'],row['path']
p=r/'reports/publication_report_server_release.json';assert not p.exists(),'Do not duplicate release'
tmp=p.with_suffix('.tmp');tmp.write_bytes(raw);os.replace(tmp,p)
print(json.dumps({'stage':'AUTHENTICATED_REPORT_EVIDENCE_RELEASE_DISPATCHED','sha256':hashlib.sha256(p.read_bytes()).hexdigest()}))
""" % (runtime['server']['remote_root'], payload)
    with connect() as client:
        dispatch = json.loads(command(client, shlex.quote(runtime['server_python'])+' -c '+shlex.quote(release_code), 120))
    assert dispatch['sha256'] == sha(ROOT/'reports/publication_report_server_release.json')
    write_json(ROOT/'reports/publication_server_release_dispatch.json', dispatch)
    state('SERVER_EVIDENCE_RELEASED_ORIGINAL_BACKUP_AND_SHUTDOWN_GATES_CONTINUE', exact_test_miou_pct=score,
          dispatch=dispatch, no_shutdown_was_issued_by_this_worker=True)


if __name__ == '__main__':
    sys.stdout = (ROOT/'reports/publication_server_release_worker.log').open('a', encoding='utf-8', buffering=1)
    sys.stderr = (ROOT/'reports/publication_server_release_worker.err.log').open('a', encoding='utf-8', buffering=1)
    try:
        main()
    except BaseException as e:
        state('SERVER_EVIDENCE_RELEASE_REVIEW_REQUIRED', reason=str(e), automatic_bypass=False)
        traceback.print_exc()
        raise
