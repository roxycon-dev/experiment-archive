"""One finite finalization dependency; no early selection or test retries."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'reports'))
from qlib.common import digest, read_json, sha, utc_now, write_json
import handoff_after_qmix16 as remote
from continuation_final_evaluation import EXPECTED_IDENTITIES

STATE = ROOT / 'reports/continuation_final_handoff_state.json'
started = time.monotonic()
owns_state = False
resume_review = None


def state(stage, **fields):
    write_json(STATE, dict(stage=stage, updated_at=utc_now(), seconds=time.monotonic()-started,
                          automatic_restart=False, resume_review=resume_review, **fields))


def local_backup_complete(name):
    path = ROOT / ('reports/run_backup_%s_24k.json' % name)
    if not path.exists():
        return False
    backup = read_json(path)
    assert backup['complete'] and backup['run'] == name and backup['endpoint'] == 24000
    assert len(backup['files']) == 327
    files = {row['path']: row for row in backup['files']}
    for filename in ('identity.json', 'status.json'):
        relative = 'runs/%s/%s' % (name, filename)
        saved = ROOT / 'server_results' / relative
        assert sha(saved) == files[relative]['sha256']
    saved_identity = read_json(ROOT / 'server_results/runs' / name / 'identity.json')
    saved_status = read_json(ROOT / 'server_results/runs' / name / 'status.json')
    assert digest(saved_identity) == saved_status['identity_sha256'] == EXPECTED_IDENTITIES[name]
    assert saved_status['stage'] == 'COMPLETE'
    assert saved_status['completed'] == saved_status['endpoint'] == 24000
    return True


def main():
    global owns_state, resume_review
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume-reviewed', action='store_true')
    args = parser.parse_args()
    if args.resume_reviewed:
        review = read_json(ROOT / 'reports/continuation_final_handoff_repair_receipt.json')
        assert review['explicit_resume_once'] and review['prior_worker_confirmed_exited']
        assert sha(STATE) == review['failed_state_sha256']
        assert read_json(STATE)['stage'] == 'TECHNICAL_PAUSE_FINALIZATION_HANDOFF'
        assert read_json(STATE)['reason'] == 'TimeoutError: timed out'
        assert sha(Path(__file__)) == review['repaired_worker_sha256']
        resume_review = 'reports/continuation_final_handoff_repair_receipt.json'
        deadline_utc = datetime.fromisoformat(review['original_deadline_utc'])
    else:
        assert not STATE.exists(), 'Inspect the existing dependency worker; never dispatch another'
        deadline_utc = datetime.now(timezone.utc).replace(microsecond=0)
        from datetime import timedelta
        deadline_utc += timedelta(minutes=180)
    owns_state = True
    expected_files = read_json(ROOT / 'reports/continuation_finalization_preparation.json')['deployed_files']
    while True:
        # The existing export worker already tracks the registered remote run.
        # Wait on its local receipt instead of opening a second SSH poll loop.
        export = read_json(ROOT / 'reports/mix2024_late_offload_state.json')
        if export['stage'].startswith('TECHNICAL_PAUSE'):
            raise RuntimeError('Registered late export paused: ' + str(export))
        current = export.get('formal_status')
        if current:
            assert current['identity_sha256'] == EXPECTED_IDENTITIES['Q_MIX_seed2024']
            assert current['endpoint'] == 24000
            if current['stage'] in ('TECHNICAL_PAUSE', 'SERIOUS_FAILURE_STOP'):
                raise RuntimeError('Registered training paused: ' + str(current))
        backed = export['stage'] == 'COMPLETE_RUN_BACKUP_SHA_VERIFIED'
        if backed and all(local_backup_complete(name) for name in EXPECTED_IDENTITIES):
            break
        if datetime.now(timezone.utc) > deadline_utc:
            raise TimeoutError('Finite finalization dependency wait expired; no launch')
        state('WAITING_FOR_COMPLETE_PAIRED_RUN_AND_LOCAL_SHA_BACKUP',
              current_run_status=current, export_stage=export['stage'],
              final_local_backup_complete=backed, deadline_utc=deadline_utc.isoformat(),
              waiting_transport='Existing local export receipt; no duplicate SSH polling')
        time.sleep(30)
    launch = remote.execute('''
import json,pathlib,subprocess,os
from qlib.common import digest,sha,utc_now
r=pathlib.Path('/root/autodl-tmp/HRDecoder_Q_20261003');os.chdir(r)
expected=json.loads(%r);identities=json.loads(%r)
for path,value in expected.items():assert sha(r/path)==value
for name,identity in identities.items():
 run=r/'runs'/name;s=json.loads((run/'status.json').read_text())
 assert s['stage']=='COMPLETE' and s['completed']==s['endpoint']==24000
 assert s['identity_sha256']==identity==digest(json.loads((run/'identity.json').read_text()))
pid=(r/'reports/formal.pid').read_text().strip();command=pathlib.Path('/proc')/pid/'cmdline'
assert not command.exists() or b'train_run.py' not in command.read_bytes()
assert not (r/'reports/continuation_final_launch_receipt.json').exists(), 'A finalization launch already exists'
assert not (r/'reports/continuation_final_selection.json').exists(), 'A frozen choice already exists'
assert not (r/'reports/continuation_final_evaluation_state.json').exists(), 'A test ledger already exists'
py='/root/miniconda3/envs/hrdecoder_q_20261003/bin/python'
subprocess.run([py,'reports/advance_qmix_campaign.py'],check=True,stdout=subprocess.DEVNULL)
with (r/'reports/continuation_final_evaluation.log').open('wb') as log:
 p=subprocess.Popen([py,'-u','reports/run_continuation_locked_test.py'],cwd=str(r),
  stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
  env={**os.environ,'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4'})
receipt=dict(stage='REGISTERED_FINALIZATION_PROCESS_LAUNCHED',pid=p.pid,created_at=utc_now(),
 prerequisites='Four complete common24k runs and complete local MIX2024 SHA backup',
 once_only=True,automatic_restart=False)
(r/'reports/continuation_final_launch_receipt.json').write_text(json.dumps(receipt,indent=2))
print(json.dumps(receipt))
''' % (json.dumps(expected_files), json.dumps(EXPECTED_IDENTITIES)))
    write_json(ROOT / 'reports/continuation_final_launch_receipt.json', launch)
    state('FINALIZATION_LAUNCHED_ONCE', launch=launch)


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        if owns_state:
            state('TECHNICAL_PAUSE_FINALIZATION_HANDOFF', reason='%s: %s' % (type(error).__name__, error))
        raise
