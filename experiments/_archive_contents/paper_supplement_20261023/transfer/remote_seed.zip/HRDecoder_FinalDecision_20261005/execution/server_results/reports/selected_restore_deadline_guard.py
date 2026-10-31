"""Preserve the same waiting finalizer if a slow locked-weight upload nears 1h."""
import hashlib
import json
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PID = 44281
LOCK = '3146ca1edafdffdae7bf4987524bddc502209929e5d6d5b0e7ba49f12b6bd66e'
STATE = ROOT / 'reports/selected_restore_deadline_guard_state.json'


def read(path):
    return json.loads(path.read_text())


def record(stage, **values):
    path = STATE.with_suffix('.tmp')
    path.write_text(json.dumps({'stage': stage, 'time': datetime.now(timezone.utc).isoformat(),
                                'guard_pid': os.getpid(), 'same_finalizer_pid': PID,
                                'no_new_training_or_test_process': True, **values}))
    os.replace(path, STATE)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def owner():
    cmd = Path(f'/proc/{PID}/cmdline').read_bytes()
    assert b'reports/primary_single_seed_boundary.py' in cmd
    assert Path(f'/proc/{PID}/cwd').resolve() == ROOT
    assert read(ROOT / 'reports/driver_state.json')['pid'] == PID


def main():
    lock = read(ROOT / 'reports/final_selection_lock.json')
    request = read(ROOT / 'reports/selected_restore_request.json')
    assert lock['lock_sha256'] == request['lock_sha256'] == LOCK
    assert lock['source_sha256'] == '937565a59a9d65f46d891cdf456a8bc75141434e2815bb875dce316b5d5d8629'
    assert (lock['arm'], lock['update'], lock['formula']) == ('S', 20000, 'fixed_fusion')
    assert request['files'] == [lock['checkpoints']['42']]
    for arm in ('C', 'S', 'D', 'SD'):
        assert read(ROOT / f'runs/{arm}_seed42/status.json')['stage'] == 'COMPLETE'
    row = request['files'][0]
    target = ROOT / row['path']
    hold_after = datetime.fromisoformat(request['time']).timestamp() + 48 * 60
    started = time.monotonic()
    held = False
    record('WAITING_ONLY_FOR_ALREADY_LOCKED_UPLOAD', lock_sha256=LOCK,
           cause='Measured upload is about .5MiB/s; guard prevents an operational 1h wait deadline from discarding the already locked selection. Training and test science are unchanged.')
    try:
        while True:
            owner()
            if target.exists():
                assert digest(target) == row['sha256']
                if held:
                    # The original storage loop sees a complete atomic checkpoint,
                    # verifies its SHA and continues the SAME one-test execution.
                    os.kill(PID, signal.SIGCONT)
                record('LOCKED_CHECKPOINT_SHA_VERIFIED_ORIGINAL_OWNER_CONTINUES',
                       lock_sha256=LOCK, checkpoint_sha256=row['sha256'],
                       operational_wait_was_held=held, lost_training_updates=0)
                return
            if (ROOT / 'reports/final_test_state.json').exists():
                assert not held
                record('ORIGINAL_LOCKED_TEST_ALREADY_STARTED_NO_INTERVENTION', lock_sha256=LOCK)
                return
            if not held and time.time() >= hold_after:
                assert read(ROOT / 'reports/campaign.json')['stage'] == 'WAITING_FOR_SELECTED_CHECKPOINT'
                assert read(ROOT / 'reports/driver_state.json')['stage'] == 'SINGLE_SEED_FINAL_SELECTION'
                assert not target.exists() and not (ROOT / 'reports/final_test_state.json').exists()
                owner()
                os.kill(PID, signal.SIGSTOP)
                held = True
                record('SAME_WAITING_OWNER_HELD_UNTIL_LOCKED_SHA_UPLOAD', lock_sha256=LOCK,
                       no_test_started_at_hold=True, lost_training_updates=0,
                       not_a_training_or_test_restart=True)
            assert time.monotonic() - started < 2 * 3600, 'Locked upload exceeded2h; diagnose transport before further actions'
            time.sleep(5)
    except BaseException as error:
        record('LOCKED_UPLOAD_GUARD_REVIEW_REQUIRED', reason=str(error), original_owner_held=held,
               automatic_restart=False)
        raise


if __name__ == '__main__':
    main()
