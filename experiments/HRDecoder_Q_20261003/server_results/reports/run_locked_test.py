"""Run the immutable, gated final evaluator once; pause with evidence on errors."""
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qlib.common import read_json, sha, utc_now, write_json

OUT = ROOT / 'reports/final_test'
STATE = ROOT / 'reports/final_evaluation_state.json'
LOCK = ROOT / 'reports/final_selection.json'


def main():
    os.chdir(ROOT)
    lock = read_json(LOCK)
    lock_sha = sha(LOCK)
    assert lock['status'] == 'LOCKED_TEST_AUTHORIZED'
    assert lock['selection']['selected']['test_eligible']
    assert lock['driver_sha256'] == sha(ROOT/'reports/final_evaluation.py')
    if STATE.exists():
        old = read_json(STATE)
        assert old['lock_sha256'] == lock_sha
        if old['stage'] == 'COMPLETE':
            assert read_json(OUT/'summary.json')['lock_sha256'] == lock_sha
            print('Same frozen final evaluation already complete; no repeated test.', flush=True)
            return
        raise RuntimeError('Existing final evaluation state requires explicit evidence review; no automatic restart.')
    OUT.mkdir(parents=True, exist_ok=True)
    state = {'stage': 'RUNNING', 'phase': 'P0', 'pid': os.getpid(),
             'lock_sha256': lock_sha, 'runner_sha256': sha(__file__),
             'started_at': utc_now(), 'automatic_restart': False}
    write_json(STATE, state)
    (ROOT/'reports/final_test.pid').write_text(str(os.getpid())+'\n')
    try:
        for phase, interpreter in (
            ('p0', '/root/miniconda3/bin/python'),
            ('q', '/root/miniconda3/envs/hrdecoder_q_20261003/bin/python'),
        ):
            state.update(phase=phase.upper(), updated_at=utc_now())
            write_json(STATE, state)
            print('Starting locked phase '+phase, flush=True)
            rc = subprocess.call([interpreter, '-u', str(ROOT/'reports/final_evaluation.py'),
                                  '--phase', phase], cwd=str(ROOT))
            if rc:
                raise RuntimeError('Locked phase '+phase+' exited '+str(rc))
        summary = read_json(OUT/'summary.json')
        assert summary['status'] == 'COMPLETE' and summary['lock_sha256'] == lock_sha
        state.update(stage='COMPLETE', phase='COMPLETE', completed_at=utc_now(), summary=summary)
        write_json(STATE, state)
    except BaseException as exc:
        state.update(stage='TECHNICAL_PAUSE', reason=f'{type(exc).__name__}: {exc}',
                     updated_at=utc_now(), automatic_restart=False,
                     evidence_log='reports/final_evaluation.log')
        write_json(STATE, state)
        traceback.print_exc()
        raise


if __name__ == '__main__':
    main()
