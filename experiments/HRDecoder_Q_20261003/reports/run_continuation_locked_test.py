"""One immutable continuation decision and one gated evaluation or count reuse."""
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'reports'))
from qlib.common import read_json, sha, utc_now, write_json
import continuation_final_evaluation as evaluation

STATE = evaluation.STATE
OUT = evaluation.OUT


def main():
    os.chdir(ROOT)
    lock = evaluation.freeze()
    lock_sha = sha(evaluation.LOCK)
    if STATE.exists():
        old = read_json(STATE)
        assert old['lock_sha256'] == lock_sha
        if old['stage'] == 'COMPLETE':
            assert read_json(OUT / 'summary.json')['lock_sha256'] == lock_sha
            print('Same locked continuation already complete; no repeated test.')
            return
        raise RuntimeError('Existing continuation state needs explicit evidence review; no automatic restart')
    OUT.mkdir(parents=True, exist_ok=True)
    state = dict(stage='RUNNING', phase='PREPARE', pid=os.getpid(), lock_sha256=lock_sha,
        runner_sha256=sha(__file__), test_action=lock['test_action'], started_at=utc_now(), automatic_restart=False)
    with os.fdopen(os.open(str(STATE), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), 'w', encoding='utf-8') as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    (ROOT / 'reports/continuation_final_test.pid').write_text(str(os.getpid()) + '\n')
    try:
        if lock['test_action'] == 'NO_TEST_NOT_ELIGIBLE':
            write_json(OUT / 'summary.json', dict(status='COMPLETE', test_action=lock['test_action'],
                lock_sha256=lock_sha, selection=lock['selection'], test_performed=False, completed_at=utc_now()))
        elif lock['test_action'] == 'REUSE_EXACT_PRIOR_TEST_COUNTS':
            state.update(phase='REUSE_PRIOR_COUNTS'); write_json(STATE, state)
            evaluation.reuse_prior_counts()
        else:
            assert lock['test_action'] == 'EVALUATE_NEW_IDENTITY_ONCE'
            for phase, interpreter in (('p0', '/root/miniconda3/bin/python'),
                                      ('q', '/root/miniconda3/envs/hrdecoder_q_20261003/bin/python')):
                state.update(phase=phase.upper(), updated_at=utc_now()); write_json(STATE, state)
                rc = subprocess.call([interpreter, '-u', str(ROOT / 'reports/continuation_final_evaluation.py'),
                                      '--phase', phase], cwd=str(ROOT))
                if rc:
                    raise RuntimeError('Locked continuation phase %s exited %s' % (phase, rc))
        summary = read_json(OUT / 'summary.json')
        assert summary['status'] == 'COMPLETE' and summary['lock_sha256'] == lock_sha
        state.update(stage='COMPLETE', phase='COMPLETE', completed_at=utc_now(), summary=summary)
        write_json(STATE, state)
    except BaseException as error:
        state.update(stage='TECHNICAL_PAUSE', reason='%s: %s' % (type(error).__name__, error),
            updated_at=utc_now(), automatic_restart=False, evidence_log='reports/continuation_final_evaluation.log')
        write_json(STATE, state)
        traceback.print_exc()
        raise


if __name__ == '__main__':
    main()
