"""Explicit one-time recovery of the failed micro loader setup; same immutable20k parent."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
import traceback
from controller import write, ROOT, HERE, REPORTS, WORK, PYTHON


def main():
    repair = json.loads((REPORTS / 'LOADER_STARTUP_REPAIR.json').read_text())
    assert repair['status'] == 'LOADER_STARTUP_REPAIR' and not repair['failed_attempt_progress_receipt_exists']
    assert not repair['failed_attempt_formal_checkpoint_exists']
    assert not (REPORTS / 'MICRO_RECOVERY_STARTED.json').exists(), 'No blind repeated retry'
    first = json.loads((REPORTS / 'MICRO_STARTED_first_attempt.json').read_text())
    plan = json.loads((HERE / 'micro_protocol.json').read_text())
    assert first['protocol'] == plan, 'Do not alter the registered metric gates after observing results'
    assert first['parent']['saved_parent_update'] == 20000
    assert (WORK / 'parent_20000.pt').exists()
    env = dict(os.environ, PYTHONPATH=str(ROOT / 'vendor/champion_repo'),
               OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='4')
    with open(ROOT / 'reports/gpu.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        gpu = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                       '--format=csv,noheader'], universal_newlines=True).strip()
        assert not gpu, 'Unrelated GPU process present; do not interfere'
        # A focused sampler reset audit plus original synthetic gradient proof;
        # both arms must independently repeat their corresponding new-code GPU gate.
        from loader import ResumableBatches
        sampler = ResumableBatches(454, 20000, 20003)
        three = list(sampler); sampler.start, sampler.stop = 20000, 20500
        assert list(sampler) == list(ResumableBatches(454, 20000, 20500))
        assert list(sampler)[:3] == three
        write(REPORTS / 'SAMPLER_RESET_CPU_PASS.json', {'status': 'PASS',
            'same_exposure_and_seed_order_after_persistent_loader_reset': True, 'GPU_used': False})
        write(REPORTS / 'MICRO_RECOVERY_STARTED.json', {'status': 'RUNNING', 'time_unix': time.time(),
            'controller_pid': os.getpid(), 'reason': repair['reason'], 'protocol_unchanged': True,
            'parent_identity_SHA': plan['parent_identity_SHA'], 'CPU_sampler_gate': 'PASS',
            'new_code_GPU_smoke_required_each_arm': True, 'failed_attempt_archived': True})
        for arm in plan['arms']:
            with open(REPORTS / (arm + '.log'), 'a', buffering=1) as log:
                process = subprocess.Popen([PYTHON, '-u', 'probe.py', '--arm', arm],
                    cwd=str(HERE), env=env, stdout=log, stderr=log)
                write(REPORTS / 'STAGING_STATUS.json', {'status': 'PAIRED_PROBE_RUNNING',
                    'arm': arm, 'pid': process.pid, 'controller_pid': os.getpid(),
                    'loader_recovery': True, 'time_unix': time.time()})
                try:
                    code = process.wait(timeout=1500)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    try: process.wait(timeout=20)
                    except subprocess.TimeoutExpired: process.kill(); process.wait()
                    raise RuntimeError('Diagnostic recovery arm exceeded25min')
                assert code == 0, 'Repaired diagnostic arm failed: ' + arm
        from gate import evaluate_gate
        outcome = evaluate_gate(ROOT, plan)
        outcome.update(time_unix=time.time(), server_kept_running=True,
            external_diagnostics_controlled=False, parent_checkpoint_preserved=True,
            loader_startup_recovery_recorded=True)
        write(REPORTS / 'MICRO_GATE.json', outcome)
        write(REPORTS / 'STAGING_STATUS.json', {'status': 'WARM_GATE_' + outcome['status'],
            'time_unix': time.time(), 'fresh_screen_required': outcome['status'] == 'PASS',
            'server_kept_running': True})
    if outcome['status'] == 'PASS':
        with open(REPORTS / 'fresh_transition.log', 'a', buffering=1) as log:
            successor = subprocess.Popen([PYTHON, '-u', str(ROOT / 'sacd_isolated_execution/controller.py')],
                cwd=str(ROOT / 'sacd_isolated_execution'), env=env, stdout=log, stderr=log, start_new_session=True)
        write(REPORTS / 'FRESH_SCREEN_DISPATCHED.json', {'controller_pid': successor.pid,
            'time_unix': time.time(), 'full40k_authorized_yet': False, 'warm_gate_status': 'PASS'})


if __name__ == '__main__':
    try:
        import sys
        sys.path.insert(0,str(ROOT/'sacd_execution'))
        main()
    except BaseException as error:
        write(REPORTS / 'CONTROLLER_ERROR.json', {'status': 'FAILED', 'error': str(error),
            'traceback': traceback.format_exc(), 'time_unix': time.time(),
            'server_kept_running': True, 'external_diagnostics_controlled': False})
        raise
