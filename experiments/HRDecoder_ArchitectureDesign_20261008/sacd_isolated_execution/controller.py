"""One fresh screen; conditional continuation of that same run; never shutdown."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REPORTS = ROOT / 'reports/sacd_isolated'
WORK = ROOT / 'training_runs/sacd_isolated'
PYTHON = '/root/miniconda3/bin/python'


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def stop_own(process):
    process.terminate()
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        process.kill(); process.wait()


def end_own_ram():
    ready = json.loads((ROOT / 'reports/sacd/RAM_CACHE_READY.json').read_text())
    try:
        command = Path('/proc/%d/cmdline' % ready['pid']).read_bytes()
        assert str(ROOT / 'sacd_execution/ram_cache_owner.py').encode() in command
        os.kill(ready['pid'], 15)
        return {'own_RAM_termination_requested': True, 'pid': ready['pid']}
    except OSError:
        return {'own_RAM_termination_requested': False}


def main():
    REPORTS.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    warm = json.loads((ROOT / 'reports/sacd_micro/MICRO_GATE.json').read_text())
    assert warm['status'] == 'PASS' and warm['full_val_counts_verified']
    assert not (WORK / 'last.pt').exists(), 'No second fresh run or implicit checkpoint reuse'
    assert not (REPORTS / 'SCREEN_STARTED.json').exists(), 'Candidate already registered'
    for name in ('CPU_SACD_PASS.json', 'CHECKPOINT_RESUME_PASS.json', 'FEATURE_ISOLATION_CPU_PASS.json'):
        value = json.loads((HERE / name).read_text())
        assert value['status'] == 'PASS' and value['torch'] == '2.0.1+cu117'
    env = dict(os.environ, PYTHONPATH=str(ROOT / 'vendor/champion_repo'),
               OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='4')
    with open(ROOT / 'reports/gpu.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        gpu = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                       '--format=csv,noheader'], universal_newlines=True).strip()
        assert not gpu, 'Unrelated GPU process present; refuse to interfere'
        with open(REPORTS / 'smoke.log', 'a', buffering=1) as log:
            probe = subprocess.Popen([PYTHON, '-u', 'train.py', '--smoke'], cwd=str(HERE),
                                     env=env, stdout=log, stderr=log)
            try:
                code = probe.wait(timeout=600)
            except subprocess.TimeoutExpired:
                stop_own(probe); raise RuntimeError('Candidate GPU smoke timed out')
            assert code == 0, 'Candidate GPU smoke failed'
        smoke = json.loads((WORK / 'GPU_SMOKE_PASS.json').read_text())
        assert smoke['status'] == 'PASS' and smoke['formal_updates'] == 0
        assert smoke['fresh_initialization_verified'] and smoke['canvas'] == 2048 and smoke['batch'] == 2
        # Nonzero-auxiliary full-shape smoke, not the faster pre1500 phase.
        estimate_screen = 8000 * smoke['mean_warm_step_seconds'] + 600
        assert time.time() + estimate_screen < 1791568800 - 180, 'Cannot finish screen within original deadline'
        registration = {'kind': 'ONE_FRESH_8K_SCREEN', 'identity_SHA': smoke['identity_SHA'],
                        'start_unix': time.time(), 'candidate_count': 1, 'candidate_limit': 1,
                        'full40k_authorized_now': False, 'full40k_condition': 'registered fresh8k screen PASS',
                        'initialization': 'fresh ImageNet W48 seed42/private20261008',
                        'warm_diagnostic_checkpoint_used': False, 'user_conditional_deadline_extension': True}
        with (REPORTS / 'SCREEN_STARTED.json').open('x') as file:
            json.dump(registration, file, indent=2)
        started = time.time()
        with open(REPORTS / 'training.log', 'a', buffering=1) as log:
            process = subprocess.Popen([PYTHON, '-u', 'train.py', '--train'], cwd=str(HERE),
                                       env=env, stdout=log, stderr=log)
            running = {'status': 'FRESH_SCREEN_RUNNING', 'pid': process.pid,
                       'controller_pid': os.getpid(), 'identity_SHA': smoke['identity_SHA'],
                       'start_unix': started, 'canvas': 2048, 'batch': 2,
                       'screen_updates': 8000, 'max_conditional_updates': 40000,
                       'shutdown_after_completion': False, 'shutdown_after_failure': False}
            write(REPORTS / 'GPU_CAMPAIGN_RUNNING.json', running)
            while process.poll() is None:
                extension_path = REPORTS / 'DEADLINE_EXTENSION_ACTIVATED.json'
                extended = extension_path.exists()
                if extended:
                    extension = json.loads(extension_path.read_text())
                    screen = json.loads((WORK / 'SCREEN_PASS.json').read_text())
                    assert extension['full40k_authorized'] and screen['identity_SHA'] == smoke['identity_SHA']
                    assert screen['time_unix'] < 1791568800
                    if running['status'] != 'FULL_40K_RUNNING_AFTER_SCREEN_PASS':
                        running.update(status='FULL_40K_RUNNING_AFTER_SCREEN_PASS', screen_pass=screen)
                        write(REPORTS / 'GPU_CAMPAIGN_RUNNING.json', running)
                # A finite safety bound prevents a hung own process; extension
                # explicitly supersedes the original2am cutoff after screen PASS.
                cutoff = started + 15 * 3600 if extended else 1791568800 - 180
                if time.time() >= cutoff:
                    stop_own(process)
                    write(REPORTS / 'DEADLINE_STOP.json', {'time_unix': time.time(),
                        'extension_active': extended, 'checkpoint_preserved': True, 'server_kept_running': True})
                    break
                time.sleep(5)
            code = process.wait()
        complete_path = WORK / 'TRAINING_COMPLETE.json'
        complete = json.loads(complete_path.read_text()) if complete_path.exists() else None
        rejected = json.loads((WORK / 'SCREEN_REJECTED.json').read_text()) if (WORK / 'SCREEN_REJECTED.json').exists() else None
        final = json.loads((WORK / 'val_40000.json').read_text()) if (WORK / 'val_40000.json').exists() else None
        full_done = code == 0 and complete is not None and complete['completed'] == 40000
        if full_done:
            assert complete['durable_checkpoint_loaded_and_verified'] and final['images'] == 76
        test = {'status': 'NOT_RUN', 'reason': 'Fresh screen rejected or training incomplete'}
        if full_done and final['mIoU_percent'] > 45.32183122738841:
            with open(REPORTS / 'test.log', 'a', buffering=1) as log:
                evaluation = subprocess.Popen([PYTHON, '-u', 'test_eval.py'], cwd=str(HERE),
                                              env=env, stdout=log, stderr=log)
                try:
                    test_code = evaluation.wait(timeout=600)
                except subprocess.TimeoutExpired:
                    stop_own(evaluation); test_code = 124
                test = {'status': 'COMPLETE' if test_code == 0 else 'FAILED', 'exit_code': test_code}
        elif full_done:
            test = {'status': 'SKIPPED', 'reason': 'Fixed40k val target not reached; combined goal not achieved'}
        outcome = {'status': 'COMPLETE' if full_done else 'SCREEN_REJECTED' if rejected and code == 0 else 'FAILED',
                   'exit_code': code, 'time_unix': time.time(), 'start_unix': started,
                   'completion': complete, 'final_validation': final, 'fresh_screen_rejection': rejected,
                   'test_evaluation': test, 'server_kept_running': True, 'shutdown_command_issued': False,
                   'external_diagnostics_controlled': False, **end_own_ram()}
        write(REPORTS / 'GPU_CAMPAIGN_OUTCOME.json', outcome)
        running.update(status=outcome['status'], processes_alive=False, end_unix=outcome['time_unix'])
        write(REPORTS / 'GPU_CAMPAIGN_RUNNING.json', running)
        os.sync()


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        write(REPORTS / 'CONTROLLER_ERROR.json', {'status': 'FAILED', 'error': str(error),
            'traceback': traceback.format_exc(), 'time_unix': time.time(),
            'server_kept_running': True, 'external_diagnostics_controlled': False})
        raise
