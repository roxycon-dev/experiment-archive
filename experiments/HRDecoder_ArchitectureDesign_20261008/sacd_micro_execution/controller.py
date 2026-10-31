"""Wait for a durable20k parent, pause only our SACD, run two registered short arms."""
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REPORTS = ROOT / 'reports/sacd_micro'
WORK = ROOT / 'training_runs/sacd_micro'
PYTHON = '/root/miniconda3/bin/python'
OLD = ROOT / 'reports/sacd'


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def cmdline(pid):
    return Path('/proc/%d/cmdline' % pid).read_bytes()


def alive(pid):
    try:
        return Path('/proc/%d/stat' % pid).read_text().split(') ', 1)[1].split()[0] != 'Z'
    except OSError:
        return False


def stop_registered(pid, expected):
    assert expected(cmdline(pid)), 'Refuse to control a process whose identity differs'
    os.kill(pid, signal.SIGTERM)
    for unused in range(100):
        if not alive(pid):
            return
        time.sleep(.1)
    # Still only the explicitly registered, identity-matched own process.
    assert expected(cmdline(pid))
    os.kill(pid, signal.SIGKILL)
    for unused in range(100):
        if not alive(pid):
            return
        time.sleep(.1)
    raise RuntimeError('Own process failed to exit')


def main():
    REPORTS.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    plan = json.loads((HERE / 'micro_protocol.json').read_text())
    assert not (REPORTS / 'MICRO_STARTED.json').exists(), 'No repeated diagnostic campaign'
    with open(REPORTS / 'staging.lock', 'a') as staging_lock:
        fcntl.flock(staging_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        running = json.loads((OLD / 'GPU_CAMPAIGN_RUNNING.json').read_text())
        assert running['status'] == 'RUNNING' and running['pid'] == 25190 and running['controller_pid'] == 25187
        write(REPORTS / 'STAGING_STATUS.json', {'status': 'WAITING_FOR_DURABLE_20K',
            'controller_pid': os.getpid(), 'parent_training_pid': running['pid'],
            'parent_controller_pid': running['controller_pid'], 'time_unix': time.time(),
            'server_shutdown': False, 'external_diagnostics_controlled': False})
        # This server-side staging controller is detached from the chat; it keeps
        # the existing GPU task running until its own full20k val and save finish.
        staging_deadline = time.time() + 3600
        while True:
            assert alive(running['pid']) and alive(running['controller_pid']), 'Parent ended before durable20k transition'
            progress = json.loads((ROOT / 'training_runs/sacd/progress.json').read_text())
            saved = progress.get('saved_update', 0)
            if saved == 20000 and (ROOT / 'training_runs/sacd/val_20000.json').exists():
                break
            assert saved < 20000, 'Missed20k immutable parent; do not silently substitute a later checkpoint'
            assert time.time() < staging_deadline, '20k staging exceeded its bounded window'
            time.sleep(2)
        parent = WORK / 'parent_20000.pt'
        os.link(str(ROOT / 'training_runs/sacd/last.pt'), str(parent))
        import torch
        state = torch.load(parent, map_location='cpu')
        assert state['completed'] == 20000 and state['identity_SHA'] == plan['parent_identity_SHA']
        assert state['optimizer']['state'] and 'rng' in state
        del state
        # End our original controller first so it cannot destroy our reusable RAM
        # cache as a reaction to the intentional diagnostic pause.
        stop_registered(running['controller_pid'], lambda cmd: str(ROOT / 'sacd_execution/controller.py').encode() in cmd)
        assert Path('/proc/%d/cwd' % running['pid']).resolve() == ROOT / 'sacd_execution'
        stop_registered(running['pid'], lambda cmd: b'train.py' in cmd and b'--train' in cmd)
        stop_registered(running['telemetry_pid'], lambda cmd: b'nvidia-smi' in cmd and b'--loop=2' in cmd)
        pause = {'status': 'PAUSED_FOR_REGISTERED_PAIRED_DIAGNOSTIC', 'saved_parent_update': 20000,
                 'last_observed_completed_before_pause': progress['completed'],
                 'discarded_unsaved_updates_possible': True, 'parent_checkpoint_preserved': True,
                 'original_identity_SHA': plan['parent_identity_SHA'], 'RAM_owner_retained': True,
                 'time_unix': time.time(), 'server_kept_running': True, 'external_diagnostics_controlled': False}
        write(OLD / 'MANUAL_DIAGNOSTIC_PAUSE.json', pause)
        running.update(status=pause['status'], processes_alive=False, end_unix=pause['time_unix'])
        write(OLD / 'GPU_CAMPAIGN_RUNNING.json', running)
        with open(ROOT / 'reports/gpu.lock', 'a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            for unused in range(30):
                gpu = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                               '--format=csv,noheader'], universal_newlines=True).strip()
                if not gpu:
                    break
                assert all(int(item.strip()) == running['pid'] for item in gpu.splitlines()), 'External GPU task present; refuse to interfere'
                time.sleep(.5)
            assert not gpu, 'Own GPU context did not exit'
            write(REPORTS / 'MICRO_STARTED.json', {'status': 'RUNNING', 'time_unix': time.time(),
                'controller_pid': os.getpid(), 'protocol': plan, 'parent': pause, 'repeat_authorized': False})
            env = dict(os.environ, PYTHONPATH=str(ROOT / 'vendor/champion_repo'),
                       OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='4')
            for arm in plan['arms']:
                with open(REPORTS / (arm + '.log'), 'a', buffering=1) as log:
                    process = subprocess.Popen([PYTHON, '-u', 'probe.py', '--arm', arm],
                                               cwd=str(HERE), env=env, stdout=log, stderr=log)
                    write(REPORTS / 'STAGING_STATUS.json', {'status': 'PAIRED_PROBE_RUNNING',
                        'arm': arm, 'pid': process.pid, 'controller_pid': os.getpid(), 'time_unix': time.time()})
                    try:
                        code = process.wait(timeout=1500)
                    except subprocess.TimeoutExpired:
                        process.terminate()
                        try:
                            process.wait(timeout=20)
                        except subprocess.TimeoutExpired:
                            process.kill(); process.wait()
                        raise RuntimeError('Registered short arm exceeded25min safety limit')
                    assert code == 0, 'Diagnostic arm failed: %s exit%s' % (arm, code)
            from gate import evaluate_gate
            outcome = evaluate_gate(ROOT, plan)
            outcome.update(time_unix=time.time(), server_kept_running=True,
                           external_diagnostics_controlled=False, parent_checkpoint_preserved=True)
            write(REPORTS / 'MICRO_GATE.json', outcome)
            # A separate sealed fresh-screen entry point must exist before using
            # the warm gate. Warm success alone never authorizes a full40k run.
            write(REPORTS / 'STAGING_STATUS.json', {'status': 'WARM_GATE_' + outcome['status'],
                'fresh_screen_required': outcome['status'] == 'PASS', 'time_unix': time.time(),
                'server_kept_running': True})
        if outcome['status'] == 'PASS':
            # Released the diagnostic GPU lock first; the new controller claims
            # it itself and rechecks foreign compute processes before launching.
            with open(REPORTS / 'fresh_transition.log', 'a', buffering=1) as log:
                successor = subprocess.Popen([PYTHON, '-u', str(ROOT / 'sacd_isolated_execution/controller.py')],
                    cwd=str(ROOT / 'sacd_isolated_execution'), env=env,
                    stdout=log, stderr=log, start_new_session=True)
            write(REPORTS / 'FRESH_SCREEN_DISPATCHED.json', {'controller_pid': successor.pid,
                'time_unix': time.time(), 'full40k_authorized_yet': False,
                'warm_gate_status': 'PASS', 'server_kept_running': True})


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        write(REPORTS / 'CONTROLLER_ERROR.json', {'status': 'FAILED', 'error': str(error),
            'traceback': traceback.format_exc(), 'time_unix': time.time(),
            'server_kept_running': True, 'external_diagnostics_controlled': False})
        raise
