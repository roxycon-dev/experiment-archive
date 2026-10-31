"""Stop this campaign only, suppress shutdown, and retain its disk artifacts."""
import datetime
import json
import os
import signal
import subprocess
import time
from pathlib import Path

ROOT = Path('/root/autodl-tmp/HRDecoder_v3_20261008')
REPORTS = ROOT / 'reports'

def read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None

def process(pid):
    try:
        path = Path('/proc') / str(pid)
        stat = (path / 'stat').read_text()
        fields = stat[stat.rfind(')') + 2:].split()
        return {'pid': pid, 'state': fields[0], 'ppid': int(fields[1]),
                'start_ticks': fields[19],
                'cmdline': (path / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace').strip(),
                'cwd': os.readlink(str(path / 'cwd'))}
    except (OSError, ValueError):
        return None

def living(original):
    now = process(original['pid'])
    return now is not None and now['state'] != 'Z' and now['start_ticks'] == original['start_ticks']

def write_json(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(path)

def main():
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / 'shutdown_hold.enabled').write_text('User requested STOP and KEEP SERVER ON. Do not remove or resume automatically.\n')
    write_json(REPORTS / 'USER_STOP_REQUESTED.json', {
        'status': 'STOP_REQUESTED_BY_USER', 'requested_at_utc': started,
        'instruction': 'Stop training, archive logs locally, do not download checkpoints, do not shut down. Await new framework.',
        'shutdown_authorized': False, 'automatic_resume_authorized': False})
    running = read_json(REPORTS / 'GPU_CAMPAIGN_RUNNING.json') or {}
    inventory = [p for p in (process(int(x.name)) for x in Path('/proc').iterdir() if x.name.isdigit()) if p]
    controller = process(int(running.get('controller_pid', -1)))
    trainer = process(int(running.get('pid', -1)))
    if controller and controller['state'] != 'Z':
        assert str(ROOT / 'training_execution/gpu_campaign_controller.py') in controller['cmdline'], controller
        os.kill(controller['pid'], signal.SIGSTOP)
    if trainer and trainer['state'] != 'Z':
        assert 'train_performance.py' in trainer['cmdline'] and trainer['cwd'] == str(ROOT / 'training_execution'), trainer
    descendants = set()
    roots = {p['pid'] for p in [controller, trainer] if p}
    while True:
        added = {p['pid'] for p in inventory if p['ppid'] in roots | descendants} - descendants
        if not added:
            break
        descendants |= added
    targets = [p for p in inventory if p['pid'] in descendants or
               (p['pid'] != os.getpid() and str(ROOT) in p['cmdline'] and
                'ram_cache_mixed.py' in p['cmdline'])]
    if trainer and trainer['pid'] not in {p['pid'] for p in targets}:
        targets.append(trainer)
    actions = []
    # The frozen controller cannot initiate shutdown after the trainer exits.
    if controller and living(controller):
        os.kill(controller['pid'], signal.SIGKILL)
        actions.append({'pid': controller['pid'], 'signal': 'SIGKILL', 'role': 'shutdown_controller'})
    for p in targets:
        if living(p):
            os.kill(p['pid'], signal.SIGTERM)
            actions.append({'pid': p['pid'], 'signal': 'SIGTERM', 'cmdline': p['cmdline']})
    deadline = time.time() + 5
    while time.time() < deadline and any(living(p) for p in targets):
        time.sleep(0.2)
    for p in targets:
        if living(p):
            os.kill(p['pid'], signal.SIGKILL)
            actions.append({'pid': p['pid'], 'signal': 'SIGKILL', 'cmdline': p['cmdline']})
    time.sleep(1)
    remaining = [process(p['pid']) for p in targets + ([controller] if controller else []) if living(p)]
    perf = read_json(ROOT / 'training_runs/v3/performance.json')
    progress = read_json(ROOT / 'training_runs/v3/progress.json')
    gpu = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv'],
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    gpu_state = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,power.draw', '--format=csv'],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    memory = Path('/proc/meminfo').read_text()
    outcome = {
        'status': 'STOPPED_BY_USER' if not remaining else 'STOP_INCOMPLETE',
        'stopped_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'requested_at_utc': started, 'protocol_completed': False,
        'shutdown_command_issued': False, 'shutdown_authorized': False,
        'server_left_running': True, 'checkpoints_download_requested': False,
        'disk_checkpoints_unchanged_by_stop': True, 'automatic_resume_authorized': False,
        'previous_running_record': running, 'performance': perf, 'progress': progress,
        'target_processes': targets, 'actions': actions, 'remaining_target_processes': remaining,
        'gpu_compute_processes_csv': gpu.stdout, 'gpu_status_csv': gpu_state.stdout,
        'gpu_query_exit_code': gpu.returncode,
        'meminfo_after_stop': memory,
        'shutdown_hold_exists': (REPORTS / 'shutdown_hold.enabled').exists()}
    write_json(REPORTS / 'GPU_CAMPAIGN_STOPPED.json', outcome)
    write_json(REPORTS / 'GPU_CAMPAIGN_OUTCOME.json', outcome)
    running.update({'status': outcome['status'], 'stopped_at_utc': outcome['stopped_at_utc'],
                    'shutdown_after_completion': False, 'shutdown_after_fatal_failure_grace_seconds': None,
                    'processes_alive': False if not remaining else True})
    write_json(REPORTS / 'GPU_CAMPAIGN_RUNNING.json', running)
    print(json.dumps({'status': outcome['status'], 'time': outcome['stopped_at_utc'],
                      'performance': perf, 'progress': progress, 'remaining': remaining,
                      'gpu_compute_processes': gpu.stdout, 'gpu_status': gpu_state.stdout}, ensure_ascii=True))
    if remaining:
        raise SystemExit(2)

if __name__ == '__main__':
    main()
