"""Restore the exact read-only CPU diagnostic interrupted during training stop."""
import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

ROOT = Path('/root/autodl-tmp/HRDecoder_v3_20261008')
SCRIPT = ROOT / 'reports/diagnostics/probe_normalization.py'
OUT = ROOT / 'reports/diagnostics/v4'

def main():
    running = []
    for p in Path('/proc').iterdir():
        if p.name.isdigit() and int(p.name) != os.getpid():
            try:
                cmd = (p / 'cmdline').read_bytes().replace(b'\0', b' ').decode()
                state = (p / 'stat').read_text().split(') ', 1)[1].split()[0]
                if str(SCRIPT) in cmd and state != 'Z':
                    running.append(int(p.name))
            except OSError:
                pass
    if running:
        print(json.dumps({'status': 'ALREADY_RUNNING_NO_DUPLICATE_LAUNCH', 'pids': running}))
        return
    receipt = json.loads((ROOT / 'reports/GPU_CAMPAIGN_STOPPED.json').read_text())
    original = next(p for p in receipt['target_processes'] if str(SCRIPT) in p['cmdline'])
    assert original['cmdline'] == '/root/miniconda3/bin/python -u ' + str(SCRIPT)
    partial = OUT / 'normalization_probe.json'
    saved = OUT / 'normalization_probe_interrupted_20261009T001430.json'
    assert not saved.exists(), 'Preserve existing backup; do not repeat restoration.'
    if partial.exists():
        shutil.copyfile(partial, saved)
    logpath = OUT / 'normalization_probe_restored_20261009.log'
    with logpath.open('ab', buffering=0) as log:
        child = subprocess.Popen(['/root/miniconda3/bin/python', '-u', str(SCRIPT)],
                                 cwd=original['cwd'], stdout=log, stderr=subprocess.STDOUT,
                                 stdin=subprocess.DEVNULL, start_new_session=True)
    time.sleep(2)
    result = {'status': 'RESTORED_CPU_DIAGNOSTIC' if child.poll() is None else 'RESTORE_EXITED',
              'pid': child.pid, 'poll': child.poll(),
              'time_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'original_pid': original['pid'], 'cwd': original['cwd'],
              'partial_output_preserved': str(saved), 'stdout_log': str(logpath),
              'script': str(SCRIPT), 'device': 'CPU', 'training_restarted': False,
              'reason': 'User clarified another AI is running this diagnostic; restore only the same read-only diagnostic. Do not stop it again.'}
    (OUT / 'DIAGNOSTIC_RESTORED_AFTER_TRAINING_STOP.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))
    if child.poll() is not None:
        print(logpath.read_text()[-4000:])
        raise SystemExit(2)

if __name__ == '__main__':
    main()
