"""Run the two predeclared validation-only trials, then wait briefly for backup."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('/root/autodl-tmp/hr100')
CODE = ROOT / 'code'


def save(value):
    tmp = ROOT / 'stage1_state.tmp'
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(ROOT / 'stage1_state.json')


def main():
    os.chdir(CODE)
    os.environ['PYTHONPATH'] = str(CODE)
    os.environ['PATH'] = '/root/miniconda3/bin:' + os.environ.get('PATH', '')
    os.environ['OMP_NUM_THREADS'] = '4'
    trials = []
    for name in ['a', 'b']:
        save({'phase': 'training', 'trial': name, 'completed': trials, 'updated_unix': time.time()})
        work = ROOT / ('trial_' + name)
        with (ROOT / ('trial_' + name + '.log')).open('wb') as output:
            rc = subprocess.call([sys.executable, 'tools/train.py',
                                  'configs/lesion/exp/hr100_' + name + '.py',
                                  '--work-dir', str(work)], stdout=output, stderr=output)
        if rc:
            save({'phase': 'failed', 'trial': name, 'returncode': rc, 'completed': trials})
            raise SystemExit(rc)
        metrics = []
        for log in work.glob('*.log.json'):
            for line in log.read_text().splitlines():
                row = json.loads(line)
                if row.get('mode') == 'val' and 'mIoU' in row:
                    metrics.append(row)
        trials.append({'trial': name, 'validation': metrics,
                       'best_checkpoints': [str(p) for p in work.glob('best_mIoU*.pth')]})
    save({'phase': 'awaiting_backup', 'completed': trials, 'updated_unix': time.time(),
          'test_evaluated': False})
    until = time.time() + 300
    while not (ROOT / 'backup_complete').exists() and time.time() < until:
        time.sleep(5)
    save({'phase': 'finished', 'completed': trials, 'updated_unix': time.time(),
          'backup_acknowledged': (ROOT / 'backup_complete').exists(), 'test_evaluated': False})


if __name__ == '__main__':
    main()
