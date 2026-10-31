"""Bounded high-resolution pair; validation only, completed test stays closed."""
import atexit
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
ROUND = ROOT/'round2_highres'


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)


def main():
    protocol = json.loads((ROUND/'protocol.json').read_text())
    lock = ROUND/'queue.lock'
    if lock.exists():
        try:
            os.kill(int(lock.read_text()), 0)
        except (ProcessLookupError, ValueError):
            lock.unlink()
        else:
            raise RuntimeError('Round 2 queue already running')
    lock.write_text(str(os.getpid()))
    atexit.register(lambda p=lock: p.unlink() if p.exists() else None)
    state_path = ROUND/'work_dirs/campaign_state.json'
    finish_path = ROUND/'work_dirs/finish_state.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else {'jobs': {}, 'started_at_epoch': time.time()}
    if state.get('status') == 'complete':
        return
    state.update(status='running', pid=os.getpid())
    state.pop('error', None)
    write(finish_path, {'status': 'running', 'test_policy': 'closed; validation-only extension'})
    try:
        for tag in protocol['job_order']:
            config = ROUND/'configs'/f'{tag}.yaml'
            expected = protocol['config_sha256'][tag]
            if hashlib.sha256(config.read_bytes()).hexdigest() != expected:
                raise ValueError('Locked configuration changed: '+tag)
            out = ROUND/'work_dirs'/tag
            out.mkdir(parents=True, exist_ok=True)
            summary = out/'completion_summary.json'
            if not summary.exists():
                state.update(active_job=tag)
                job = state['jobs'].setdefault(tag, {'started_at_epoch': time.time()})
                job.update(status='running', config_sha256=expected)
                if (out/'ckpt_last.pth').exists():
                    job['resumed_at_epoch'] = time.time()
                write(state_path, state)
                args = [sys.executable, '-u', str(ROOT/'train.py'), '--config', str(config),
                        '--stage', 'ddr_finetune', '--mode', 'formal', '--out-dir', str(out)]
                if (out/'ckpt_last.pth').exists():
                    args += ['--resume', str(out/'ckpt_last.pth')]
                with (out/'train.log').open('a', encoding='utf-8') as log:
                    child = subprocess.Popen(args, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                    state['jobs'][tag]['pid'] = child.pid
                    write(state_path, state)
                    rc = child.wait()
                if rc or not summary.exists():
                    raise RuntimeError('Training engineering failure '+tag+' exit='+str(rc))
            done = json.loads(summary.read_text())
            state['jobs'][tag].update(status=done['status'], summary=done, finished_at_epoch=time.time())
            # Identical final and last snapshots are redundant after a stage
            # ends. Preserve both best and full resumable last.
            final, last = out/'ckpt_final.pth', out/'ckpt_last.pth'
            if final.exists() and last.exists():
                import torch
                a = torch.load(final, map_location='cpu')
                b = torch.load(last, map_location='cpu')
                if (a['update'] == b['update'] and a['config_sha256'] == b['config_sha256'] and
                    a['model_state'].keys() == b['model_state'].keys() and
                    all(torch.equal(v, b['model_state'][k]) for k,v in a['model_state'].items())):
                    final.unlink()
                del a,b
            write(state_path, state)
        results = {}
        for tag in protocol['job_order']:
            rows = [json.loads(p.read_text()) for p in (ROUND/'work_dirs'/tag).glob('eval_u*.json')]
            best = max(rows, key=lambda row: row['fixed_fusion_miou'])
            results[tag] = {'best_update': best['update'], 'standalone_miou': best['mIoU_lesion4'],
                            'fixed_fusion_miou': best['fixed_fusion_miou'], 'baseline_miou': best['fixed_fusion_baseline_miou'],
                            'gain': best['fixed_fusion_gain'], 'updates_done': state['jobs'][tag]['summary']['updates_done']}
        status = {'status': 'complete', 'results': results, 'completed_at_epoch': time.time(),
                  'test_policy': 'No further evaluation on the previously completed test split.'}
        write(ROUND/'reports/round_result.json', status)
        write(finish_path, status)
        state.update(status='complete', active_job=None, finished_at_epoch=time.time())
        write(state_path, state)
    except Exception as error:
        state.update(status='engineering_attention_required', error=repr(error))
        write(state_path, state)
        write(finish_path, {'status': 'engineering_attention_required', 'error': repr(error)})
        raise


if __name__ == '__main__':
    main()
