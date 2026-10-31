"""Phase-aware45min small-receipt archive. No checkpoints or whole cache reads."""
import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import shlex
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / 'local_preparation'))
from remote import connect
REMOTE = '/root/autodl-tmp/HRDecoder_v3_20261008'


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--mark-probe-reported', action='store_true')
    parser.add_argument('--mark-fresh-reported', type=int, nargs='*', default=[])
    args = parser.parse_args()
    now = datetime.now(timezone(timedelta(hours=8)))
    ledger_path = HERE / 'validation_reporting.json'
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {'reported_fresh_steps': [], 'warm_probe_reported': False}
    if args.mark_probe_reported or args.mark_fresh_reported:
        series = json.loads((HERE / 'screening_series.json').read_text(encoding='utf-8'))
        assert set(args.mark_fresh_reported) <= {r['completed'] for r in series['fresh_val']}
        if args.mark_probe_reported:
            assert series['warm_gate'] and series['warm_gate']['full_val_counts_verified']
            ledger['warm_probe_reported'] = True
        ledger['reported_fresh_steps'] = sorted(set(ledger['reported_fresh_steps']) | set(args.mark_fresh_reported))
        ledger['last_reported_at'] = now.isoformat()
        write(ledger_path, ledger)
        print(json.dumps({'status': 'MARKED_LOCALLY', 'SSH_poll': False})); return
    # Explicit launch verification/repair does not postpone the next scheduled
    # heartbeat by an additional45min. Routine observations have their own clock.
    if ledger.get('last_routine_checked_at') and not args.force:
        due = datetime.fromisoformat(ledger['last_routine_checked_at']) + timedelta(minutes=45)
        if now < due - timedelta(seconds=60):
            print(json.dumps({'status': 'WAITING_FOR_MONITOR_WINDOW', 'next_check_at': due.isoformat(), 'SSH_poll': False})); return
    client = connect()
    downloaded = []
    try:
        with client.open_sftp() as ftp:
            pairs = [('reports/sacd_micro', 'sacd_micro'), ('training_runs/sacd_micro', 'sacd_micro'),
                     ('reports/sacd_isolated', 'sacd_isolated'), ('training_runs/sacd_isolated', 'sacd_isolated'),
                     ('sacd_micro_execution', 'sacd_micro'), ('sacd_isolated_execution', 'sacd_isolated')]
            def archive(directory, run):
                try:
                    files = ftp.listdir_attr(REMOTE + '/' + directory)
                except OSError:
                    return
                import stat
                for entry in files:
                    rel = directory + '/' + entry.filename
                    if stat.S_ISDIR(entry.st_mode):
                        archive(rel, run)
                    elif entry.filename.endswith(('.json', '.log')):
                        assert entry.st_size < 15 * 1024**2, 'Unexpectedly large text artifact'
                        raw = ftp.open(REMOTE + '/' + rel, 'rb').read()
                        if entry.filename.endswith('.json'):
                            json.loads(raw)
                        target = ROOT / 'cache_execution/server_receipts' / run / rel
                        target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(raw)
                        downloaded.append(rel)
            for pair in pairs:
                archive(*pair)
            # One final original20k val may arrive during the automatic transition.
            for name in ('val_20000.json', 'progress.json', 'performance.json'):
                try:
                    raw = ftp.open(REMOTE + '/training_runs/sacd/' + name, 'rb').read()
                except OSError:
                    continue
                target = ROOT / 'cache_execution/server_receipts/sacd/training_runs/sacd' / name
                target.write_bytes(raw)
        code = '''import json,time,subprocess
from pathlib import Path
root=Path('/root/autodl-tmp/HRDecoder_v3_20261008')
out={'server_online':True,'checked_at_unix':time.time()}
for name in ('sacd_micro','sacd_isolated'):
 for file in ('STAGING_STATUS.json','GPU_CAMPAIGN_RUNNING.json','GPU_CAMPAIGN_OUTCOME.json','CONTROLLER_ERROR.json'):
  path=root/'reports'/name/file
  if path.exists():out[name+'/'+file]=json.loads(path.read_text())
out['GPU_status']=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,power.draw','--format=csv'],universal_newlines=True)
print(json.dumps(out))
'''
        _, out, err = client.exec_command('/root/miniconda3/bin/python -c ' + shlex.quote(code))
        raw, error = out.read().decode(), err.read().decode()
        if out.channel.recv_exit_status():
            raise RuntimeError(error)
        runtime = json.loads(raw)
    finally:
        client.close()
    from gate import verify
    ref = json.loads((ROOT / 'sacd_execution/val_label_reference.json').read_text())
    warm_root = ROOT / 'cache_execution/server_receipts/sacd_micro'
    warm_path = warm_root / 'reports/sacd_micro/MICRO_GATE.json'
    warm = json.loads(warm_path.read_text()) if warm_path.exists() else None
    warm_val = []
    for path in sorted((warm_root / 'training_runs/sacd_micro').glob('*/val_probe_*.json')):
        if path.parent.name not in ('control', 'feature_isolated'):
            continue  # Failed loader attempt evidence is preserved, not an extra selected arm.
        value = verify(json.loads(path.read_text()), ref['ground_truth_positive_pixels'])
        warm_val.append(value)
    fresh_root = ROOT / 'cache_execution/server_receipts/sacd_isolated'
    fresh_val = []
    for path in sorted((fresh_root / 'training_runs/sacd_isolated').glob('val_*.json')):
        value = verify(json.loads(path.read_text()), ref['ground_truth_positive_pixels'])
        fresh_val.append(value)
    original20 = ROOT / 'cache_execution/server_receipts/sacd/training_runs/sacd/val_20000.json'
    if original20.exists():
        parent = verify(json.loads(original20.read_text()), ref['ground_truth_positive_pixels'])
        source = ROOT / 'sacd_execution/validation_series.json'
        original_series = json.loads(source.read_text(encoding='utf-8'))
        if not any(r['step'] == 20000 for r in original_series['records']):
            previous = original_series['records'][-1]
            historical = json.loads((ROOT / 'training_execution/historical_checkpoint_comparison.json').read_text(encoding='utf-8'))
            baseline = {r['step']: r['mIoU'] for r in historical['baseline_history']}
            cet = json.loads((ROOT / 'cet_execution/validation_series.json').read_text(encoding='utf-8'))
            cet_map = {r['step']: r['mIoU'] for r in cet['records']}
            v3 = json.loads((ROOT / 'training_execution/validation_series.json').read_text(encoding='utf-8'))
            v3_map = {r['step']: r['mIoU_percent'] for r in v3['records']}
            classes = [{'name': name, 'counts': parent['counts'][i],
                        'IoU': parent['IoU_percent'][i], 'precision': parent['precision_percent'][i],
                        'recall': parent['recall_percent'][i]} for i, name in enumerate(ref['class_order'])]
            original_series['records'].append({'step': 20000, 'mIoU': parent['mIoU_percent'],
                'classes': classes, 'counts_verified': True,
                'delta_mIoU_pp': parent['mIoU_percent'] - previous['mIoU'],
                'delta_class_IoU_pp': [c['IoU'] - p['IoU'] for c, p in zip(classes, previous['classes'])],
                'versus_original_HRDecoder_pp': parent['mIoU_percent'] - baseline[20000],
                'versus_CET_same_step_pp': parent['mIoU_percent'] - cet_map[20000],
                'versus_v3_same_step_pp': parent['mIoU_percent'] - v3_map[20000] if 20000 in v3_map else None,
                'base_head_mIoU': parent['base_head_metrics']['mIoU_percent']})
            original_series['best_val'] = max(original_series['records'], key=lambda r: r['mIoU'])
            original_ledger = json.loads((ROOT / 'sacd_execution/validation_reporting.json').read_text(encoding='utf-8'))
            original_series['unreported_steps'] = [r['step'] for r in original_series['records'] if r['step'] not in original_ledger['reported_steps']]
            write(source, original_series)
    series = {'checked_at': now.isoformat(), 'warm_gate': warm, 'warm_val': warm_val, 'fresh_val': fresh_val,
              'counts_verified': True, 'runtime': runtime,
              'unreported_fresh_steps': [r['completed'] for r in fresh_val if r['completed'] not in ledger['reported_fresh_steps']],
              'warm_gate_unreported': bool(warm) and not ledger['warm_probe_reported']}
    write(HERE / 'screening_series.json', series)
    ledger['last_checked_at'] = now.isoformat()
    ledger['last_manual_checked_at' if args.force else 'last_routine_checked_at'] = now.isoformat()
    write(ledger_path, ledger)
    write(HERE / 'latest_snapshot.json', {'checked_at': now.isoformat(), 'runtime': runtime, 'downloaded': downloaded})
    state_path = ROOT / 'iterative_execution/iteration_state.json'
    state = json.loads(state_path.read_text(encoding='utf-8'))
    state['screening_latest_observation'] = runtime
    state['last_real_monitor_at'] = now.isoformat()
    state['next_scheduled_monitor'] = (now + timedelta(minutes=45)).isoformat()
    fresh_running = runtime.get('sacd_isolated/GPU_CAMPAIGN_RUNNING.json')
    if fresh_running:
        state['active_run'] = 'sacd_isolated'; state['status'] = fresh_running['status']
        if fresh_running.get('screen_pass'):
            state['conditional_extension_activated'] = True
            state['another_full_run_authorized'] = True
            state['deadline_rule_active'] = 'Until this same screen-passed fresh run finishes; no additional candidates'
    elif runtime.get('sacd_micro/STAGING_STATUS.json', {}).get('status') != 'WAITING_FOR_DURABLE_20K':
        state['active_run'] = 'sacd_micro'
        state['status'] = runtime.get('sacd_micro/STAGING_STATUS.json', {}).get('status', 'SCREENING_PREPARED')
    write(state_path, state)
    print(json.dumps({'status': 'ARCHIVED', 'phase': state['status'],
                      'warm_gate_unreported': series['warm_gate_unreported'],
                      'unreported_fresh_steps': series['unreported_fresh_steps'], 'runtime': runtime}))


if __name__ == '__main__':
    main()
