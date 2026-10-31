"""Bounded, read-only SSH health check; credentials stay in process memory."""
import argparse
import getpass
import json
import os
import shlex
from pathlib import Path

import paramiko

ROOT = Path(__file__).resolve().parents[1]
REMOTE_CHECK = r'''
import json, pathlib, shutil, subprocess, time
r = pathlib.Path('/root/autodl-tmp/HRDecoder_Ready_20261002')
state = json.loads((r/'work_dirs/campaign_state.json').read_text())
active = state.get('active_job') or ''
active_dir = r/'work_dirs'/active
def tail(path, n=12):
    return path.read_text(errors='replace').splitlines()[-n:] if path.exists() else []
ps = subprocess.check_output(['ps', '-eo', 'pid,etime,pcpu,stat,args'], text=True)
stage_logs = list((r/'reports/finish').glob('*.log'))
stage_log = max(stage_logs, key=lambda p:p.stat().st_mtime) if stage_logs else None
calibration = {}
for micro in (4, 8):
    p = r/('reports/speed/H1_ddr_finetune_m%d_w4.log' % micro)
    if p.exists(): calibration[str(micro)] = tail(p, 5)
choice = r/'reports/speed/H_DDR_selection.json'
result = {
    'checked_at_epoch': time.time(),
    'gpu': subprocess.check_output(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu', '--format=csv,noheader'], text=True).strip(),
    'memory_current_bytes': int(pathlib.Path('/sys/fs/cgroup/memory.current').read_text()),
    'memory_max_bytes': int(pathlib.Path('/sys/fs/cgroup/memory.max').read_text()),
    'disk_free_bytes': shutil.disk_usage(r).free,
    'campaign': state,
    'finish': json.loads((r/'work_dirs/finish_state.json').read_text()),
    'processes': [x for x in ps.splitlines() if any(k in x for k in ['campaign/run_campaign.py', 'campaign/finish_campaign.py', 'train.py --config', 'tools/gpu_preflight.py', 'tools/export_champion.py', 'tools/export_predictions.py', 'tools/crossfit_fusion.py', 'tools/final_eval_locked.py']) and 'python -c' not in x],
    'active_log_tail': tail(active_dir/'train.log'),
    'finish_log_path': str(stage_log) if stage_log else None,
    'finish_log_tail': tail(stage_log) if stage_log else [],
    'finish_log_age_seconds': time.time()-stage_log.stat().st_mtime if stage_log else None,
    'active_eval_history_tail': tail(active_dir/'eval_history.csv', 6),
    'speed_calibration_logs': calibration,
    'speed_selection': json.loads(choice.read_text()) if choice.exists() else None,
}
result['processes'] = sorted(set(result['processes'] + [x for x in ps.splitlines() if 'campaign/run_round2.py' in x and 'python -c' not in x]))
print(json.dumps(result, ensure_ascii=False))
'''


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--remote-root',default='/root/autodl-tmp/HRDecoder_Ready_20261002')
    ap.add_argument('--local-root',default=str(ROOT))
    args=ap.parse_args()
    check_code=REMOTE_CHECK.replace('/root/autodl-tmp/HRDecoder_Ready_20261002',args.remote_root)
    password = os.environ.pop('HRD_MONITOR_PASSWORD', None)
    if password is None:
        password = getpass.getpass('SSH password: ')
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect('connect.nmb1.seetacloud.com', port=12191, username='root',
                       password=password, timeout=20, allow_agent=False, look_for_keys=False)
        password = None
        _, out, err = client.exec_command('/root/miniconda3/bin/python -c ' +
                                           shlex.quote(check_code), timeout=30)
        stdout = out.read().decode('utf-8')
        stderr = err.read().decode('utf-8', 'replace')
        if out.channel.recv_exit_status():
            raise RuntimeError(stderr)
        result = json.loads(stdout)
    finally:
        password = None
        client.close()
    target = Path(args.local_root)/'reports/monitor_health.json'
    target.parent.mkdir(parents=True,exist_ok=True)
    temp = target.with_suffix('.json.part')
    temp.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(target)
    summary = {k: result[k] for k in ('checked_at_epoch', 'gpu', 'memory_current_bytes',
                                    'memory_max_bytes', 'disk_free_bytes', 'processes',
                                    'active_log_tail', 'active_eval_history_tail',
                                    'speed_calibration_logs', 'speed_selection')}
    summary.update(active_job=result['campaign'].get('active_job'),
                   campaign_status=result['campaign'].get('status'),
                   finish_status=result['finish'].get('status'))
    summary['active_log_tail'] = [line[:800] for line in summary['active_log_tail']]
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
