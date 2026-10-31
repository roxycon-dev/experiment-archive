set -eu
ROOT=/root/autodl-tmp/HRDecoder_Q_20261003
cd "$ROOT"
/root/miniconda3/envs/hrdecoder_q_20261003/bin/python advance_campaign.py
/root/miniconda3/envs/hrdecoder_q_20261003/bin/python - <<'PY'
import json, os
from pathlib import Path
from qlib.common import read_json, utc_now, write_json
root=Path('.')
snapshot={'checked_at':utc_now(),'campaign':read_json('reports/campaign_state.json'),'runs':{},'storage':{}}
for directory in sorted((root/'runs').glob('Q_*_seed*')):
    state=read_json(directory/'status.json') if (directory/'status.json').exists() else {}
    nodes={}
    for path in sorted(directory.glob('val_u*/metrics.json')):
        r=read_json(path)
        nodes[str(r['update'])]={k:r[k]['miou_pct'] for k in ('single','fixed_fusion','p0')}
    snapshot['runs'][directory.name]={'state':state,'nodes':nodes}
pid_path=root/'reports/formal.pid'
if pid_path.exists():
    pid=int(pid_path.read_text())
    try:
        os.kill(pid,0)
        own_training=b'train_run.py' in Path(f'/proc/{pid}/cmdline').read_bytes()
    except (ProcessLookupError,FileNotFoundError):
        own_training=False
    snapshot['formal_pid']=pid
    snapshot['formal_process_alive']=own_training
for base in ('/root/autodl-tmp','/root/autodl-fs'):
    path=Path(base)
    if path.exists():
        stat=os.statvfs(path)
        snapshot['storage'][base]={'mounted':os.path.ismount(path),'available_bytes':stat.f_bavail*stat.f_frsize}
snapshot['recent_logs']={}
for name in snapshot['runs']:
    log=root/'reports'/('formal_'+name+'.log')
    if log.exists():
        snapshot['recent_logs'][name]=log.read_text(encoding='utf-8',errors='replace').splitlines()[-3:]
final_state=root/'reports/final_evaluation_state.json'
if final_state.exists():
    state=read_json(final_state)
    final={'state':state,'process_alive':False,'phases':{}}
    try:
        os.kill(state['pid'],0)
        final['process_alive']=b'run_locked_test.py' in Path(f'/proc/{state["pid"]}/cmdline').read_bytes()
    except (ProcessLookupError,FileNotFoundError):pass
    for name in ('p0_manifest','seed42','seed2024','summary'):
        path=root/'reports/final_test'/(name+'.json')
        if path.exists():
            x=read_json(path)
            final['phases'][name]={'status':x.get('status'),'completed_images':len(x.get('images',{})),
                'lock_sha256':x.get('lock_sha256'),'selected':x.get('selected') if x.get('status')=='COMPLETE' else None}
    log=root/'reports/final_evaluation.log'
    if log.exists():
        with log.open('rb') as f:
            f.seek(max(0,log.stat().st_size-12000))
            final['recent_log']=f.read().decode(errors='replace').splitlines()[-12:]
    snapshot['final_evaluation']=final
write_json('reports/heartbeat_snapshot.json',snapshot)
print(json.dumps(snapshot,ensure_ascii=False))
PY
