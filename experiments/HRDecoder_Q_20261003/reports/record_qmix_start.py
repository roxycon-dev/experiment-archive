"""Read one bounded launch snapshot and persist actual progress, never treating a queue as training."""
import json
import pathlib
import sys
from datetime import datetime,timezone

ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from ops.remote import connect

def main():
    command="/root/miniconda3/envs/hrdecoder_q_20261003/bin/python - <<'PY'\n"+r'''
import json,pathlib,shutil,hashlib
r=pathlib.Path('/root/autodl-tmp/HRDecoder_Q_20261003')
paths=['reports/qmix_launcher_state.json','reports/qmix_formal_launch_receipt.json','reports/qmix_remote_asset_receipt.json',
 'reports/stream_Q_MIX_seed42.summary.json','runs/Q_MIX_seed42/status.json','runs/Q_MIX_seed42/identity.json','reports/campaign_state.json']
x={'files':{},'free_bytes':shutil.disk_usage(r).free,'process_alive':False,'current_run_val_nodes':{}}
for name in paths:
 p=r/name
 if p.exists():x['files'][name]=json.loads(p.read_text())
for p in (r/'runs/Q_MIX_seed42').glob('val_u*/metrics.json'):
 x['current_run_val_nodes'][p.parent.name.removeprefix('val_u')]=json.loads(p.read_text())
p=r/'reports/formal.pid'
if p.exists():
 pid=p.read_text().strip();c=pathlib.Path('/proc')/pid/'cmdline'
 x['formal_pid']=int(pid);x['process_alive']=c.exists() and b'train_run.py' in c.read_bytes()
for name in ('reports/qmix_launcher.log','reports/formal_Q_MIX_seed42.log'):
 p=r/name
 if p.exists():
  with p.open('rb') as f:f.seek(max(0,p.stat().st_size-4000));x[name]=f.read().decode('utf-8','replace')
print(json.dumps(x))
PY'''
    with connect() as c:
        _,out,err=c.exec_command(command,timeout=60)
        data=out.read().decode();error=err.read().decode();code=out.channel.recv_exit_status()
        if code:raise RuntimeError(error)
    x=json.loads(data);now=datetime.now(timezone.utc).isoformat();x['checked_at']=now
    (ROOT/'reports/qmix_start_actual_snapshot.json').write_text(json.dumps(x,indent=2),encoding='utf-8')
    files=x['files'];status=files.get('runs/Q_MIX_seed42/status.json',{})
    launcher=files.get('reports/qmix_launcher_state.json',{})
    actual_training=x['process_alive'] and status.get('stage')=='TRAINING' and status.get('completed',0)>0
    if actual_training:
        campaign_path=ROOT/'reports/campaign_state.json'
        prior=json.loads(campaign_path.read_text())
        if prior.get('arms')==['Q_DDR']:
            (ROOT/'reports/Q_DDR_completed_campaign_snapshot.json').write_text(json.dumps(prior,indent=2),encoding='utf-8')
        campaign_path.write_text(json.dumps(files['reports/campaign_state.json'],indent=2),encoding='utf-8')
        for remote_name,local_name in (
            ('reports/qmix_formal_launch_receipt.json','qmix_formal_launch_receipt.json'),
            ('reports/stream_Q_MIX_seed42.summary.json','stream_Q_MIX_seed42.server.summary.json'),
            ('runs/Q_MIX_seed42/identity.json','Q_MIX_seed42_actual_identity.json')):
            (ROOT/'reports'/local_name).write_text(json.dumps(files[remote_name],indent=2),encoding='utf-8')
        path=ROOT/'reports/preparation_status.json';p=json.loads(path.read_text())
        if 'prior_Q_DDR_validation_state' not in p:
            p['prior_Q_DDR_validation_state']={k:p.get(k) for k in
                ('completed_val_nodes','discovery_val_nodes','discovery_complete','review_complete')}
        current_nodes={u:{formula:m[formula]['miou_pct'] for formula in ('single','fixed_fusion','p0')}
                       for u,m in x['current_run_val_nodes'].items()}
        p.update(stage='FORMAL_Q_MIX_SEED42_TRAINING',arm='Q_MIX',seed=42,endpoint=16000,
            Q_MIX='READY_FORMAL_STARTED',last_completed_update=status['completed'],formal_process_alive=True,
            formal_pid=x['formal_pid'],data_disk_available_bytes=x['free_bytes'],updated_at=now,
            formal_training_started=True,shutdown_allowed_now=False,next_action='COMPLETE_DISCOVERY_AT_COMMON_ENDPOINT',
            completed_val_nodes=current_nodes,discovery_val_nodes=current_nodes,
            discovery_complete=files['reports/campaign_state.json']['discovery_complete'],review_complete=False,
            Q_DDR_final_test_complete=True,continuation_final_test_complete=False,
            legacy_final_evaluation_fields_scope='Completed Q_DDR-only frozen stage')
        path.write_text(json.dumps(p,indent=2),encoding='utf-8')
        path=ROOT/'reports/original_plan_completion.json';p=json.loads(path.read_text())
        p['tasks']['official_external_supervision'].update(state='COMPLETE_ACCEPTED_WITH_DIRECT_USER_IDRID_SOURCE_OVERRIDE',
            manifest='reports/external_acceptance/manifest.server.accepted.json',remote_SHA_receipt='reports/qmix_remote_asset_receipt.json',visual_review_count=20)
        p['tasks']['Q_MIX_paired_experiment'].update(state='DISCOVERY_SEED42_TRAINING',protocol='reports/qmix_continuation_protocol.json',completed_update=status['completed'])
        p.update(updated_at=now,shutdown_allowed=False)
        path.write_text(json.dumps(p,indent=2),encoding='utf-8')
        path=ROOT/'reports/external_readiness.json';p=json.loads(path.read_text());p.update(Q_MIX='READY',updated_at=now)
        path.write_text(json.dumps(p,indent=2),encoding='utf-8')
    print(json.dumps({'actual_training':actual_training,'process_alive':x['process_alive'],'completed_update':status.get('completed'),
        'run_stage':status.get('stage'),'launcher_stage':launcher.get('stage'),'free_bytes':x['free_bytes'],
        'launcher_log_tail':x.get('reports/qmix_launcher.log'),'training_log_tail':x.get('reports/formal_Q_MIX_seed42.log')},indent=2),flush=True)

if __name__=='__main__':main()
