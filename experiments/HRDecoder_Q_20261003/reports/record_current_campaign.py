"""One bounded actual-state read for whichever paired run is active."""
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ops.remote import connect
from qlib.common import utc_now, write_json, read_json


def main():
    code = r'''
import pathlib,json,shutil
r=pathlib.Path('/root/autodl-tmp/HRDecoder_Q_20261003')
x=dict(statuses={},val_nodes={},free_bytes=shutil.disk_usage(r).free,formal_alive=False,files={})
for arm in ('Q_DDR','Q_MIX'):
 for seed in (42,2024):
  name=f'{arm}_seed{seed}';run=r/'runs'/name
  if (run/'status.json').exists():x['statuses'][name]=json.loads((run/'status.json').read_text())
  x['val_nodes'][name]={}
  for p in run.glob('val_u*/metrics.json'):
   m=json.loads(p.read_text())
   x['val_nodes'][name][p.parent.name.removeprefix('val_u')]={k:m[k] for k in ('single','fixed_fusion','p0')}
for name in ('reports/campaign_state.json','reports/qmix_launcher_state.json',
             'reports/qmix_formal_launch_receipt.json','reports/qmix16_next_run_receipt.json',
             'reports/ddr42_24k_next_run_receipt.json','reports/mix42_24k_next_run_receipt.json',
             'reports/ddr2024_24k_next_run_receipt.json',
             'reports/continuation_final_launch_receipt.json',
             'reports/continuation_final_evaluation_state.json',
             'reports/continuation_final_selection.json',
             'reports/continuation_final_test/summary.json',
             'reports/continuation_final_test/reuse_receipt.json'):
 p=r/name
 if p.exists():x['files'][name]=json.loads(p.read_text())
p=r/'reports/formal.pid'
if p.exists():
 pid=p.read_text().strip();cmd=pathlib.Path('/proc')/pid/'cmdline'
 x['formal_pid']=int(pid)
 if cmd.exists():
  value=cmd.read_bytes();x['formal_alive']=b'train_run.py' in value
  x['formal_command']=value.decode('utf-8','replace').replace(chr(0),' ')
receipt=x['files'].get('reports/qmix_formal_launch_receipt.json',{})
final=x['files'].get('reports/continuation_final_launch_receipt.json',{})
if final:
 pid=str(final['pid']);cmd=pathlib.Path('/proc')/pid/'cmdline'
 x['continuation_final_pid']=int(pid)
 x['continuation_final_alive']=cmd.exists() and b'run_continuation_locked_test.py' in cmd.read_bytes()
if receipt:
 name=f"{receipt['arm']}_seed{receipt['seed']}"
 x['current_run']=name
 p=r/'reports'/f'formal_{name}.log'
 if p.exists():
  with p.open('rb') as f:
   f.seek(max(0,p.stat().st_size-2000));x['current_log_tail']=f.read().decode('utf-8','replace')
for name in ('qmix_next_launcher.log','qmix_launcher.log','mix42_24k_launcher.log',
             'ddr2024_24k_launcher.log','mix2024_24k_launcher.log','continuation_final_evaluation.log'):
 p=r/'reports'/name
 if p.exists():
  with p.open('rb') as f:
   f.seek(max(0,p.stat().st_size-2000));x[name]=f.read().decode('utf-8','replace')
print(json.dumps(x))
'''
    with connect() as client:
        _, out, err = client.exec_command("/root/miniconda3/envs/hrdecoder_q_20261003/bin/python - <<'PY'\n"+code+"\nPY", timeout=60)
        data, error = out.read().decode(), err.read().decode()
        if out.channel.recv_exit_status():raise RuntimeError(error)
    x = json.loads(data)
    x['checked_at'] = utc_now()
    write_json(ROOT/'reports/current_campaign_actual_snapshot.json', x)
    campaign = x['files']['reports/campaign_state.json']
    write_json(ROOT/'reports/campaign_state.json', campaign)
    name = x.get('current_run')
    status = x['statuses'].get(name, {})
    launch = x['files'].get('reports/qmix_formal_launch_receipt.json', {})
    actual = x['formal_alive'] and status.get('stage') == 'TRAINING' and status.get('completed',0)>0
    p = read_json(ROOT/'reports/preparation_status.json')
    review_complete=all(x['statuses'].get(f'{arm}_seed2024',{}).get('stage')=='COMPLETE'
        and x['statuses'][f'{arm}_seed2024'].get('completed')==campaign['common_endpoint'] for arm in campaign['arms'])
    final_summary=x['files'].get('reports/continuation_final_test/summary.json',{})
    final_lock=x['files'].get('reports/continuation_final_selection.json',{})
    final_complete=final_summary.get('status')=='COMPLETE'
    p.update(stage='PAIRED_CAMPAIGN_TRAINING' if actual else 'PAIRED_CAMPAIGN_TRANSITION_OR_COMPLETE',
        arm=launch.get('arm'),seed=launch.get('seed'),endpoint=campaign['common_endpoint'],
        last_completed_update=status.get('completed'),formal_process_alive=x['formal_alive'],
        formal_pid=x.get('formal_pid'),last_heartbeat_at=x['checked_at'],
        data_disk_available_bytes=x['free_bytes'],updated_at=x['checked_at'],
        shutdown_allowed_now=False,next_action=campaign['next_action'],
        discovery_complete=campaign['discovery_complete'],review_complete=review_complete,
        Q_DDR_final_test_complete=True,
        continuation_final_test_complete=final_complete and final_lock.get('test_action')!='NO_TEST_NOT_ELIGIBLE',
        continuation_finalization_complete=final_complete,
        continuation_selection_frozen=bool(final_lock),
        legacy_final_evaluation_fields_scope='Completed Q_DDR-only frozen stage',
        campaign_statuses=x['statuses'],campaign_val_nodes=x['val_nodes'])
    if name in x['val_nodes']:
        nodes={u:{k:m[k]['miou_pct'] for k in ('single','fixed_fusion','p0')} for u,m in x['val_nodes'][name].items()}
        discovery_name=f"{launch['arm']}_seed42"
        discovery_nodes={u:{k:m[k]['miou_pct'] for k in ('single','fixed_fusion','p0')}
            for u,m in x['val_nodes'].get(discovery_name,{}).items()}
        p.update(completed_val_nodes=nodes,discovery_val_nodes=discovery_nodes)
    write_json(ROOT/'reports/preparation_status.json',p)
    p = read_json(ROOT/'reports/original_plan_completion.json')
    p['tasks']['Q_MIX_paired_experiment'].update(state='PAIRED_TRAINING_COMPLETE_FINALIZATION_PENDING' if review_complete else 'PAIRED_CAMPAIGN_IN_PROGRESS',
        current_run=name,common_endpoint=campaign['common_endpoint'],completed_update=status.get('completed'),
        discovery_complete=campaign['discovery_complete'],actual_training=actual)
    p.update(updated_at=x['checked_at'],shutdown_allowed=False)
    write_json(ROOT/'reports/original_plan_completion.json',p)
    print(json.dumps(dict(checked_at=x['checked_at'],current_run=name,actual_training=actual,
        current_status=status,free_bytes=x['free_bytes'],next_action=campaign['next_action']),ensure_ascii=False))


if __name__ == '__main__':main()
