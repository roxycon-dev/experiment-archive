"""Recover completed locked evaluation evidence from the assigned instance with SHA checks."""
import json
import sys
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ops.remote import connect
from qlib.common import sha, read_json, write_json, utc_now

REMOTE = '/root/autodl-tmp/HRDecoder_Q_20261003'
command = """/root/miniconda3/envs/hrdecoder_q_20261003/bin/python - <<'PY'
import json,hashlib
from pathlib import Path
r=Path('/root/autodl-tmp/HRDecoder_Q_20261003')
state=json.loads((r/'reports/final_evaluation_state.json').read_text())
assert state['stage']=='COMPLETE' and state['phase']=='COMPLETE'
required=['reports/final_selection.json','reports/final_evaluation_state.json',
 'reports/final_evaluation.log','reports/final_evaluation.py','reports/run_locked_test.py',
 'reports/campaign_state.json','reports/heartbeat_snapshot.json','reports/p0_historical_test_reference.json',
 'reports/source_identity.json','reports/pretraining_identity.json','reports/data_identity.json',
 'configs/runtime.server.json','configs/protocol.json','configs/p0_original_lock.json','configs/p0_identity.json']
required+=['reports/final_test/'+name+'.json' for name in ['p0_manifest','seed42','seed2024','summary']]
files=[]
for name in required:
 p=r/name
 assert p.is_file(),name
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 files.append({'path':name,'bytes':p.stat().st_size,'sha256':h.hexdigest()})
print(json.dumps({'stage':'COMPLETE','lock_sha256':state['lock_sha256'],'files':files}))
PY
"""
started = time.perf_counter()
with connect() as client:
    _, out, err = client.exec_command(command, timeout=60)
    stdout, stderr = out.read().decode(), err.read().decode()
    if out.channel.recv_exit_status():
        raise RuntimeError(stderr)
    manifest = json.loads(stdout)
    assert manifest['lock_sha256'] == sha(ROOT/'reports/final_selection.json')
    with client.open_sftp() as sftp:
        for item in manifest['files']:
            local = ROOT/'server_results'/item['path']
            local.parent.mkdir(parents=True, exist_ok=True)
            if local.exists() and local.stat().st_size == item['bytes'] and sha(local) == item['sha256']:
                continue
            temporary = local.with_suffix(local.suffix+'.downloadtemp')
            sftp.get(REMOTE+'/'+item['path'], str(temporary))
            assert temporary.stat().st_size == item['bytes'] and sha(temporary) == item['sha256'], item['path']
            temporary.replace(local)

for seed in (42, 2024):
    result = read_json(ROOT/f'server_results/reports/final_test/seed{seed}.json')
    assert result['status']=='COMPLETE' and len(result['images'])==227
    assert result['lock_sha256']==manifest['lock_sha256']
summary = read_json(ROOT/'server_results/reports/final_test/summary.json')
assert summary['status']=='COMPLETE' and not summary['selection_changed_after_test']
manifest.update(local_backup_state='SHA_VERIFIED',verified_files=len(manifest['files']),
                completed_at=utc_now(),seconds=time.perf_counter()-started)
write_json(ROOT/'reports/final_evidence_backup_receipt.json', manifest)

snapshot=read_json(ROOT/'server_results/reports/heartbeat_snapshot.json')
write_json(ROOT/'reports/heartbeat_snapshot.json', snapshot)
write_json(ROOT/'reports/campaign_state.json', snapshot['campaign'])
status=read_json(ROOT/'reports/preparation_status.json')
status.update(stage='Q_DDR_FINAL_EVALUATION_COMPLETE_ORIGINAL_PLAN_CONTINUES',
    updated_at=utc_now(),last_heartbeat_at=snapshot['checked_at'],
    final_evaluation_checked_at=snapshot['checked_at'],final_evaluation_phase='COMPLETE',
    final_test_complete=True,final_test_scores_pct=summary['seed_scores_pct'],
    final_mean_test_pct=summary['mean_test_pct'],final_evidence_sha_verified=True,
    review_backup_sha_verified=True,formal_process_alive=False,
    data_disk_available_bytes=snapshot['storage']['/root/autodl-tmp']['available_bytes'],
    shutdown_allowed_now=False,original_plan_complete=False)
write_json(ROOT/'reports/preparation_status.json', status)
scope=read_json(ROOT/'reports/original_plan_completion.json')
scope['updated_at_utc']=utc_now()
scope['tasks']['locked_Q_DDR_final_evaluation'].update(state='COMPLETE_LOCAL_SHA_VERIFIED',
    receipt='reports/final_evidence_backup_receipt.json',seed_scores_pct=summary['seed_scores_pct'])
scope['tasks']['Q_DDR_training_backups']={'state':'COMPLETE_SHA_VERIFIED',
    'receipts':['reports/run_backup_Q_DDR_seed42.json','reports/run_backup_Q_DDR_seed2024.json']}
write_json(ROOT/'reports/original_plan_completion.json',scope)
print(json.dumps({'final_evidence_sha_verified':True,'files':len(manifest['files']),
    'test_scores_pct':summary['seed_scores_pct'],'mean_test_pct':summary['mean_test_pct'],
    'shutdown_allowed':False,'seconds':time.perf_counter()-started}))
