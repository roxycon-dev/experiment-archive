"""Wait on file receipts only; authorize the existing guard after prior root review.

This process does not consume model quota, conduct experiments, or skip the
manual report review. It never runs shutdown itself.
"""
from pathlib import Path
import json,sys,time
sys.path.insert(0,str(Path(__file__).resolve().parent))
from mirror_remote import ROOT,LOCAL,REMOTE,connect,remote_exec,file_sha,write_json
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
G_LOCAL=['champion','U0_native_whole','U0_native_slide','U0_long2048_whole','U0_long2048_slide',
         'U1_long2048_slide','H1_square2048_whole','H1_square2048_whole_ordinary','U0_highres_native_slide']
PERSISTENT=['U1_native_whole','U1_native_slide','U1_long2048_whole']
def main():
 approval=read(ROOT/'reports/primary_review_approval.json');assert approval['status']=='PRIMARY_REVIEW_COMPLETE'
 report=Path(approval['report']);assert file_sha(report)==approval['report_sha256']
 completion=read(LOCAL/'reports/campaign_completion.json')
 assert completion['status']=='EXPERIMENTS_COMPLETE_BACKUP_REVIEW_PENDING'
 assert file_sha(LOCAL/'work/final_test.json')==approval['final_test_sha256']
 snapshot=read(ROOT/'reports/closed_node_metadata_manifest.json')
 assert file_sha(ROOT/'reports/closed_node_metadata_manifest.json')==approval['metadata_manifest_sha256']
 for rel,row in snapshot['files'].items():assert file_sha(Path(snapshot['root'])/rel)==row['sha256'],rel
 package=read(ROOT/'reports/reproducibility_package.json')
 assert file_sha(package['path'])==approval['package_sha256']==package['sha256']
 assets=read(ROOT/'reports/frozen_asset_local_map.json')['assets']
 historical=[dict(path=row['local_path'],sha256=row['sha256']) for row in assets.values()]
 historical.append(approval['original_deployment_lock'])
 for row in historical:assert file_sha(row['path'])==row['sha256']
 for rel,h in completion['required_files'].items():assert file_sha(LOCAL/rel)==h,rel
 stems=[x['sid'] for x in read(LOCAL/'work/G/current_deployment/metrics.json')['per_image']]
 assert len(set(stems))==76
 caches=['work/G/'+x for x in G_LOCAL]+['work/M_seed42/'+a+'/val_u1000' for a in ['M0','M1']]
 deadline=time.time()+3*3600
 while time.time()<deadline:
  assert not (ROOT/'reports/completion_ready.json').exists(),'A ready marker already exists; do not replace'
  manifest=read(ROOT/'reports/mirror_manifest.json');required={};missing=[];counts={}
  for cache in caches:
   valid=0
   for sid in stems:
    rel=cache+'/'+sid+'.npz';row=manifest.get(rel);p=LOCAL/rel
    if row and p.is_file() and p.stat().st_size==row['size']:
     assert len(row['sha256'])==64
     required[rel]=row['sha256'];valid+=1
    else:missing.append(rel)
   counts[cache]=valid
  if missing:
   write_json(ROOT/'reports/backup_verification.json',dict(status='WAITING_REQUIRED_PROBABILITIES',time=time.time(),
       probability_files_remaining=len(missing),complete_counts=counts,root_review_complete=True,
       model_and_node_metadata_verified=True,shutdown_attempted=False))
   time.sleep(60);continue
  preserve=read(LOCAL/'reports/persistent_nonwinner_G.json');assert preserve['status']=='PASS'
  assert sorted(x['cache'] for x in preserve['caches'])==sorted(PERSISTENT)
  payload={}
  for group in preserve['caches']:
   files={k:v for k,v in group['files'].items() if k.endswith('.npz')};assert len(files)==76
   for name,row in files.items():
    local_rel='work/G/'+group['cache']+'/'+name
    local=LOCAL/local_rel;m=manifest.get(local_rel)
    if m and local.is_file() and local.stat().st_size==m['size'] and m['sha256']==row['sha256']:
     required[local_rel]=row['sha256']
    else:payload[group['destination']+'/'+name]=row['sha256']
  # Recheck persistent copies of any negative G arrays not yet mirrored locally.
  c=connect()
  try:
   code="""from pathlib import Path
import hashlib,json
files=json.loads(PAYLOAD);r=Path('/root/autodl-tmp/HRDecoder_Next_20261003/cold_storage/G').resolve()
for name,h in files.items():
 p=Path(name).resolve();assert r in p.parents and p.is_file()
 d=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(4*1024**2),b''):d.update(chunk)
 assert d.hexdigest()==h,name
print(json.dumps(dict(status='PASS',persistent_probability_files=len(files))))
""".replace('PAYLOAD',repr(json.dumps(payload)))
   receipt=json.loads(remote_exec(c,"/root/miniconda3/bin/python - <<'PY'\n"+code+"\nPY",120))
  finally:c.close()
  for rel,h in completion['required_files'].items():assert file_sha(LOCAL/rel)==h,rel
  assert file_sha(report)==approval['report_sha256']
  ready=dict(status='REVIEWED_COMPLETE',user_authorized_shutdown=True,report=str(report),
      required_files=completion['required_files'],historical_assets=historical,
      required_probability_files=required,completion_sha256=file_sha(LOCAL/'reports/campaign_completion.json'),
      metadata_snapshot_manifest_sha256=approval['metadata_manifest_sha256'],
      primary_review_approval_sha256=file_sha(ROOT/'reports/primary_review_approval.json'),
      persistent_negative_probability_receipt=receipt,created_at=time.time())
  write_json(ROOT/'reports/backup_verification.json',dict(status='PASS',time=time.time(),
      critical_files_verified=len(completion['required_files']),historical_assets_verified=len(historical),
      probability_files_locally_sha_verified=len(required),closed_metadata_files_verified=len(snapshot['files']),
      persistent_negative_files_sha_verified=receipt['persistent_probability_files'],
      probability_validation='Exact server SHA matched at atomic download; receipt and final file size rechecked',
      report_review_complete=True,old_deployment_retained=True))
  write_json(ROOT/'reports/completion_ready.json',ready)
  print('REVIEWED_COMPLETE; existing shutdown guard will recheck server and dispatch authorized shutdown',flush=True)
  return
 raise TimeoutError('Required verified probability backup exceeded bounded 3-hour window')
if __name__=='__main__':main()
