"""Reuse protected inputs, recover only verified old probabilities, stage identical science."""
import sys,json,shlex,zipfile,shutil,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'reports'))
from qlib.common import read_json,write_json,sha,utc_now
from parallel_remote import connect_node2,command

remote='/root/autodl-tmp/HRDecoder_FinalDecision_20261005'
old=ROOT.parent.parent/'HRDecoder_NextPlan_20261004/execution'
receipt=read_json(old/'reports/final_backup_receipt.json');assert receipt['stage']=='FULL_BACKUP_VERIFIED'
rows=[f for f in receipt['files'] if f['path'].endswith('.npz') and f['path'].startswith(('runs/E_NEG_seed42/val_u24000/','runs/F4_seed42/val_u12000/','runs/F12_seed42/val_u12000/'))]
assert rows
# The preceding successful cleanup already verified every local/remote SHA.
assert read_json(ROOT/'reports/parallel_assignment_probe.json')['server']['assignment_confirmed']
with connect_node2() as c:
    code="""
import json,hashlib,shutil
from pathlib import Path
r=Path(%r);r.mkdir(parents=True,exist_ok=True);(r/'reports').mkdir(exist_ok=True)
assert not (r/'reports/driver.lock').exists()
old=Path('/root/autodl-tmp/HRDecoder_Finetune_20261004');rows=%r
if (r/'reports/parallel_space_cleanup.json').exists():
 previous=json.loads((r/'reports/parallel_space_cleanup.json').read_text());assert previous['stage']=='VERIFIED_OLD_PROBABILITIES_RELEASED';print(json.dumps({'stage':'EXISTING_CLEANUP_RECEIPT_REUSED','free_disk_bytes':shutil.disk_usage(r).free}));raise SystemExit(0)
eligible=[];missing=[]
for row in rows:
 p=(old/row['path']).resolve();assert p.is_relative_to(old) and p.suffix=='.npz'
 if not p.exists():missing.append(row['path']);continue
 assert p.stat().st_size==row['bytes']
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 assert h.hexdigest()==row['sha256'];eligible.append((p,row))
for p,row in eligible:p.unlink()
result={'stage':'VERIFIED_OLD_PROBABILITIES_RELEASED','files':[row for p,row in eligible],
 'already_missing':missing,'free_disk_bytes':shutil.disk_usage(r).free,
 'all_old_scripts_logs_models_preserved':True,'old_campaign_not_reopened':True}
(r/'reports/parallel_space_cleanup.json').write_text(json.dumps(result,indent=2))
print(json.dumps({'stage':result['stage'],'files_released':len(eligible),'free_disk_bytes':result['free_disk_bytes']}))
"""%(remote,rows)
    print(command(c,"/root/miniconda3/envs/hrdecoder_q_20261003/bin/python - <<'PY'\n"+code+"\nPY",300),flush=True)
    files=set(read_json(ROOT/'reports/source_identity.json')['files'])
    files.update(['reports/source_identity.json','configs/runtime.server.json','reports/pretraining_identity.json',
        'reports/data_identity.json','reports/TJDR_manifest.json','reports/cpu_acceptance.json',
        'reports/window_acceptance.json','reports/storage_plan.json','reports/stream_identity.json',
        'reports/stream_DDR_seed42.jsonl','reports/stream_TJDR_seed42.jsonl',
        'reports/stream_DDR_seed2024.jsonl','reports/stream_TJDR_seed2024.jsonl',
        'reports/stream_PARENT_seed2024.jsonl','reports/kernel_source_identity.json'])
    files.update(p.relative_to(ROOT).as_posix() for p in (ROOT/'configs').glob('*') if p.is_file() and p.name!='runtime.local.json' and p.name!='parallel.local.json')
    package=ROOT/'reports/parallel_node2_source.zip'
    with zipfile.ZipFile(package,'w',zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        for rel in sorted(files):z.write(ROOT/rel,rel)
    manifest={rel:sha(ROOT/rel) for rel in files};package_sha=sha(package)
    with c.open_sftp() as ftp:ftp.put(str(package),remote+'/parallel_source.zip')
    code="""
from pathlib import Path
import hashlib,zipfile,json
r=Path(%r);p=r/'parallel_source.zip'
assert hashlib.sha256(p.read_bytes()).hexdigest()==%r
with zipfile.ZipFile(p) as z:
 for n in z.namelist():assert (r/n).resolve().is_relative_to(r)
 z.extractall(r)
manifest=%r
for rel,h in manifest.items():assert hashlib.sha256((r/rel).read_bytes()).hexdigest()==h
p.unlink()
(r/'reports/parallel_source_deployment.json').write_text(json.dumps({'stage':'IDENTICAL_SOURCE_FILES_SHA_VERIFIED','files':manifest,'package_sha256':%r}))
print('IDENTICAL_SOURCE_FILES_SHA_VERIFIED')
"""%(remote,package_sha,manifest,package_sha)
    print(command(c,"/root/miniconda3/envs/hrdecoder_q_20261003/bin/python - <<'PY'\n"+code+"\nPY",120),flush=True)
    write_json(ROOT/'reports/parallel_node2_deployment.json',{'stage':'SOURCE_STAGED','time':utc_now(),'package_sha256':package_sha,'files':manifest,'science_source_sha256':read_json(ROOT/'reports/source_identity.json')['sha256'],'runtime_sha256':sha(ROOT/'configs/runtime.server.json')})
    package.unlink()
