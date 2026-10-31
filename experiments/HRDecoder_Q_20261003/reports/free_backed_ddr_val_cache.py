"""Explicit remote-only removal of regenerable Q_DDR NPZ caches, after both copies' SHA match."""
import base64
import hashlib
import json
import pathlib
import sys
from datetime import datetime, timezone

ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from ops.remote import connect

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''): h.update(b)
    return h.hexdigest()

def main():
    rows=[]
    for seed in (42,2024):
        receipt=json.loads((ROOT/f'reports/run_backup_Q_DDR_seed{seed}.json').read_text())
        assert receipt['complete']
        for row in receipt['files']:
            p=pathlib.PurePosixPath(row['path'])
            if p.suffix!='.npz' or p.parent.name not in ('val_u12000','val_u16000'): continue
            assert p.parts[:2]==('runs',f'Q_DDR_seed{seed}')
            local=(ROOT/'server_results'/p).resolve()
            assert local.is_relative_to((ROOT/'server_results').resolve())
            assert local.stat().st_size==row['bytes'] and sha(local)==row['sha256']
            rows.append({**row,'reason':'Regenerable late probability cache; complete local backup independently SHA-verified; keep every model, log and per-image metric'})
    assert len(rows)==304
    plan={'created_at':datetime.now(timezone.utc).isoformat(),'remote_root':'/root/autodl-tmp/HRDecoder_Q_20261003',
          'delete_files':rows,'local_files_deleted':False,'protected_assets':'All weights/checkpoints/last, data, streams, code/config, logs, counts, P0 caches'}
    (ROOT/'reports/qmix_remote_cache_cleanup_plan.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
    payload=base64.b64encode(json.dumps(plan).encode()).decode()
    code=r'''
import base64,hashlib,json,pathlib,shutil
from datetime import datetime,timezone
plan=json.loads(base64.b64decode('PAYLOAD'))
root=pathlib.Path(plan['remote_root']).resolve()
assert root==pathlib.Path('/root/autodl-tmp/HRDecoder_Q_20261003')
for seed in (42,2024):
 status=json.loads((root/f'runs/Q_DDR_seed{seed}/status.json').read_text())
 assert status['stage']=='COMPLETE' and status['completed']==16000
for pname in ('formal.pid','final_test.pid'):
 p=root/'reports'/pname
 if p.exists():
  proc=pathlib.Path('/proc')/p.read_text().strip()/'cmdline'
  if proc.exists():
   cmd=proc.read_bytes()
   assert b'train_run.py' not in cmd and b'run_locked_test.py' not in cmd
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()
checked=[]
for row in plan['delete_files']:
 rel=pathlib.PurePosixPath(row['path']);p=(root/rel).resolve()
 assert p.is_relative_to(root/'runs') and rel.suffix=='.npz' and rel.parent.name in ('val_u12000','val_u16000')
 assert rel.parts[1] in ('Q_DDR_seed42','Q_DDR_seed2024')
 assert p.stat().st_size==row['bytes'] and sha(p)==row['sha256']
 checked.append((p,row))
before=shutil.disk_usage(root).free
for p,row in checked:p.unlink()
receipt={'status':'COMPLETE','deleted_files':plan['delete_files'],'deleted_bytes':sum(r['bytes'] for _,r in checked),
 'remote_SHA_matched_local_backup':True,'local_files_deleted':False,'free_bytes_before':before,'free_bytes_after':shutil.disk_usage(root).free,
 'completed_at':datetime.now(timezone.utc).isoformat()}
(root/'reports/qmix_remote_cache_cleanup_receipt.json').write_text(json.dumps(receipt,indent=2))
print(json.dumps({k:v for k,v in receipt.items() if k!='deleted_files'}))
'''.replace('PAYLOAD',payload)
    command="/root/miniconda3/envs/hrdecoder_q_20261003/bin/python - <<'PY'\n"+code+"\nPY"
    with connect() as c:
        _,out,err=c.exec_command(command,timeout=240)
        stdout=out.read().decode();stderr=err.read().decode();ret=out.channel.recv_exit_status()
        if ret: raise RuntimeError(stderr)
        with c.open_sftp() as s:
            s.get('/root/autodl-tmp/HRDecoder_Q_20261003/reports/qmix_remote_cache_cleanup_receipt.json',str(ROOT/'reports/qmix_remote_cache_cleanup_receipt.json'))
        print(stdout,flush=True)

if __name__=='__main__':main()
