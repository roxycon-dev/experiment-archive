"""Two bounded receivers for immutable endpoints throughout training; full SHA gate."""
import sys,json,shlex,os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from qlib.common import read_json,write_json,sha,utc_now
from ops.remote import connect,command

def main(full=False,endpoints=False):
    runtime=read_json(ROOT/'configs/runtime.local.json');remote=runtime['server']['remote_root'];output=ROOT/'server_results';output.mkdir(exist_ok=True)
    lock=ROOT/'reports/mirror.lock'
    try:
        with lock.open('x',encoding='utf-8') as f:json.dump({'pid':os.getpid(),'time':utc_now()},f)
    except FileExistsError:return {'stage':'EXISTING_RECEIVER_LOCK','action':'Review recorded PID; never duplicate a healthy receiver'}
    try:
        code="""
from pathlib import Path
import hashlib,json
r=Path(%r);full=%r;endpoints=%r
def h(p):
 d=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):d.update(b)
 return d.hexdigest()
campaign=json.loads((r/'reports/campaign.json').read_text()) if (r/'reports/campaign.json').exists() else {}
if full:
 assert campaign.get('stage')=='READY_FOR_LOCAL_BACKUP_AND_SHUTDOWN'
 assert not (r/'reports/driver.lock').exists() and not (r/'reports/campaign.lock').exists()
files=[]
for sub in ('reports','configs','runs','qlib','ops','tests','vendor/msda_compat','vendor/msda_build'):
 for p in (r/sub).rglob('*'):
  if not p.is_file() or p.is_symlink() or p.suffix in ('.tmp','.download','.zip','.pyc') or p.name.endswith('.lock'):continue
  if p.suffix not in ('.json','.jsonl','.log','.md','.pth','.npz','.py','.cpp','.cu','.h','.so','.jpg','.png'):continue
  binary=p.suffix in ('.pth','.npz','.so')
  if binary and not full:
   if not endpoints:continue
   if p.suffix=='.npz' and not (p.parent/'metrics.json').exists():continue
   if p.suffix=='.pth' and not p.name.startswith('checkpoint_u'):continue
   if p.suffix=='.so':continue
  if p.name=='last.pth' and (p.parent/'status.json').exists():
   s=json.loads((p.parent/'status.json').read_text())
   cold=r/'reports/storage_offload'/f'{p.parent.name}_u{s.get("completed")}.json'
   if s.get('stage')=='COMPLETE' and ((p.parent/f'checkpoint_u{s["completed"]}.pth').exists() or cold.exists()):continue
  files.append({'path':p.relative_to(r).as_posix(),'bytes':p.stat().st_size,'sha256':h(p)})
for p in r.glob('*.py'):files.append({'path':p.name,'bytes':p.stat().st_size,'sha256':h(p)})
print(json.dumps({'files':files,'campaign':campaign,'full':full,'endpoints':endpoints}))
"""%(remote,full,endpoints)
        with connect() as client:inventory=json.loads(command(client,shlex.quote(runtime['server_python'])+" - <<'PY'\n"+code+"\nPY",300))
        def recover(item):
            target=(output/item['path']).resolve();assert target.is_relative_to(output.resolve());target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists() and sha(target)==item['sha256']:return item
            temporary=target.with_suffix(target.suffix+'.download')
            with connect() as client:
                with client.open_sftp() as sftp:sftp.get(remote+'/'+item['path'],str(temporary),prefetch=True,max_concurrent_prefetch_requests=64)
            if sha(temporary)!=item['sha256']:
                if full:raise ValueError('Final SHA differs: '+item['path'])
                return None # Active status/log changed during copying: not training failure.
            temporary.replace(target);return item
        verified=[]
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(recover,item) for item in inventory['files']]
            for future in as_completed(futures):
                item=future.result()
                if item:verified.append(item)
                write_json(ROOT/'reports/transfer_progress.json',{'stage':'TRANSFERRING','full':full,'verified_files':len(verified),'expected_files':len(inventory['files']),
                    'verified_bytes':sum(r['bytes'] for r in verified),'expected_bytes':sum(r['bytes'] for r in inventory['files']),'updated_at':utc_now()})
        if full:
            assert len(verified)==len(inventory['files'])
            # Recoveries released from the remote disk are already stored once
            # in this same output tree. Include and recheck every cold artifact.
            bypath={r['path']:r for r in verified}
            for p in (ROOT/'reports/storage_offload').glob('*.json'):
                receipt=read_json(p);assert receipt['stage']=='LOCAL_SHA_VERIFIED_REMOTE_RELEASED'
                assert receipt['server_instance']==runtime['server']['instance_id']
                for row in receipt['files']:
                    assert sha(output/row['path'])==row['sha256']
                    if row['path'] in bypath:assert bypath[row['path']]['sha256']==row['sha256']
                    bypath[row['path']]=row
            verified=list(bypath.values())
        elif endpoints:
            from ops.cold_archive import release_verified,restore_selected
            release_verified();restore_selected()
        inventory.update(stage='FULL_BACKUP_VERIFIED' if full else 'SNAPSHOT',files=verified,verified_at=utc_now())
        write_json(ROOT/('reports/final_backup_receipt.json' if full else 'reports/monitor_snapshot.json'),inventory)
        print(json.dumps({'stage':inventory['stage'],'files':len(verified),'campaign_stage':inventory['campaign'].get('stage')}));return inventory
    finally:lock.unlink(missing_ok=True)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--full',action='store_true');p.add_argument('--endpoints',action='store_true');a=p.parse_args();main(a.full,a.endpoints)
