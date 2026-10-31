"""Incremental evidence transfer, full state backup on completion, verified SHA."""
import argparse
import json
from pathlib import Path
import shlex
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from qlib.common import read_json,sha,utc_now,write_json
from ops.remote import connect,command


def main(full=False):
    runtime=read_json(ROOT/'configs/runtime.local.json');remote=runtime['server']['remote_root']
    output=ROOT/'server_results';output.mkdir(exist_ok=True)
    code='''
from pathlib import Path
import hashlib,json
r=Path(%r);full=%r
def h(p):
 d=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):d.update(b)
 return d.hexdigest()
files=[]
for sub in ('reports','configs','runs'):
 for p in (r/sub).rglob('*'):
  if not p.is_file() or p.is_symlink():continue
  if p.suffix in ('.tmp','.download','.zip') or p.name in ('campaign.lock','driver.lock'):continue
  if p.suffix not in ('.json','.jsonl','.log','.md','.pth','.npz'):continue
  if not full and p.suffix in ('.pth','.npz'):continue
  # Endpoint checkpoint includes model/optimizer/RNG; avoid a redundant last.pth download.
  if p.name=='last.pth' and (p.parent/'status.json').exists():
   s=json.loads((p.parent/'status.json').read_text())
   if s.get('stage')=='COMPLETE' and (p.parent/f'checkpoint_u{s["completed"]}.pth').exists():continue
  files.append({'path':p.relative_to(r).as_posix(),'bytes':p.stat().st_size,'sha256':h(p)})
print(json.dumps({'files':files,'campaign':json.loads((r/'reports/campaign.json').read_text()) if (r/'reports/campaign.json').exists() else {},'full':full}))
'''%(remote,full)
    with connect() as client:
        inventory=json.loads(command(client,shlex.quote(runtime['server_python'])+" - <<'PY'\n"+code+"\nPY",timeout=180))
        with client.open_sftp() as sftp:
            for item in inventory['files']:
                target=(output/item['path']).resolve()
                assert target.is_relative_to(output.resolve())
                target.parent.mkdir(parents=True,exist_ok=True)
                if target.exists() and sha(target)==item['sha256']:continue
                temporary=target.with_suffix(target.suffix+'.download')
                sftp.get(remote+'/'+item['path'],str(temporary))
                if sha(temporary)!=item['sha256']:
                    if not full:continue  # A growing live log will be captured on the next check.
                    raise ValueError('Backup SHA differs: '+item['path'])
                temporary.replace(target)
    inventory.update(stage='FULL_BACKUP_VERIFIED' if full else 'SNAPSHOT',verified_at=utc_now())
    write_json(ROOT/('reports/final_backup_receipt.json' if full else 'reports/monitor_snapshot.json'),inventory)
    print(json.dumps({'stage':inventory['stage'],'files':len(inventory['files']),'campaign_stage':inventory['campaign'].get('stage')}))
    return inventory


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--full',action='store_true');args=ap.parse_args();main(args.full)
