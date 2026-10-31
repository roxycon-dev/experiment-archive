"""Read-only early recovery of immutable completed-run binaries during later training."""
import json
from pathlib import Path
import shlex
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from ops.remote import connect,command
from qlib.common import read_json,sha,utc_now,write_json
from ops.transfer_acceleration import install
install()


def main():
    runtime=read_json(ROOT/'configs/runtime.local.json');remote=runtime['server']['remote_root']
    output=ROOT/'server_results';output.mkdir(exist_ok=True)
    code='''
import json,hashlib
from pathlib import Path
r=Path(%r);files=[];runs=[]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
for run in (r/'runs').iterdir():
 status=run/'status.json'
 if not run.is_dir() or not status.exists():continue
 s=json.loads(status.read_text())
 if s.get('stage')!='COMPLETE':continue
 assert (run/f'checkpoint_u{s["completed"]}.pth').exists()
 runs.append(run.name)
 for p in run.rglob('*'):
  if not p.is_file() or p.is_symlink() or p.suffix not in ('.pth','.npz') or p.name=='last.pth':continue
  files.append({'path':p.relative_to(r).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)})
print(json.dumps({'runs':runs,'files':files}))
'''%remote
    receipt_path=ROOT/'reports/early_completed_backup.json'
    with connect() as client:
        inventory=json.loads(command(client,shlex.quote(runtime['server_python'])+' -c '+shlex.quote(code),180))
        required=sum(item['bytes'] for item in inventory['files']
                     if not (output/item['path']).exists())
        if shutil.disk_usage(output).free<required+2*1024**3:
            raise OSError('Insufficient local space for verified early recovery')
        receipt={'stage':'RECOVERING','started_at':utc_now(),'runs':inventory['runs'],
                 'files':inventory['files'],'verified_files':0,'remote_files_deleted':False,
                 'purpose':'Overlap completed-result transfer with later GPU training'}
        write_json(receipt_path,receipt)
        with client.open_sftp() as sftp:
            for item in inventory['files']:
                target=(output/item['path']).resolve()
                assert target.is_relative_to(output.resolve())
                target.parent.mkdir(parents=True,exist_ok=True)
                if not target.exists() or sha(target)!=item['sha256']:
                    temporary=target.with_suffix(target.suffix+'.early.download')
                    if not (temporary.exists() and temporary.stat().st_size==item['bytes'] and sha(temporary)==item['sha256']):
                        sftp.get(remote+'/'+item['path'],str(temporary))
                    if sha(temporary)!=item['sha256']:
                        raise ValueError('Immutable completed-run backup SHA differs: '+item['path'])
                    temporary.replace(target)
                receipt['verified_files']+=1
                write_json(receipt_path,receipt)
                print(receipt['verified_files'],len(inventory['files']),item['path'],flush=True)
        receipt.update(stage='COMPLETED_RUN_BINARIES_VERIFIED',completed_at=utc_now())
        write_json(receipt_path,receipt)
        print(receipt['stage'],flush=True)


if __name__=='__main__':
    try:main()
    except BaseException as error:
        write_json(ROOT/'reports/early_completed_backup_error.json',{
            'stage':'RECOVERY_ERROR','time':utc_now(),'reason':str(error),
            'training_affected':False,'remote_files_deleted':False})
        raise
