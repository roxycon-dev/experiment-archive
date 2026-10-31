"""Resume a SHA-verified full backup of one stopped run; never delete remotely."""
import argparse
import json
from pathlib import Path
import sys
import time

root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root))
from ops.remote import connect
from qlib.common import read_json,sha,write_json,utc_now

ap=argparse.ArgumentParser()
ap.add_argument('--arm',choices=['Q_DDR','Q_MIX'],default='Q_DDR')
ap.add_argument('--seed',choices=[42,2024],type=int,default=42)
args=ap.parse_args()
run_name=f'{args.arm}_seed{args.seed}'
receipt=root/'reports'/f'run_backup_{run_name}.json'
state_path=root/'reports'/f'backup_state_{run_name}.json'
started=time.perf_counter()
verified=[]
try:
    with connect() as client:
        command="""cd /root/autodl-tmp/HRDecoder_Q_20261003
/root/miniconda3/envs/hrdecoder_q_20261003/bin/python - <<'PY'
from pathlib import Path
import json,stat
from qlib.common import read_json,sha,write_json,utc_now
run=Path('runs')/RUN_NAME
status=read_json(run/'status.json')
assert status['stage'] in ('COMPLETE','SERIOUS_FAILURE_STOP','TECHNICAL_PAUSE'), 'Cannot freeze active-run backup'
files=[]
for p in sorted(run.rglob('*')):
    assert not p.is_symlink(), 'No unreviewed symlink in run backup'
    if p.is_file():
        assert not p.name.endswith(('.tmp','.download')), 'Incomplete file exists'
        files.append({'path':p.as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)})
plan={'run':RUN_NAME,'status':status,'files':files,'created_at':utc_now()}
write_json(Path('reports')/('backup_plan_'+RUN_NAME+'.json'),plan)
print(json.dumps(plan))
PY
""".replace('RUN_NAME',repr(run_name))
        _,stdout,stderr=client.exec_command(command,timeout=120)
        payload=stdout.read().decode()
        if stdout.channel.recv_exit_status()!=0:
            raise RuntimeError(stderr.read().decode()[-1000:])
        plan=json.loads(payload)
        write_json(root/'reports'/f'backup_plan_{run_name}.json',plan)
        total=sum(f['bytes'] for f in plan['files'])
        with client.open_sftp() as sftp:
            for item in plan['files']:
                target=root/'server_results'/item['path']
                target.parent.mkdir(parents=True,exist_ok=True)
                if not target.exists() or sha(target)!=item['sha256']:
                    temporary=target.with_suffix(target.suffix+'.download')
                    sftp.get('/root/autodl-tmp/HRDecoder_Q_20261003/'+item['path'],str(temporary))
                    if sha(temporary)!=item['sha256']:
                        raise ValueError('Downloaded SHA mismatch: '+item['path'])
                    temporary.replace(target)
                verified.append(item)
                done=sum(f['bytes'] for f in verified)
                state={'stage':'BACKING_UP','run':run_name,'verified_files':len(verified),
                       'expected_files':len(plan['files']),'verified_bytes':done,'expected_bytes':total,
                       'seconds':time.perf_counter()-started,'updated_at':utc_now()}
                write_json(state_path,state)
                write_json(receipt,{'complete':False,'run':run_name,'files':verified,'verified_at':utc_now()})
                if len(verified)%10==0:
                    print(f'{run_name} backup verified {len(verified)}/{len(plan["files"])} files, {done/1024**3:.2f}/{total/1024**3:.2f} GiB',flush=True)
    write_json(receipt,{'complete':True,'run':run_name,'files':verified,'verified_at':utc_now(),
        'total_bytes':total,'seconds':time.perf_counter()-started,'remote_deletion_performed':False})
    write_json(state_path,{**state,'stage':'BACKUP_SHA_VERIFIED','updated_at':utc_now()})
    print(f'{run_name} complete SHA-verified local backup: {total/1024**3:.2f} GiB',flush=True)
except BaseException as exc:
    write_json(state_path,{'stage':'BACKUP_ERROR','run':run_name,'verified_files':len(verified),
                         'reason':f'{type(exc).__name__}: {exc}','automatic_restart':False,'updated_at':utc_now()})
    raise
