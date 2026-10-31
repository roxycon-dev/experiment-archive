"""Non-training protocol fix before formal run identities exist; no GPU restart."""
import sys,json,shlex
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from qlib.common import read_json,write_json,sha,digest,utc_now
from ops.remote import connect,command
old=read_json(ROOT/'reports/source_identity.json')
files={rel:sha(ROOT/rel) for rel in old['files']}
assert [rel for rel in files if files[rel]!=old['files'][rel]]==['decision.py']
assert read_json(ROOT/'reports/endpoint_policy_acceptance.json')['status']=='PASS'
new={'sha256':digest(files),'files':files,'time':utc_now()}
runtime=read_json(ROOT/'configs/runtime.local.json');remote=runtime['server']['remote_root'];python=runtime['server_python']
worker='''import json,time,sys
from pathlib import Path
r=Path(%r);sys.path.insert(0,str(r))
from qlib.common import read_json,write_json,sha,utc_now,assert_source
old=%r;new=%r
archive=r/'reports/preparation_repairs/endpoint_policy_20261005'
for _ in range(14400):
    gate=read_json(r/'reports/gpu_readiness.json')
    if gate['status']=='READY':break
    if read_json(r/'reports/driver_state.json')['stage']=='TECHNICAL_PAUSE':raise RuntimeError('GPU acceptance paused; do not advance receipt')
    time.sleep(.5)
else:raise TimeoutError('GPU readiness did not complete')
assert not list((r/'runs').glob('*/identity.json')),'Formal identity already exists; require explicit diagnosis'
assert assert_source(r)['sha256']==new
assert gate['source_sha256']==old and set(gate['benchmarks'])=={'C','S','D','SD'}
assert all(row['complete_updates']==100 for row in gate['benchmarks'].values())
early=read_json(r/'reports/early_gpu_gates.json');assert early['status']=='PASS' and early['source_sha256']==old
write_json(archive/'gpu_readiness.original.json',gate);write_json(archive/'early_gpu_gates.original.json',early)
extension={'status':'PASS','only_changed_file':'decision.py','training_and_GPU_gate_files_unchanged':True,
    'original_validated_source_sha256':old,'source_sha256':new,'time':utc_now(),
    'reason':'Protocol section8 allows both completed paired20k endpoints after a later scientific stop; no missing32k is fabricated.'}
for name,receipt in [('early_gpu_gates.json',early),('gpu_readiness.json',gate)]:
    receipt.update(source_sha256=new,non_training_source_extension=extension)
    write_json(r/'reports'/name,receipt)
write_json(r/'reports/endpoint_policy_revision.json',extension)
print('ENDPOINT_POLICY_SOURCE_EXTENSION_PASS',flush=True)
'''%(remote,old['sha256'],new['sha256'])
with connect() as client:
    guard="from pathlib import Path\nimport sys\nr=Path(%r);sys.path.insert(0,str(r))\nfrom qlib.common import read_json,assert_source\nassert assert_source(r)['sha256']==%r\nassert read_json(r/'reports/driver_state.json')['stage']=='GPU_ACCEPTANCE'\nassert not (r/'reports/campaign.json').exists()\nassert not list((r/'runs').glob('*/identity.json'))\nprint('GUARD_PASS')"%(remote,old['sha256'])
    print(command(client,shlex.quote(python)+" - <<'PY'\n"+guard+"\nPY"))
    archive=ROOT/'reports/preparation_repairs/endpoint_policy_20261005';archive.mkdir(parents=True,exist_ok=True)
    remote_archive=remote+'/reports/preparation_repairs/endpoint_policy_20261005'
    command(client,'mkdir -p '+shlex.quote(remote_archive))
    with client.open_sftp() as sftp:
        sftp.get(remote+'/decision.py',str(archive/'decision.original.py'))
        assert sha(archive/'decision.original.py')==old['files']['decision.py']
        write_json(archive/'source_identity.original.json',old)
        sftp.put(str(archive/'decision.original.py'),remote_archive+'/decision.original.py')
        sftp.put(str(archive/'source_identity.original.json'),remote_archive+'/source_identity.original.json')
        for rel,body in [('reports/endpoint_policy_receipt_extension.py',worker),('reports/source_identity.json',json.dumps(new,ensure_ascii=False,indent=2))]:
            with sftp.open(remote+'/'+rel+'.incoming','w') as f:f.write(body)
        sftp.put(str(ROOT/'decision.py'),remote+'/decision.py.incoming')
        sftp.put(str(ROOT/'reports/endpoint_policy_acceptance.json'),remote+'/reports/endpoint_policy_acceptance.json')
        sftp.rename(remote+'/reports/endpoint_policy_receipt_extension.py.incoming',remote+'/reports/endpoint_policy_receipt_extension.py')
        # Guarded while GPU gate is active, before campaign imports decision.
        print(command(client,shlex.quote(python)+" - <<'PY'\n"+guard+"\nPY"))
        sftp.posix_rename(remote+'/decision.py.incoming',remote+'/decision.py')
        sftp.posix_rename(remote+'/reports/source_identity.json.incoming',remote+'/reports/source_identity.json')
    launch='cd '+shlex.quote(remote)+'\nnohup '+shlex.quote(python)+' -u reports/endpoint_policy_receipt_extension.py > reports/endpoint_policy_receipt_extension.log 2>&1 < /dev/null &\necho $!'
    pid=int(command(client,launch).strip())
write_json(ROOT/'reports/source_identity.json',new)
write_json(ROOT/'reports/endpoint_policy_revision_launch.json',{'pid':pid,'old_source_sha256':old['sha256'],'source_sha256':new['sha256'],
    'only_changed_file':'decision.py','formal_updates_existed_at_revision':False,'time':utc_now()})
print('ENDPOINT_POLICY_STAGED',pid,new['sha256'])
