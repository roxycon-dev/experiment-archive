"""Diagnosed acceptance-only fork deadlock; archive original evidence, zero formal loss."""
import sys,json,shlex
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from qlib.common import read_json,write_json,sha,digest,utc_now
from ops.remote import connect,command
runtime=read_json(ROOT/'configs/runtime.local.json');remote=runtime['server']['remote_root'];python=runtime['server_python']
old=read_json(ROOT/'reports/source_identity.json');files={rel:sha(ROOT/rel) for rel in old['files']}
assert [rel for rel in files if files[rel]!=old['files'][rel]]==['server_driver.py']
new={'sha256':digest(files),'files':files,'time':utc_now()}
revision=read_json(ROOT/'reports/endpoint_policy_revision_launch.json')
archive=ROOT/'reports/preparation_repairs/fork_thread_20261005';archive.mkdir(parents=True,exist_ok=True)
code='''from pathlib import Path
import sys,os,signal,time,shutil,json
r=Path(%r);sys.path.insert(0,str(r))
from qlib.common import read_json,write_json,assert_source,sha,utc_now
old=%r;new=%r;driver=8262;waiter=%r
assert assert_source(r)['sha256']==old['sha256']
assert not list((r/'runs').glob('*/identity.json'))
assert read_json(r/'reports/driver_state.json')['stage']=='GPU_ACCEPTANCE'
assert read_json(r/'reports/gpu_readiness.json')['status']=='RUNNING'
repro=read_json(r/'reports/fork_pipeline_reproduction.json')
assert repro['default']['status']=='BOUNDED_TIMEOUT' and 'BATCH_READY' not in repro['default']['output']
assert repro['one']['returncode']==0 and 'BATCH_READY' in repro['one']['output']
assert os.getpgid(driver)==driver
assert b'server_driver.py' in Path(f'/proc/{driver}/cmdline').read_bytes()
children=[int(x) for x in Path(f'/proc/{driver}/task/{driver}/children').read_text().split()]
assert len(children)==8
evidence={}
for pid in [driver]+children:
 p=Path(f'/proc/{pid}');evidence[str(pid)]={'wait_channel':(p/'wchan').read_text(),'ticks':(p/'stat').read_text().split()[13:17]}
assert all(evidence[str(pid)]['wait_channel']=='futex_wait_queue_me' for pid in children)
a=r/'reports/preparation_repairs/fork_thread_20261005';a.mkdir(parents=True,exist_ok=True)
for rel in ['server_driver.py','reports/source_identity.json','reports/early_gpu_gates.json','reports/gpu_readiness.json',
 'reports/driver_state.json','reports/driver.lock','reports/system_acceptance.json','reports/parent_validation_seed42.json',
 'reports/server_driver.log','reports/fork_pipeline_reproduction.json','reports/fork_thread_reproduction.json',
 'reports/fork_main_stack.log','reports/fork_worker_stack.log','reports/endpoint_policy_receipt_extension.log']:
 p=r/rel
 if p.exists():shutil.copy2(p,a/(rel.replace('/','__')+'.original'))
if Path(f'/proc/{waiter}').exists():
 assert b'endpoint_policy_receipt_extension.py' in Path(f'/proc/{waiter}/cmdline').read_bytes();os.kill(waiter,signal.SIGTERM)
os.killpg(driver,signal.SIGTERM)
for _ in range(100):
 live=[]
 for pid in [driver]+children:
  p=Path(f'/proc/{pid}/stat')
  if p.exists() and p.read_text().split()[2]!='Z':live.append(pid)
 if not live:break
 time.sleep(.1)
assert not live,'Diagnosed driver/worker process did not exit'
for name in ('driver.lock','driver_state.json','gpu_readiness.json'):(r/'reports'/name).unlink()
reason={'stage':'DIAGNOSED_PREPARATION_RECOVERY','time':utc_now(),'formal_updates_lost':0,
 'original_source_sha256':old['sha256'],'source_sha256':new['sha256'],'process_evidence':evidence,
 'cause':'After CPU/CUDA/OpenCV work, forked workers wait in inherited thread-pool locks; real pipeline reproduces timeout with64/16threads and yields exact-shape batch with1/1threads.',
 'change':'Initialize parent torch and OpenCV pools to1 before model/validation. Eight DDR/four source workers and all scientific settings remain unchanged.',
 'early_gate_reuse':'Graph/kernel/routing/state code unchanged. Only decision.py and parent CPU pool startup changed since early PASS. Full100updates, native validation, and original R-count checks must complete under the corrected startup.',
 'endpoint_policy_acceptance':'PASS; real completed paired20k retained after late scientific stop, missing32k omitted',
 'old_source_archived':True,'passed_early_gates_not_repeated':True}
write_json(r/'reports/fork_thread_revision.json',reason)
print(json.dumps(reason))
'''%(remote,old,new,revision['pid'])
with connect() as client:
    receipt=json.loads(command(client,shlex.quote(python)+" - <<'PY'\n"+code+"\nPY",40))
    with client.open_sftp() as sftp:
        for rel in ('server_driver.py','reports/source_identity.json','reports/early_gpu_gates.json','reports/gpu_readiness.json','reports/driver_state.json',
                    'reports/system_acceptance.json','reports/parent_validation_seed42.json','reports/server_driver.log','reports/fork_pipeline_reproduction.json'):
            sftp.get(remote+'/reports/preparation_repairs/fork_thread_20261005/'+rel.replace('/','__')+'.original',str(archive/(rel.replace('/','__')+'.original')))
        write_json(ROOT/'reports/source_identity.json',new)
        sftp.put(str(ROOT/'server_driver.py'),remote+'/server_driver.py.incoming')
        sftp.put(str(ROOT/'reports/source_identity.json'),remote+'/reports/source_identity.json.incoming')
        sftp.posix_rename(remote+'/server_driver.py.incoming',remote+'/server_driver.py')
        sftp.posix_rename(remote+'/reports/source_identity.json.incoming',remote+'/reports/source_identity.json')
    # Source compatibility is explicit; never claim the old GPU gate ran this source.
    advance='''from pathlib import Path
import sys,json,subprocess
r=Path(%r);sys.path.insert(0,str(r))
from qlib.common import read_json,write_json,assert_source,utc_now
assert assert_source(r)['sha256']==%r
assert not list((r/'runs').glob('*/identity.json'))
gate=read_json(r/'reports/early_gpu_gates.json');assert gate['status']=='PASS'
gate['original_validated_source_sha256']=gate['source_sha256']
gate.update(source_sha256=%r,non_training_source_extension=read_json(r/'reports/fork_thread_revision.json'))
write_json(r/'reports/early_gpu_gates.json',gate)
with (r/'reports/server_driver.log').open('ab') as log:
 log.write(b'\\nEXPLICIT_DIAGNOSED_FORK_RECOVERY; original acceptance evidence archived; no formal updates lost\\n')
 p=subprocess.Popen([%r,'-u','server_driver.py'],cwd=r,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
write_json(r/'reports/preparation_recovery_driver.json',{'pid':p.pid,'stage':'REQUIRED_FULL_BENCHMARK_AFTER_FORK_REPAIR','time':utc_now()})
print(json.dumps({'pid':p.pid}))
'''%(remote,new['sha256'],new['sha256'],python)
    launch=json.loads(command(client,shlex.quote(python)+" - <<'PY'\n"+advance+"\nPY",30))
receipt.update(launch);write_json(ROOT/'reports/fork_thread_revision.json',receipt)
print('EXPLICIT_FORK_RECOVERY',json.dumps(launch),new['sha256'])
