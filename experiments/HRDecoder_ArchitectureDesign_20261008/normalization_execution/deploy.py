"""Upload only small new diagnostic sources; immutable server-side parent."""
import hashlib
import json
from pathlib import Path
import py_compile
import shlex
import sys
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
REMOTE='/root/autodl-tmp/HRDecoder_v3_20261008'
sys.path.insert(0,str(ROOT/'local_preparation'))
from remote import connect

files=[HERE/n for n in ('controller.py','probe.py','intervention.py','probe_protocol.json','verify_normalization.py')]
for p in files:
    if p.suffix=='.py':py_compile.compile(str(p),doraise=True)
sealed={'registered_before_probe_results':True,'files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        'no_automatic_fresh_or_extension':True,'old_expert_and_residual_rejections_unchanged':True}
(HERE/'SEALED_SOURCE.json').write_text(json.dumps(sealed,indent=2)+'\n')
client=connect()
try:
    with client.open_sftp() as ftp:
        for directory in ('normalization_execution','reports/residual_normalization','training_runs/residual_normalization'):
            try:ftp.mkdir(REMOTE+'/'+directory)
            except OSError:pass
        for p in files+[HERE/'SEALED_SOURCE.json']:ftp.put(str(p),REMOTE+'/normalization_execution/'+p.name)
    code='''import hashlib,json,os,subprocess,time
from pathlib import Path
r=Path('/root/autodl-tmp/HRDecoder_v3_20261008');h=r/'normalization_execution';p=r/'reports/residual_normalization'
assert not(p/'DISPATCHED.json').exists()
assert not Path('/proc/36273').exists() and not Path('/proc/36083').exists()
assert json.loads((r/'reports/expert/GPU_CAMPAIGN_OUTCOME.json').read_text())['status']=='SCREEN_REJECTED'
sealed=json.loads((h/'SEALED_SOURCE.json').read_text())
for n,d in sealed['files'].items():assert hashlib.sha256((h/n).read_bytes()).hexdigest()==d,n
parent=r/'training_runs/residual_normalization/parent_08000.pt'
assert not parent.exists();os.link(str(r/'training_runs/residual/last.pt'),str(parent))
env=dict(os.environ,PYTHONPATH=str(r/'vendor/champion_repo'),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='4')
with open(p/'controller.log','a',buffering=1) as f:
 q=subprocess.Popen(['/root/miniconda3/bin/python','-u','controller.py'],cwd=str(h),stdout=f,stderr=f,start_new_session=True,env=env)
receipt={'status':'CPU_GATE_THEN_PAIRED_PROBE','controller_pid':q.pid,'time_unix':time.time(),
 'parent_immutable_hardlink':str(parent),'own_stop_unix':1791568620,'automatic_fresh_training':False,
 'deadline_extension':False,'server_kept_running':True,'shutdown_command_issued':False}
(p/'DISPATCHED.json').write_text(json.dumps(receipt,indent=2)+'\\n');print(json.dumps(receipt))
'''
    _,output,error=client.exec_command('/root/miniconda3/bin/python -c '+shlex.quote(code))
    raw,err=output.read().decode(),error.read().decode();assert output.channel.recv_exit_status()==0,err
    receipt=json.loads(raw)
    (HERE/'DISPATCHED.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))
finally:client.close()
