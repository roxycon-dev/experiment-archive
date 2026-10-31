"""Upload the reviewed bounded recovery controller and dispatch it once."""
import hashlib
import json
from pathlib import Path
import shlex
import sys
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(ROOT/'local_preparation'))
from remote import connect
REMOTE='/root/autodl-tmp/HRDecoder_v3_20261008'
client=connect()
try:
    with client.open_sftp() as ftp:
        ftp.put(str(HERE/'recovery_controller.py'),REMOTE+'/sacd_micro_execution/recovery_controller.py')
    code='''import json,os,subprocess,py_compile
from pathlib import Path
root=Path('/root/autodl-tmp/HRDecoder_v3_20261008')
here=root/'sacd_micro_execution';reports=root/'reports/sacd_micro'
for name in ('probe.py','recovery_controller.py'):py_compile.compile(str(here/name),doraise=True)
assert not (reports/'MICRO_RECOVERY_DISPATCHED.json').exists()
with (reports/'recovery_controller.log').open('a',buffering=1) as log:
 p=subprocess.Popen(['/root/miniconda3/bin/python','-u',str(here/'recovery_controller.py')],cwd=str(here),stdout=log,stderr=log,start_new_session=True,env=dict(os.environ,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='4'))
receipt={'status':'RECOVERY_DISPATCHED','controller_pid':p.pid,'parent_weights_unchanged':True,'registered_gates_unchanged':True}
(reports/'MICRO_RECOVERY_DISPATCHED.json').write_text(json.dumps(receipt,indent=2)+'\\n')
print(json.dumps(receipt))
'''
    _,out,err=client.exec_command('/root/miniconda3/bin/python -c '+shlex.quote(code))
    raw,error=out.read().decode(),err.read().decode()
    if out.channel.recv_exit_status():raise RuntimeError(error)
    receipt=json.loads(raw)
    receipt['new_probe_source_SHA256']=hashlib.sha256((HERE/'probe.py').read_bytes()).hexdigest()
    (HERE/'MICRO_RECOVERY_DISPATCHED.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))
finally:client.close()
