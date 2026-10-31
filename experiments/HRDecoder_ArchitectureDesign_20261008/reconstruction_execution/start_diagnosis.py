"""Immediately queue one useful GPU diagnosis under the renewed user authorization."""
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
        try:ftp.mkdir(REMOTE+'/reconstruction_execution')
        except OSError:pass
        ftp.put(str(HERE/'live_gradient_diagnosis.py'),REMOTE+'/reconstruction_execution/live_gradient_diagnosis.py')
    code='''import json,os,subprocess,py_compile,time
from pathlib import Path
root=Path('/root/autodl-tmp/HRDecoder_v3_20261008');here=root/'reconstruction_execution'
reports=root/'reports/reconstruction';reports.mkdir(exist_ok=True)
assert not (reports/'DIAGNOSIS_DISPATCHED.json').exists()
py_compile.compile(str(here/'live_gradient_diagnosis.py'),doraise=True)
with open(reports/'gradient_diagnosis.log','a',buffering=1) as log:
 p=subprocess.Popen(['/root/miniconda3/bin/python','-u',str(here/'live_gradient_diagnosis.py')],cwd=str(here),stdout=log,stderr=log,start_new_session=True,env=dict(os.environ,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='4'))
receipt={'status':'DISPATCHED','pid':p.pid,'time_unix':time.time(),'optimizer_updates':0,'purpose':'train-only gradient-conflict evidence for next architecture','shutdown':False}
(reports/'DIAGNOSIS_DISPATCHED.json').write_text(json.dumps(receipt,indent=2)+'\\n')
print(json.dumps(receipt))
'''
    _,out,err=client.exec_command('/root/miniconda3/bin/python -c '+shlex.quote(code))
    raw,error=out.read().decode(),err.read().decode()
    if out.channel.recv_exit_status():raise RuntimeError(error)
    value=json.loads(raw)
    (HERE/'DIAGNOSIS_DISPATCHED.json').write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(value))
finally:client.close()
