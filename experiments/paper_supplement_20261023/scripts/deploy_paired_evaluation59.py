"""Deploy new evaluation-only workers and its endpoint publisher, never training source."""
from datetime import datetime,timezone
import json
import shlex
from training_remote_control import connect,execute,ROOT,REMOTE,LAN

stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
files=['evaluate_paired_seed59.py','queue_paired_evaluation59.py','publish_paired_evaluation_jobs.py']
destination=LAN+'/evaluation_workspace/paper_supplement_20261023/scripts'
with connect() as c:
    with c.open_sftp() as s:
        for name in files:s.put(str(ROOT/'scripts'/name),REMOTE+'/scripts/'+name)
    code=r'''
from pathlib import Path
import subprocess,shlex,json
root=Path(ROOT_VALUE);dst=DEST_VALUE
opts=['-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-i','/root/.ssh/codex_supp_to59_20261023']
for name in ['evaluate_paired_seed59.py','queue_paired_evaluation59.py']:
 subprocess.run(['scp',*opts,'-P','40560',str(root/'scripts'/name),'studentuser001@127.0.0.1:'+dst+'/'+name],check=True)
code59="from pathlib import Path;import subprocess,json;r=Path('/home/studentuser001/paper_supplement_20261023');f=(r/'logs/paired_evaluation59_supervisor.log').open('ab',buffering=0);p=subprocess.Popen(['/usr/bin/python3','-u',str(r/'evaluation_workspace/paper_supplement_20261023/scripts/queue_paired_evaluation59.py')],stdout=f,stderr=subprocess.STDOUT,start_new_session=True);(r/'logs/paired_evaluation59_supervisor_pid.json').write_text(json.dumps({'pid':p.pid}));print(json.dumps({'paired_evaluation_queue_pid':p.pid}))"
subprocess.run(['ssh',*opts,'-p','40560','studentuser001@127.0.0.1','/usr/bin/python3 -c '+shlex.quote(code59)],check=True)
f=(root/('log_endpoint_publisher_'+STAMP_VALUE+'.txt')).open('ab',buffering=0)
p=subprocess.Popen(['/root/miniconda3/envs/hrdecoder_q_20261003/bin/python','-u',str(root/'scripts/publish_paired_evaluation_jobs.py')],stdout=f,stderr=subprocess.STDOUT,start_new_session=True,env=dict(__import__('os').environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'))
(root/'paired_endpoint_publisher_pid.json').write_text(json.dumps({'pid':p.pid}))
print(json.dumps({'endpoint_publisher_pid':p.pid}))
'''.replace('ROOT_VALUE',repr(REMOTE)).replace('DEST_VALUE',repr(destination)).replace('STAMP_VALUE',repr(stamp))
    result=execute(c,'/root/miniconda3/bin/python -c '+shlex.quote(code))
    assert result['exit_code']==0,result
    print(json.dumps(result))
