"""Take the two authorized shared cards only when idle; never stop another process."""
from pathlib import Path
import fcntl
import json
import os
import subprocess
import time

TASK=Path('/home/studentuser001/paper_supplement_20261023')
ROOT=TASK/'evaluation_workspace/paper_supplement_20261023'
python=TASK/'legacy_env/bin/python'
status=TASK/'logs/common_cache59_queue.json'
jobs=[{'gpu':0,'branch':'A','cohorts':['canary','ddr','idrid']},
      {'gpu':1,'branch':'U','cohorts':['canary','ddr']}]
locks=[]
for j in jobs:
    f=(TASK/'logs'/('cache_gpu'+str(j['gpu'])+'.lock')).open('w')
    fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
log=(TASK/'logs/common_cache59_P0.log').open('ab',buffering=0)
environment=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='')
cpu=subprocess.Popen([str(python),'-u',str(ROOT/'scripts/cache_common_system59.py'),'--branch','P0'],stdout=log,stderr=subprocess.STDOUT,env=environment)
for j in jobs:j.update(stage='WAITING_FOR_IDLE_GPU',pid=None,process=None)
while True:
    gpu_text=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    resources={int(x[0]):(int(x[1]),int(x[2])) for x in [r.split(',') for r in gpu_text.splitlines()]}
    for j in jobs:
        proc=j['process']
        if proc is not None:
            rc=proc.poll()
            if rc is not None:
                j['stage']='COMPLETE' if rc==0 and all((ROOT/'results/common_cache59'/c/j['branch']/'completion.json').exists() for c in j['cohorts']) else 'FAILED'
                j['exit_code']=rc
            continue
        if all((ROOT/'results/common_cache59'/c/j['branch']/'completion.json').exists() for c in j['cohorts']):
            j['stage']='COMPLETE';continue
        used,util=resources[j['gpu']]
        if used>256 or util>5:continue
        env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',CUDA_VISIBLE_DEVICES=str(j['gpu']))
        handle=(TASK/'logs'/('common_cache59_'+j['branch']+'.log')).open('ab',buffering=0)
        proc=subprocess.Popen([str(python),'-u',str(ROOT/'scripts/cache_common_system59.py'),'--branch',j['branch']],stdout=handle,stderr=subprocess.STDOUT,env=env)
        j.update(process=proc,pid=proc.pid,stage='RUNNING')
    state={'jobs':[{k:v for k,v in j.items() if k!='process'} for j in jobs],'P0_pid':cpu.pid,'P0_exit_code':cpu.poll(),'resources':resources,'updated_unix':time.time()}
    temp=status.with_suffix('.tmp');temp.write_text(json.dumps(state,indent=2));temp.replace(status)
    if all(j['stage'] in ('COMPLETE','FAILED') for j in jobs) and cpu.poll() is not None:break
    time.sleep(15)
