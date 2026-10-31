"""Dispatch only ready endpoints to idle GPUs; CPU fusion releases its GPU for the next job."""
from pathlib import Path
import fcntl
import json
import os
import subprocess
import time

TASK=Path('/home/studentuser001/paper_supplement_20261023')
ROOT=TASK/'evaluation_workspace/paper_supplement_20261023'
lock=(TASK/'logs/paired_evaluation59.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
status=TASK/'logs/paired_evaluation59_queue.json'
order=[(a,s) for s in (42,2024,3407) for a in ('C','S')]
running={};finished={};failed={}
while True:
    for name,(proc,gpu) in list(running.items()):
        rc=proc.poll()
        if rc is None:continue
        run=ROOT/'results/paired_seed59'/name
        if rc==0 and (run/'completion.json').exists():finished[name]={'GPU':gpu,'exit_code':rc}
        else:failed[name]={'GPU':gpu,'exit_code':rc}
        del running[name]
    resources={}
    for line in subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).splitlines():
        gpu,used,util=map(int,line.split(','));resources[gpu]={'memory_used_MiB':used,'utilization_pct':util}
    waiting=[]
    if not failed and (TASK/'logs/evaluation_environment.json').exists():
        for arm,seed in order:
            name=arm+'_seed'+str(seed);run=ROOT/'results/paired_seed59'/name
            if name in running or name in finished:continue
            if (run/'completion.json').exists():finished[name]={'previously_completed':True};continue
            if (run/'failure.json').exists():failed[name]={'previous_failure_requires_diagnosed_recovery':True};break
            job=ROOT/'eval_jobs'/(name+'.json')
            if not job.exists():waiting.append({'name':name,'reason':'20k endpoint not ready'});continue
            occupied=set()
            common_status=TASK/'logs/common_cache59_queue.json'
            if common_status.exists():
                occupied.update(j['gpu'] for j in json.loads(common_status.read_text()).get('jobs',[]) if j['stage'] in ('RUNNING','WAITING_FOR_IDLE_GPU'))
            baseline_status=TASK/'logs/baseline_queue_status.json'
            if baseline_status.exists() and json.loads(baseline_status.read_text()).get('stage')=='BASELINE_TRAINING':occupied.update((2,3))
            for other,(proc,gpu) in running.items():
                p=ROOT/'results/paired_seed59'/other/'status.json'
                stage=json.loads(p.read_text()).get('stage') if p.exists() else None
                if stage!='CPU_FUSION':occupied.add(gpu)
            idle=[g for g,r in sorted(resources.items()) if r['memory_used_MiB']<256 and r['utilization_pct']<=5 and g not in occupied]
            if not idle:waiting.append({'name':name,'reason':'no idle GPU'});continue
            gpu=idle[0]
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
            log=(TASK/'logs'/('paired_eval_'+name+'.log')).open('ab',buffering=0)
            proc=subprocess.Popen([str(TASK/'evaluation_env/bin/python'),'-u',str(ROOT/'scripts/evaluate_paired_seed59.py'),'--job',str(job)],stdout=log,stderr=subprocess.STDOUT,env=env)
            running[name]=(proc,gpu);resources[gpu]['memory_used_MiB']=99999
    state={'stage':'TECHNICAL_PAUSE' if failed else ('COMPLETE_EVALUATION_RECOVERY_PENDING' if len(finished)==6 else 'RUNNING_OR_WAITING_READY_ENDPOINTS'),
           'running':{n:{'pid':p.pid,'GPU':g} for n,(p,g) in running.items()},'finished':finished,'failed':failed,'waiting':waiting,'resources':resources,'updated_unix':time.time()}
    temp=status.with_suffix('.tmp');temp.write_text(json.dumps(state,indent=2));temp.replace(status)
    if (failed and not running) or len(finished)==6:break
    time.sleep(15)
