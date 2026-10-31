"""One dispatcher for baseline and six fixed endpoints, adopting existing workers without restarting them."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

TASK=Path('/home/studentuser001/paper_supplement_20261023')
ROOT=TASK/'evaluation_workspace/paper_supplement_20261023'
BASE=TASK/'baseline_v1'
status=TASK/'logs/paired_evaluation59_queue.json'

def read(p):return json.loads(p.read_text()) if p.exists() else None

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()

def save(p,value):
    p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(value,indent=2));t.replace(p)

def run_dir(name):return ROOT/'results/baseline_retrained59' if name=='baseline_seed42' else ROOT/'results/paired_seed59'/name

class AdoptedProcess:
    def __init__(self,pid,name):self.pid=pid;self.name=name
    def poll(self):
        p=Path('/proc')/str(self.pid)
        if p.exists() and p.joinpath('stat').read_text().split()[2]!='Z':return None
        return 0 if (run_dir(self.name)/'completion.json').exists() else 1

def baseline_job():
    completed=read(TASK/'logs/baseline_queue_status.json')
    if not completed or completed['stage']!='BASELINE_TRAINING_COMPLETE_EVALUATION_PENDING':return None
    target=ROOT/'eval_jobs/baseline_seed42.json'
    if target.exists():return target
    work=BASE/'runs/baseline_seed42';best=list(work.glob('best_mIoU_iter_*.pth'))
    assert len(best)==1 and (work/'iter_40000.pth').exists(), 'Require the original hook-selected best and40k endpoint'
    selected=int(re.fullmatch(r'best_mIoU_iter_(\d+)\.pth',best[0].name).group(1));assert selected in range(4000,40001,4000)
    records=[]
    for log in sorted(work.glob('*.log.json')):
        last_train=None
        for text in log.read_text().splitlines():
            row=json.loads(text)
            if row.get('mode')=='train':last_train=row['iter']
            elif row.get('mode')=='val':records.append({'training_node':last_train,'original_log_record':row,'log':log.name})
    assert {r['training_node'] for r in records}==set(range(4000,40001,4000)),'Every prescribed validation node must be present'
    job={'kind':'baseline','checkpoint_path':str(best[0]),'checkpoint_sha256':sha(best[0]),'selected_node':selected,
         'training_finished_at':40000,'selected_by':'Original validation mIoU hook, never test',
         'config_sha256':sha(BASE/'baseline_reproduction_ddp2.py'),'all_validation_records':records,
         'endpoint_checkpoint_sha256':sha(work/'iter_40000.pth')}
    save(target,job);return target

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--adopt-state',type=Path);args=parser.parse_args()
    lock=(TASK/'logs/paired_evaluation59.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    running={};finished={};failed={}
    if args.adopt_state:
        previous=read(args.adopt_state);finished=previous.get('finished',{});failed=previous.get('failed',{})
        for name,row in previous.get('running',{}).items():
            proc=AdoptedProcess(row['pid'],name);path=Path('/proc')/str(proc.pid)/'cmdline'
            if path.exists():assert b'evaluate_paired_seed59.py' in path.read_bytes(), 'Adopt only our exact inference worker'
            running[name]=(proc,row['GPU'])
    order=['baseline_seed42']+[a+'_seed'+str(s) for s in (42,2024,3407) for a in ('C','S')]
    while True:
        for name,(proc,gpu) in list(running.items()):
            rc=proc.poll()
            if rc is None:continue
            if rc==0 and (run_dir(name)/'completion.json').exists():finished[name]={'GPU':gpu,'exit_code':rc}
            else:failed[name]={'GPU':gpu,'exit_code':rc}
            del running[name]
        resources={}
        for line in subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).splitlines():
            gpu,used,util=map(int,line.split(','));resources[gpu]={'memory_used_MiB':used,'utilization_pct':util}
        occupied=set()
        common=read(TASK/'logs/common_cache59_queue.json') or {}
        occupied.update(j['gpu'] for j in common.get('jobs',[]) if j['stage'] in ('RUNNING','WAITING_FOR_IDLE_GPU'))
        baseline=read(TASK/'logs/baseline_queue_status.json') or {}
        if baseline.get('stage')=='BASELINE_TRAINING':occupied.update((2,3))
        for other,(proc,gpu) in running.items():
            stage=(read(run_dir(other)/'status.json') or {}).get('stage')
            if stage!='CPU_FUSION':occupied.add(gpu)
        waiting=[]
        if not failed:
            for name in order:
                folder=run_dir(name)
                if name in running or name in finished:continue
                if (folder/'completion.json').exists():finished[name]={'previously_completed':True};continue
                if (folder/'failure.json').exists():failed[name]={'previous_failure_requires_diagnosed_recovery':True};break
                job=baseline_job() if name=='baseline_seed42' else ROOT/'eval_jobs'/(name+'.json')
                if not job or not job.exists():waiting.append({'name':name,'reason':'fixed training endpoint not ready'});continue
                if name!='baseline_seed42' and not (TASK/'logs/evaluation_environment.json').exists():continue
                idle=[g for g,r in sorted(resources.items()) if r['memory_used_MiB']<256 and r['utilization_pct']<=5 and g not in occupied]
                if not idle:waiting.append({'name':name,'reason':'no idle GPU'});continue
                gpu=idle[0];env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
                worker='evaluate_retrained_baseline59.py' if name=='baseline_seed42' else 'evaluate_paired_seed59.py'
                python=TASK/('legacy_env/bin/python' if name=='baseline_seed42' else 'evaluation_env/bin/python')
                log=(TASK/'logs'/('paired_eval_'+name+'.log')).open('ab',buffering=0)
                proc=subprocess.Popen([str(python),'-u',str(ROOT/'scripts'/worker),'--job',str(job)],stdout=log,stderr=subprocess.STDOUT,env=env)
                running[name]=(proc,gpu);occupied.add(gpu)
        state={'stage':'TECHNICAL_PAUSE' if failed else ('ALL_SEVEN_EVALUATIONS_COMPLETE_RECOVERY_PENDING' if len(finished)==7 else 'RUNNING_OR_WAITING_READY_ENDPOINTS'),
               'running':{n:{'pid':p.pid,'GPU':g} for n,(p,g) in running.items()},'finished':finished,'failed':failed,'waiting':waiting,
               'resources':resources,'updated_unix':time.time(),'controller':'supplement_v2_with_baseline_and_adopted_workers'}
        save(status,state)
        if (failed and not running) or len(finished)==7:break
        time.sleep(15)

if __name__=='__main__':main()
