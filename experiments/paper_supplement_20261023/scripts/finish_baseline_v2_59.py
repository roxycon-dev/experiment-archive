"""Original validation-hook selection and one frozen evaluation of corrected baseline v2."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

TASK=Path('/home/studentuser001/paper_supplement_20261023');BASE=TASK/'baseline_v2'
ROOT=TASK/'evaluation_workspace/paper_supplement_20261023';STATUS=TASK/'logs/baseline_v2_queue_status.json'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def save(stage,**extra):
    target=STATUS.with_suffix('.tmp');target.write_text(json.dumps(dict(stage=stage,no_shutdown=True,
        overall_campaign_complete=False,updated_unix=time.time(),**extra),indent=2));target.replace(STATUS)


def main():
    assert json.loads(STATUS.read_text())['stage']=='BASELINE_V2_TRAINING_COMPLETE_EVALUATION_PENDING'
    work=BASE/'runs/baseline_seed42';best=list(work.glob('best_mIoU_iter_*.pth'));endpoint=work/'iter_40000.pth'
    assert len(best)==1 and endpoint.exists()
    selected=int(re.fullmatch(r'best_mIoU_iter_(\d+)\.pth',best[0].name).group(1));assert selected in range(4000,40001,4000)
    records=[]
    for log in sorted(work.glob('*.log.json')):
        last=None
        for line in log.read_text().splitlines():
            row=json.loads(line)
            if row.get('mode')=='train':last=row['iter']
            elif row.get('mode')=='val':records.append(dict(training_node=last,original_log_record=row,log=log.name))
    assert {r['training_node'] for r in records}==set(range(4000,40001,4000))
    job={'kind':'baseline_v2','checkpoint_path':str(best[0]),'checkpoint_sha256':sha(best[0]),
        'selected_node':selected,'training_finished_at':40000,'selected_by':'Original validation mIoU hook, never test',
        'config_sha256':sha(BASE/'baseline_reproduction_ddp2.py'),'all_validation_records':records,
        'endpoint_checkpoint_sha256':sha(endpoint),'restored_scale_ratio':[.75,1.25],
        'v1_role':'Rejected reconstruction retained, not the primary baseline'}
    path=ROOT/'eval_jobs/baseline_v2_seed42.json';assert not path.exists()
    path.write_text(json.dumps(job,indent=2))
    while True:
        resources={}
        for line in subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).splitlines():
            g,m,u=map(int,line.split(','));resources[g]=(m,u)
        free=[g for g in (2,3,1,0) if resources[g][0]<256 and resources[g][1]<=5]
        if free:break
        save('BASELINE_V2_WAITING_IDLE_EVALUATION_GPU',resources=resources);time.sleep(15)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(free[0]),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    save('BASELINE_V2_EVALUATING',GPU=free[0],selected_node=selected)
    with (TASK/'logs/baseline_v2_evaluation.log').open('wb') as handle:
        rc=subprocess.run([str(TASK/'legacy_env/bin/python'),'-u',str(ROOT/'scripts/evaluate_retrained_baseline_v2_59.py'),
            '--job',str(path)],env=env,stdout=handle,stderr=subprocess.STDOUT).returncode
    if rc:
        save('TECHNICAL_PAUSE',evaluation_returncode=rc);raise RuntimeError('Inspect corrected baseline evaluation failure')
    save('BASELINE_V2_EVALUATION_COMPLETE_LOCAL_RECOVERY_PENDING',selected_node=selected)


if __name__=='__main__':main()
