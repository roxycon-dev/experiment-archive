"""Run the frozen baseline configuration correction on idle authorized GPUs, then evaluate it once."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import fcntl

TASK=Path('/home/studentuser001/paper_supplement_20261023');BASE=TASK/'baseline_v2'
EVAL=TASK/'evaluation_workspace/paper_supplement_20261023'
STATUS=TASK/'logs/baseline_v2_queue_status.json'
EXPECTED='d8b5c8f311952c17aff5eccb642f4f0d0a5554862deb36f399e5fbd2730207dc'


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8388608),b''):h.update(b)
    return h.hexdigest()


def save(value):
    value.update(updated_unix=time.time(),no_shutdown=True,overall_campaign_complete=False)
    temp=STATUS.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2));temp.replace(STATUS)


def main():
    lock=(TASK/'logs/baseline_v2.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    config=BASE/'baseline_reproduction_ddp2.py';assert sha(config)==EXPECTED
    work=BASE/'runs/baseline_seed42';assert not work.exists(),'Do not silently restart a partially executed correction'
    # Copy of the exact v1 source; the only recipe correction is in the new config.
    for rel in ('mmseg/models/segmentors/HRDecoder.py','mmseg/models/segmentors/lesion_encoder_decoder.py','tools/train.py'):
        assert sha(BASE/'repo'/rel)==sha(TASK/'baseline_v1/repo'/rel)
    while True:
        resources={}
        for line in subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).splitlines():
            g,m,u=map(int,line.split(','));resources[g]=(m,u)
        if all(resources[g][0]<256 and resources[g][1]<=5 for g in (2,3)):break
        save({'stage':'WAITING_IDLE_GPU2_3','resources':resources});time.sleep(15)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='2,3',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1')
    python=str(TASK/'legacy_env/bin/python')
    command=[python,'-m','torch.distributed.run','--nproc_per_node=2','--master_port=44241',
             str(BASE/'baseline_train_entry.py'),str(config),'--launcher','pytorch','--seed','42']
    save({'stage':'BASELINE_V2_TRAINING','global_batch':2,'FP32':True,'GPUs':[2,3],'updates':40000,
          'restored_scale_ratio':[.75,1.25],'config_sha256':EXPECTED,'attempt_v1_retained':True})
    with (TASK/'logs/baseline_v2_train.log').open('wb') as handle:
        rc=subprocess.run(command,cwd=BASE/'repo',env=env,stdout=handle,stderr=subprocess.STDOUT).returncode
    if rc:
        save({'stage':'TECHNICAL_PAUSE','returncode':rc,'recipe_automatically_changed':False});raise RuntimeError('Inspect v2 training failure; no automatic recipe change')
    save({'stage':'BASELINE_V2_TRAINING_COMPLETE_EVALUATION_PENDING','training_endpoint':40000})
    # Evaluation is a separate immutable job and result path; v1 files are retained.
    subprocess.run([python,str(EVAL/'scripts/finish_baseline_v2_59.py')],check=True)


if __name__=='__main__':main()
