"""Start the preregistered FP32 baseline once assets/environment pass; never change its recipe."""
import json
import os
from pathlib import Path
import subprocess
import time

ROOT=Path('/home/studentuser001/paper_supplement_20261023');BASE=ROOT/'baseline_v1'
started=time.monotonic()
required=[ROOT/'logs/environment_ready.json',ROOT/'assets/ddr_split613.tar',ROOT/'assets/hrnetv2_w48-d2186c55.pth']
status=ROOT/'logs/baseline_queue_status.json'
while not all(p.exists() for p in required):
    status.write_text(json.dumps({'stage':'WAITING_VERIFIED_ASSETS_AND_ENVIRONMENT','missing':[p.name for p in required if not p.exists()]}))
    if time.monotonic()-started>7200:raise TimeoutError('Baseline preparation exceeded2hours; inspect install/transfer logs')
    time.sleep(10)
subprocess.run(['/usr/bin/python3',str(BASE/'accept_baseline59_data.py')],check=True)
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='2,3',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1')
python=ROOT/'legacy_env/bin/python'
command=[str(python),'-m','torch.distributed.run','--nproc_per_node=2','--master_port=44239',
         str(BASE/'baseline_train_entry.py'),str(BASE/'baseline_reproduction_ddp2.py'),'--launcher','pytorch','--seed','42']
status.write_text(json.dumps({'stage':'BASELINE_TRAINING','global_batch':2,'FP32':True,'GPUs':[2,3],'updates':40000}))
with (ROOT/'logs/baseline_train.log').open('w') as log:
    code=subprocess.run(command,cwd=BASE/'repo',env=env,stdout=log,stderr=subprocess.STDOUT).returncode
if code:
    status.write_text(json.dumps({'stage':'TECHNICAL_PAUSE','returncode':code,'automatic_recipe_change':False}))
    raise RuntimeError('Baseline training failed; inspect log, no automatic batch/resolution change')
status.write_text(json.dumps({'stage':'BASELINE_TRAINING_COMPLETE_EVALUATION_PENDING','baseline_inference_completed':False,
                             'next_required':'Freeze validation-selected checkpoint, DDR/IDRiD masks/counts, MA analysis and efficiency; no shutdown'}))
