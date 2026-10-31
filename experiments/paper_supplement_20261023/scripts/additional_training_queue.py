"""Finite serial C/S queue; technical failures stop, completed jobs never repeat."""
import json
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1];TRAIN=ROOT/'training_v1'
PY='/root/miniconda3/envs/hrdecoder_q_20261003/bin/python'
status=ROOT/'training_queue_progress.json'
def write(obj):status.write_text(json.dumps(obj,indent=2))
def read(p):return json.loads(p.read_text())
# The first C100 process was already launched after data acceptance.
first=TRAIN/'runs/C_seed2024'
while not (first/'benchmark.json').exists():
    s=read(first/'status.json') if (first/'status.json').exists() else {}
    if s.get('stage')=='TECHNICAL_PAUSE':raise RuntimeError(s)
    write({'stage':'WAIT_FIRST_C100','first_C':s})
    time.sleep(10)
tasks=[('S',2024,100,False),('C',2024,20000,True),('S',2024,20000,True),
       ('C',3407,20000,False),('S',3407,20000,False)]
for arm,seed,stop,resume in tasks:
    run=TRAIN/f'runs/{arm}_seed{seed}'
    if stop==20000 and (run/'completion.json').exists():continue
    if stop==100 and (run/'benchmark.json').exists():continue
    command=[PY,'-u',str(TRAIN/'new_seed_train.py'),'--arm',arm,'--seed',str(seed),'--stop-at',str(stop)]
    if resume:command.append('--resume')
    write({'stage':'RUNNING','arm':arm,'seed':seed,'stop_at':stop,'new_training_runs_planned':4})
    with (ROOT/f'log_{arm}{seed}_to{stop}.txt').open('w') as log:
        p=subprocess.Popen(command,cwd=str(TRAIN),stdout=log,stderr=subprocess.STDOUT)
        code=p.wait()
    if code:
        write({'stage':'TECHNICAL_PAUSE','arm':arm,'seed':seed,'returncode':code})
        raise RuntimeError(f'{arm}{seed} failed; no automatic hyperparameter change or restart')
    if (TRAIN/'runs/S_seed2024/benchmark.json').exists():
        c=read(first/'benchmark.json');s=read(TRAIN/'runs/S_seed2024/benchmark.json')
        remaining=0
        for a in ['C','S']:
            for sd in [2024,3407]:
                rr=TRAIN/f'runs/{a}_seed{sd}'
                progress=read(rr/'status.json') if (rr/'status.json').exists() else {}
                remaining+=max(0,20000-progress.get('completed',0))*(c if a=='C' else s)['mean_seconds']
        (ROOT/'training_ETA.json').write_text(json.dumps({'C_seconds_per_update':c['mean_seconds'],
            'S_seconds_per_update':s['mean_seconds'],'remaining_adaptation_training_hours':remaining/3600,
            'excludes':'baseline training, initialization, future validation/inference, checkpoint overhead'},indent=2))
write({'stage':'FOUR_ADAPTATION_TRAININGS_COMPLETE','runs':4,'baseline_status':'separately scheduled',
       'next_required':'20k paired evaluation and result recovery; do not mark the whole campaign complete'})
