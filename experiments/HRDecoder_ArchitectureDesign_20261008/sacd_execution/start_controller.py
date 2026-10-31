"""Launch only this SACD controller, with PID and cmdline duplicate protection."""
import json
import os
from pathlib import Path
import subprocess
import time
from datetime import datetime, timezone, timedelta

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
REPORTS=ROOT/'reports/sacd'
SCRIPT=str(HERE/'controller.py')
for p in Path('/proc').iterdir():
    if p.name.isdigit() and int(p.name)!=os.getpid():
        try:
            if SCRIPT.encode() in (p/'cmdline').read_bytes():
                raise RuntimeError('Existing SACD controller PID '+p.name+'; do not launch duplicate')
        except OSError:pass
assert not (ROOT/'training_runs/sacd/last.pt').exists(),'Existing checkpoint requires explicit resume, not fresh launch'
cpu=json.loads((HERE/'CPU_SACD_PASS.json').read_text())
resume=json.loads((HERE/'CHECKPOINT_RESUME_PASS.json').read_text())
smoke=json.loads((ROOT/'training_runs/sacd/GPU_SMOKE_PASS.json').read_text())
assert cpu['status']==resume['status']==smoke['status']=='PASS'
assert smoke['discarded_optimizer_steps']==12 and smoke['formal_updates']==0
deadline=datetime(2026,10,10,2,0,tzinfo=timezone(timedelta(hours=8))).timestamp()
seconds=smoke['mean_warm_step_seconds']
# CET measured full-val ~32s each, checkpoints/initialization ~2min,
# locked227-test <=3min: ~11min work plus ~9min safety, 20min total.
estimated=time.time()+40000*seconds+1200
gate={'status':'PASS' if estimated<deadline else 'INSUFFICIENT_TIME',
      'deadline_unix':deadline,'measured_warm_step_seconds':seconds,
      'projected_finish_with_20min_reserve_unix':estimated,'checked_unix':time.time(),
      'reserve_seconds':1200,'reserve_basis':'10x32s full-val + ~120s checkpoint/init + 180s test + 580s safety',
      'additional_full_runs_allowed':1,'new_runs_after_this':0}
(REPORTS/'DEADLINE_GATE.json').write_text(json.dumps(gate,indent=2)+'\n')
assert gate['status']=='PASS','Full 40k does not fit measured deadline budget'
# An exclusive receipt makes a second fresh full run impossible through this entry point.
registry=REPORTS/'FINAL_ROUND_STARTED.json'
with registry.open('x') as receipt:
    json.dump({'status':'START_REQUESTED','time_unix':time.time(),
               'identity_SHA':smoke['identity_SHA'],'full_runs':1,'limit':1},receipt,indent=2)
log=open(REPORTS/'controller.log','ab',buffering=0)
child=subprocess.Popen(['/root/miniconda3/bin/python','-u',SCRIPT],cwd=str(HERE),
                       stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
time.sleep(2)
if child.poll() is not None:
    raise RuntimeError('Controller exited: '+(REPORTS/'controller.log').read_text()[-3000:])
print(json.dumps({'status':'CONTROLLER_LAUNCHED','pid':child.pid,'server_shutdown':False}))
