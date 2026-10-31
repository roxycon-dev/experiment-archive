"""Handoff only our dispatcher; inference children keep their PIDs and state."""
from pathlib import Path
import hashlib
import json
import os
import signal
import subprocess
import time

TASK=Path('/home/studentuser001/paper_supplement_20261023')
ROOT=TASK/'evaluation_workspace/paper_supplement_20261023'
pidfile=TASK/'logs/paired_evaluation59_supervisor_pid.json'
old=json.loads(pidfile.read_text())['pid']
oldproc=Path('/proc')/str(old)
assert b'queue_paired_evaluation59.py' in (oldproc/'cmdline').read_bytes(), 'Only the previous task dispatcher is replaceable'
assert (ROOT/'scripts/queue_supplement_evaluation59.py').exists() and (ROOT/'scripts/evaluate_retrained_baseline59.py').exists()
stamp=time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())
os.kill(old,signal.SIGSTOP)
try:
    previous=json.loads((TASK/'logs/paired_evaluation59_queue.json').read_text())
    adopted=dict(previous.get('running',{}))
    children=(oldproc/'task'/str(old)/'children').read_text().split()
    for child in children:
        proc=Path('/proc')/child
        if not proc.exists():continue
        cmd=(proc/'cmdline').read_bytes().split(b'\0')
        if not any(b'evaluate_paired_seed59.py' in arg for arg in cmd):continue
        index=cmd.index(b'--job');name=Path(os.fsdecode(cmd[index+1])).stem
        environment=dict(x.split(b'=',1) for x in (proc/'environ').read_bytes().split(b'\0') if b'=' in x)
        gpu=int(environment[b'CUDA_VISIBLE_DEVICES'])
        adopted[name]={'pid':int(child),'GPU':gpu}
    previous['running']=adopted
    snapshot=TASK/'logs'/('evaluation_dispatcher_adopt_'+stamp+'.json');snapshot.write_text(json.dumps(previous,indent=2))
    receipt={'old_dispatcher_pid':old,'adopted_workers':adopted,'worker_processes_stopped':False,
             'reason':'Add the validation-selected baseline to the same idle-GPU dispatcher; keep existing fixed-seed inference intact',
             'original_dispatcher_sha256':hashlib.sha256((ROOT/'scripts/queue_paired_evaluation59.py').read_bytes()).hexdigest(),'adopt_state':str(snapshot)}
    os.kill(old,signal.SIGTERM)
finally:
    try:os.kill(old,signal.SIGCONT)
    except ProcessLookupError:pass
for _ in range(50):
    if not oldproc.exists() or (oldproc/'stat').read_text().split()[2]=='Z':break
    time.sleep(.1)
else:raise RuntimeError('Old dispatcher did not exit; do not start a duplicate')
handle=(TASK/'logs'/('supplement_evaluation59_dispatcher_'+stamp+'.log')).open('ab',buffering=0)
p=subprocess.Popen(['/usr/bin/python3','-u',str(ROOT/'scripts/queue_supplement_evaluation59.py'),'--adopt-state',str(snapshot)],
                   stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
receipt['new_dispatcher_pid']=p.pid;receipt['new_controller']='queue_supplement_evaluation59.py'
pidfile.write_text(json.dumps({'pid':p.pid,'controller':receipt['new_controller']}))
(ROOT/'evidence/evaluation59_dispatcher_handoff.json').write_text(json.dumps(receipt,indent=2))
print(json.dumps(receipt),flush=True)
