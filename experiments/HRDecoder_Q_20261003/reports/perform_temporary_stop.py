"""One-shot platform shutdown worker, run only on the assigned instance after backup."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from datetime import datetime,timezone

ROOT=Path('/root/autodl-tmp/HRDecoder_Q_20261003')
REQUEST=ROOT/'reports/temporary_shutdown_request.json'

def now():return datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(p.read_text())
def save(x):REQUEST.write_text(json.dumps(x,indent=2)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

request=read(REQUEST)
assert request['stage']=='SCHEDULED' and request['original_plan_complete'] is False
time.sleep(5)
override=read(ROOT/'reports/execution_scope_override.json')
handoff=read(ROOT/'reports/temporary_stop_handoff.json')
assert override['temporary_stop_authorized_by_latest_no_idle_instruction']
assert handoff['local_results_sha_verified'] and handoff['temporary_stop_for_no_idle']
assert read(ROOT/'reports/final_evaluation_state.json')['stage']=='COMPLETE'
assert sha(ROOT/'reports/final_selection.json')==handoff['final_selection_sha256']
for seed in (42,2024):
    assert read(ROOT/f'runs/Q_DDR_seed{seed}/status.json')['stage']=='COMPLETE'
for name,tag in [('formal.pid',b'train_run.py'),('final_test.pid',b'run_locked_test.py')]:
    p=ROOT/'reports'/name
    if p.exists():
        pid=int(p.read_text())
        process=Path(f'/proc/{pid}/cmdline')
        assert not process.exists() or tag not in process.read_bytes(),'An authorized task is still running'
control=Path('/usr/bin/shutdown')
assert sha(control)=='0358e83eeeaf542aa98f64ba9e339c91df46f1e025892d52dba159f4fb1cf027'
request.update(stage='EXECUTING_OFFICIAL_SHUTDOWN',executing_at=now())
save(request)
result=subprocess.run(['/bin/bash',str(control)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=60)
private=ROOT/'assets/temporary_shutdown_output.private'
private.write_bytes(result.stdout+b'\n'+result.stderr)
private.chmod(0o600)
request.update(stage='OFFICIAL_SHUTDOWN_RETURNED_SUCCESS' if result.returncode==0 else 'SHUTDOWN_COMMAND_FAILED',
    return_code=result.returncode,command_completed_at=now(),output_sha256=sha(private),
    platform_OFF_verified=False)
save(request)
