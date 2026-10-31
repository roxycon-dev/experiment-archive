"""One authorized final shutdown on the assigned instance, with no automatic retry."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,subprocess,time

ROOT=Path('/root/autodl-tmp/HRDecoder_Q_20261003')
REQUEST=ROOT/'reports/final_shutdown_request.json'
def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def now(): return datetime.now(timezone.utc).isoformat()
def save(x): REQUEST.write_text(json.dumps(x,indent=2)+'\n')

request=read(REQUEST)
assert request['stage']=='SCHEDULED_AFTER_COMPLETE_SCIENTIFIC_DELIVERY'
time.sleep(5)
delivery=read(ROOT/'reports/original_plan_delivery_manifest.json')
scope=read(ROOT/'reports/original_plan_completion.json')
assert delivery['stage']=='COMPLETE_SCIENTIFIC_DELIVERY_LOCAL_SHA_AND_CLEANUP'
assert delivery['shutdown_authorized'] and scope['shutdown_allowed'] and scope['delivery_complete']
assert request['delivery_manifest_sha256']==sha(ROOT/'reports/original_plan_delivery_manifest.json')
assert sha(ROOT/'reports/continuation_final_selection.json')==delivery['selected_lock_sha256']
assert read(ROOT/'reports/continuation_final_evaluation_state.json')['stage']=='COMPLETE'
for run in ('Q_DDR_seed42','Q_DDR_seed2024','Q_MIX_seed42','Q_MIX_seed2024'):
    s=read(ROOT/'runs'/run/'status.json')
    assert s['stage']=='COMPLETE' and s['completed']==24000
for proc in Path('/proc').iterdir():
    if not proc.name.isdigit(): continue
    try: command=(proc/'cmdline').read_bytes()
    except (FileNotFoundError,PermissionError,ProcessLookupError): continue
    assert not (b'train_run.py' in command or b'run_continuation_locked_test.py' in command), 'A formal task is still running'
control=Path('/usr/bin/shutdown')
assert sha(control)=='0358e83eeeaf542aa98f64ba9e339c91df46f1e025892d52dba159f4fb1cf027'
request.update(stage='EXECUTING_OFFICIAL_FINAL_SHUTDOWN',executing_at=now())
save(request)
result=subprocess.run(['/bin/bash',str(control)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=60)
private=ROOT/'assets/final_shutdown_output.private'
private.write_bytes(result.stdout+b'\n'+result.stderr);private.chmod(0o600)
request.update(stage='OFFICIAL_SHUTDOWN_COMMAND_RETURNED_SUCCESS' if result.returncode==0 else 'SHUTDOWN_COMMAND_FAILED',
    return_code=result.returncode,command_completed_at=now(),output_sha256=sha(private),platform_OFF_verified=False)
save(request)
