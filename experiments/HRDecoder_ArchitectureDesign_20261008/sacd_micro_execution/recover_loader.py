"""Repair only the failed diagnostic loader setup; preserve the immutable parent and attempt evidence."""
import json
from pathlib import Path
import shlex
import sys
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(ROOT/'local_preparation'))
from remote import connect
REMOTE='/root/autodl-tmp/HRDecoder_v3_20261008'
client=connect()
try:
    code='''import json,os,signal,time
from pathlib import Path
root=Path('/root/autodl-tmp/HRDecoder_v3_20261008')
reports=root/'reports/sacd_micro';work=root/'training_runs/sacd_micro'
status=json.loads((reports/'STAGING_STATUS.json').read_text())
assert status['status']=='PAIRED_PROBE_RUNNING' and status['arm']=='control'
assert status['controller_pid']==30376 and status['pid']==30734
arm=work/'control'
assert not (arm/'probe_last.pt').exists() and not (arm/'PROBE_COMPLETE.json').exists()
assert not (reports/'MICRO_GATE.json').exists()
controller_cmd=Path('/proc/30376/cmdline').read_bytes()
probe_cmd=Path('/proc/30734/cmdline').read_bytes()
assert b'sacd_micro_execution/controller.py' in controller_cmd and b'probe.py' in probe_cmd
assert Path('/proc/30734/cwd').resolve()==root/'sacd_micro_execution'
children=[int(x) for x in Path('/proc/30734/task/30734/children').read_text().split()]
observations=[]
for pid in children:
 p=Path('/proc/%d'%pid)
 assert (p/'stat').read_text().split(') ',1)[1].split()[1]=='30734'
 assert Path('/proc/%d/cwd'%pid).resolve()==root/'sacd_micro_execution'
 observations.append({'pid':pid,'cmdline':(p/'cmdline').read_bytes().replace(b'\\0',b' ').decode(),'wchan':(p/'wchan').read_text()})
receipt={'status':'LOADER_STARTUP_REPAIR','time_unix':time.time(),'original_parent_preserved':True,
 'reason':'Second worker fork after CUDA/validation; GPU waiting without first25-step progress receipt',
 'failed_attempt_progress_receipt_exists':(arm/'progress.json').exists(),
 'failed_attempt_formal_checkpoint_exists':False,'discarded_steps_known':3,
 'formal_updates_unreported_upper_bound':24,'children':observations,
 'server_kept_running':True,'external_diagnostics_controlled':False}
assert not receipt['failed_attempt_progress_receipt_exists'],'Use explicit saved progress rather than restart'
(reports/'LOADER_STARTUP_REPAIR.json').write_text(json.dumps(receipt,indent=2)+'\\n')
os.kill(30376,signal.SIGTERM)
time.sleep(.3)
os.kill(30734,signal.SIGTERM)
for pid in children:
 try:os.kill(pid,signal.SIGTERM)
 except ProcessLookupError:pass
time.sleep(1)
for pid in [30376,30734]+children:
 p=Path('/proc/%d/stat'%pid)
 if p.exists() and p.read_text().split(') ',1)[1].split()[0]!='Z':os.kill(pid,signal.SIGKILL)
attempt=work/'failed_loader_attempt_control'
assert not attempt.exists()
arm.rename(attempt)
oldlog=reports/'control.log'
oldlog.rename(reports/'failed_loader_attempt_control.log')
(reports/'MICRO_STARTED.json').rename(reports/'MICRO_STARTED_first_attempt.json')
print(json.dumps(receipt))
'''
    _,out,err=client.exec_command('/root/miniconda3/bin/python -c '+shlex.quote(code))
    raw,error=out.read().decode(),err.read().decode()
    if out.channel.recv_exit_status():raise RuntimeError(error)
    receipt=json.loads(raw)
    (HERE/'LOADER_STARTUP_REPAIR.json').write_text(json.dumps(receipt,indent=2)+'\n')
    with client.open_sftp() as ftp:
        ftp.put(str(HERE/'probe.py'),REMOTE+'/sacd_micro_execution/probe.py')
    print(json.dumps(receipt))
finally:client.close()
