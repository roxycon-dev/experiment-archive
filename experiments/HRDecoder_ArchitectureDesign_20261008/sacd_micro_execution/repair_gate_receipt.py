"""Complete an existing diagnostic's text-only audit; do not repeat any optimizer updates."""
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
    with client.open_sftp() as ftp:
        ftp.put(str(ROOT/'sacd_execution/val_label_reference.json'),REMOTE+'/sacd_execution/val_label_reference.json')
        ftp.put(str(HERE/'gate.py'),REMOTE+'/sacd_micro_execution/gate.py')
    code='''import json,time,sys,subprocess,hashlib
from pathlib import Path
root=Path('/root/autodl-tmp/HRDecoder_v3_20261008')
sys.path.insert(0,str(root/'sacd_micro_execution'))
from gate import evaluate_gate
from controller import write
reports=root/'reports/sacd_micro'
assert not (reports/'MICRO_GATE.json').exists(),'Already audited; do not repeat selection'
plan=json.loads((root/'sacd_micro_execution/micro_protocol.json').read_text())
out=evaluate_gate(root,plan)
out.update(time_unix=time.time(),server_kept_running=True,external_diagnostics_controlled=False,
           parent_checkpoint_preserved=True,loader_startup_recovery_recorded=True,
           reference_file_deployment_repair=True,no_optimizer_updates_during_gate_repair=True,
           gate_code_SHA256=hashlib.sha256((root/'sacd_micro_execution/gate.py').read_bytes()).hexdigest())
write(reports/'MICRO_GATE.json',out)
error=json.loads((reports/'CONTROLLER_ERROR.json').read_text())
assert 'val_label_reference.json' in error['error']
(reports/'CONTROLLER_ERROR.json').rename(reports/'CONTROLLER_ERROR_before_reference_repair.json')
write(reports/'GATE_RECEIPT_REPAIR.json',{'status':'COMPLETE','time_unix':time.time(),
 'reference_file_uploaded':True,'previous_error_preserved':True,'GPU_training_repeated':False,
 'criteria_relaxed':False,'micro_gate':out['status']})
write(reports/'STAGING_STATUS.json',{'status':'WARM_GATE_'+out['status'],'time_unix':time.time(),
 'fresh_screen_required':out['status']=='PASS','controller_alive':False,'server_kept_running':True})
if out['status']=='PASS':
 with open(reports/'fresh_transition.log','a',buffering=1) as log:
  p=subprocess.Popen(['/root/miniconda3/bin/python','-u',str(root/'sacd_isolated_execution/controller.py')],
                     cwd=str(root/'sacd_isolated_execution'),stdout=log,stderr=log,start_new_session=True)
 write(reports/'FRESH_SCREEN_DISPATCHED.json',{'controller_pid':p.pid,'time_unix':time.time(),'full40k_authorized_yet':False})
print(json.dumps({k:v for k,v in out.items() if k!='records'}))
'''
    _,out,err=client.exec_command('/root/miniconda3/bin/python -c '+shlex.quote(code))
    raw,error=out.read().decode(),err.read().decode()
    if out.channel.recv_exit_status():raise RuntimeError(error)
    value=json.loads(raw)
    (HERE/'GATE_RECEIPT_REPAIR.json').write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(value))
finally:client.close()
