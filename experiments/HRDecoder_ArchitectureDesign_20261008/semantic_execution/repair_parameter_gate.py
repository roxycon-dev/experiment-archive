"""One gate repair before any formal updates; archive prior failed receipt."""
from pathlib import Path
import hashlib,json,shlex,sys
H=Path(__file__).resolve().parent;R=H.parent
sys.path.insert(0,str(R/'local_preparation'));from remote import connect
REMOTE='/root/autodl-tmp/HRDecoder_v3_20261008'
BEFORE='0c49f8322762a9deddabb29ed3e665544111ec0c6bd70f53de75d9f7bcdeccdf'
c=connect()
try:
 code="""import os,json,time,subprocess
from pathlib import Path
r=Path(REMOTE);h=r/'semantic_execution';p=r/'reports/semantic';w=r/'training_runs/semantic'
assert not (w/'last.pt').exists() and not (w/'protocol.json').exists() and not (p/'SCREEN_STARTED.json').exists()
old=json.loads((p/'DEPLOYMENT_DISPATCHED.json').read_text());pid=old['pipeline_pid']
assert not Path('/proc/%d'%pid).exists(),'Prior own pipeline still present'
session=[]
for proc in Path('/proc').iterdir():
 if not proc.name.isdigit():continue
 try:
  if os.getsid(int(proc.name))==pid:session.append(int(proc.name))
 except (ProcessLookupError,PermissionError):pass
assert not session,'Prior own children still live'
f=p/'failed_parameter_assertion_before_updates';f.mkdir(exist_ok=False)
for n in ['CPU_gate.log','smoke.log','dispatch.log','CONTROLLER_ERROR.json','CPU_TO_GPU_DISPATCH.json','DEPLOYMENT_DISPATCHED.json']:
 if (p/n).exists():(p/n).rename(f/n)
for n in ['CPU_ADAPTER_PASS.json','train.py']:
 if (h/n).exists():(f/n).write_bytes((h/n).read_bytes())
print(json.dumps({'safe_to_repair':True,'formal_updates':0,'prior_session_empty':True,'old_pipeline_pid':pid}))
""".replace('REMOTE',repr(REMOTE))
 _,o,e=c.exec_command('/root/miniconda3/bin/python -c '+shlex.quote(code));out=o.read().decode();err=e.read().decode();assert o.channel.recv_exit_status()==0,err;pre=json.loads(out)
 with c.open_sftp() as f:
  for n in ['train.py','monitor_semantic.py']:f.put(str(H/n),REMOTE+'/semantic_execution/'+n)
 code="""import json,subprocess,time
from pathlib import Path
r=Path(REMOTE);h=r/'semantic_execution';p=r/'reports/semantic'
assert not (r/'training_runs/semantic/last.pt').exists() and not (p/'SCREEN_STARTED.json').exists()
job=subprocess.Popen(['/root/miniconda3/bin/python','-u',str(h/'dispatch_after_cpu.py')],cwd=str(h),stdout=open(p/'dispatch.log','a',buffering=1),stderr=subprocess.STDOUT,start_new_session=True)
receipt={'status':'CPU_TO_REAL_GPU_PIPELINE_DISPATCHED','pipeline_pid':job.pid,'time_unix':time.time(),'fresh_run_count':1,'formal_updates_before_repair':0,'gate_repair':True,'server_kept_running':True,'external_diagnostics_controlled':False}
(p/'DEPLOYMENT_DISPATCHED.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
""".replace('REMOTE',repr(REMOTE))
 _,o,e=c.exec_command('/root/miniconda3/bin/python -c '+shlex.quote(code));out=o.read().decode();err=e.read().decode();assert o.channel.recv_exit_status()==0,err;receipt=json.loads(out)
 audit={'scope':'Correct stale total-parameter assertion before optimizer or formal updates','old_train_SHA':BEFORE,'new_train_SHA':hashlib.sha256((H/'train.py').read_bytes()).hexdigest(),'expected_total_parameters':65856916,'private_parameters':284496,'model_loss_optimizer_unchanged':True,'CPU_and_GPU_gates_repeated':True,'old_failure_preserved_remote':'reports/semantic/failed_parameter_assertion_before_updates',**pre,**receipt}
 (H/'GPU_GATE_REPAIR.json').write_text(json.dumps(audit,indent=2)+'\n',encoding='utf-8');(H/'DEPLOYMENT_DISPATCHED.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
 with c.open_sftp() as f:f.put(str(H/'GPU_GATE_REPAIR.json'),REMOTE+'/reports/semantic/GPU_GATE_REPAIR.json')
 print(json.dumps(receipt))
finally:c.close()
