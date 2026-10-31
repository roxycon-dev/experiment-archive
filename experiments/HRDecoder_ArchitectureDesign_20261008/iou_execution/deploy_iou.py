"""Deploy sealed small sources and start one fresh CPU-to-GPU pipeline."""
import hashlib,json,shlex,sys,time
from pathlib import Path
H=Path(__file__).resolve().parent; R=H.parent
sys.path.insert(0,str(R/'local_preparation'))
from remote import connect
REMOTE='/root/autodl-tmp/HRDecoder_v3_20261008'
names=('baseline.py','model.py','private_modules.py','stable_objectives.py','fast_objectives.py','objectives.py','data.py','loader.py','train.py','verify_adapter.py','controller.py','dispatch_after_cpu.py','screen_gate.py','screen_protocol.json','test_eval.py','audit_final_outcome.py','monitor_iou.py','WORKER_RAM_PASS.json','architecture_spec.json','validation_reporting.json')
def command(c,s):
 _,o,e=c.exec_command(s); out=o.read().decode();err=e.read().decode();code=o.channel.recv_exit_status()
 if code:raise RuntimeError('Remote command failed: '+out+err)
 return out
def main():
 c=connect()
 try:
  pre="from pathlib import Path; r=Path(%r); assert not (r/'training_runs/iou/last.pt').exists(); assert not (r/'reports/iou/SCREEN_STARTED.json').exists(); [(r/x).mkdir(parents=True,exist_ok=True) for x in ('iou_execution','reports/iou','training_runs/iou')]"%REMOTE
  command(c,'/root/miniconda3/bin/python -c '+shlex.quote(pre))
  with c.open_sftp() as f:
   for n in names:f.put(str(H/n),REMOTE+'/iou_execution/'+n)
   refs={}
   for n,k in (('sacd_micro_execution/gate.py','validation_helper_SHA'),('sacd_execution/val_label_reference.json','val_reference_SHA')):
    expected=hashlib.sha256((R/n).read_bytes()).hexdigest()
    with f.open(REMOTE+'/'+n,'rb') as src:actual=hashlib.sha256(src.read()).hexdigest()
    assert expected==actual, 'Small validator/reference source mismatch'
    refs[k]=actual
   seal=dict(refs,remote_sources_match=True,locked_unix=time.time(),data_cache_hash_rescan=False)
   with f.open(REMOTE+'/reports/iou/VALIDATOR_LOCK.json','w') as dest:dest.write(json.dumps(seal,indent=2))
   (H/'VALIDATOR_LOCK.json').write_text(json.dumps(seal,indent=2)+'\n',encoding='utf-8')
  launch="""import json,os,subprocess,time
from pathlib import Path
r=Path(%r);h=r/'iou_execution';p=r/'reports/iou'
assert not (p/'DEPLOYMENT_DISPATCHED.json').exists(),'Already dispatched; do not duplicate fresh'
log=open(p/'dispatch.log','a',buffering=1)
job=subprocess.Popen(['/root/miniconda3/bin/python','-u',str(h/'dispatch_after_cpu.py')],cwd=str(h),stdout=log,stderr=log,start_new_session=True)
receipt={'status':'CPU_TO_REAL_GPU_PIPELINE_DISPATCHED','pipeline_pid':job.pid,'time_unix':time.time(),'fresh_run_count':1,'server_kept_running':True,'external_diagnostics_controlled':False}
(p/'DEPLOYMENT_DISPATCHED.json').write_text(json.dumps(receipt,indent=2))
print(json.dumps(receipt))
"""%REMOTE
  result=json.loads(command(c,'/root/miniconda3/bin/python -c '+shlex.quote(launch)))
  (H/'DEPLOYMENT_DISPATCHED.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
  print(json.dumps(result))
 finally:c.close()
if __name__=='__main__':main()
