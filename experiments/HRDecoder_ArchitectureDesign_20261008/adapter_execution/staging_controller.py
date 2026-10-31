"""CPU prepare concurrently; start one registered adapter after owned probe ends."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
REPORTS=ROOT/'reports/adapter'
PARENT_PID=40900

def write(name,value):
 p=REPORTS/name;p.parent.mkdir(parents=True,exist_ok=True)
 tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(p)

def main():
 REPORTS.mkdir(parents=True,exist_ok=True)
 assert not (REPORTS/'STAGING_STARTED.json').exists(),'No repeated staging'
 write('STAGING_STARTED.json',{'pid':os.getpid(),'time_unix':time.time(),'after_owned_diagnostic_pid':PARENT_PID,
  'candidate':'direct multiscale adapter','fresh_checkpoint_used':False,'server_kept_online':True})
 env=dict(os.environ,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='4',PYTHONPATH=str(ROOT/'vendor/champion_repo'))
 with (REPORTS/'cpu.log').open('a') as log:
  subprocess.run([sys.executable,'verify_adapter.py'],cwd=str(HERE),stdout=log,stderr=log,env=env,check=True,timeout=120)
 gate=json.loads((HERE/'CPU_ADAPTER_PASS.json').read_text());assert gate['status']=='PASS' and gate['torch']=='2.0.1+cu117'
 write('STAGING_STATUS.json',{'status':'CPU_PASS_WAITING_OWN_PROBE','pid':os.getpid(),'time_unix':time.time()})
 started=time.time()
 while True:
  path=Path('/proc/%d/cmdline'%PARENT_PID)
  alive=path.exists() and b'research_v10/run_paired.py' in path.read_bytes()
  complete=ROOT/'training_runs/diagnostic_v10/PROBE_COMPLETE.json'
  failure=ROOT/'reports/diagnostic_v10/ERROR.json'
  if complete.exists():
   value=json.loads(complete.read_text())
   assert value['status']=='COMPLETE' and value['updates_per_arm']==500
   assert value['provider_state_and_full_val_counts_identical'] and value['all_real_training_inputs_shared_unchanged']
   if alive:
    # Saved terminal evidence exists. Stop only the completed probe session to
    # release persistent workers/CUDA if Python is still in loader finalization.
    assert os.getsid(PARENT_PID)==PARENT_PID
    os.killpg(PARENT_PID,signal.SIGTERM)
    time.sleep(2)
   break
  if failure.exists() and not alive:
   # This candidate was chosen before these loss-probe results. A failed probe
   # is never used as evidence to select its KL; it keeps original all-valid KL.
   write('PREVIOUS_PROBE_FAILURE.json',json.loads(failure.read_text()));break
  assert time.time()-started<3600,'Owned diagnostic staging timeout'
  time.sleep(2)
 for _ in range(15):
  gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],universal_newlines=True).strip()
  if not gpu:break
  time.sleep(2)
 assert not gpu,'GPU still occupied; do not interfere'
 write('STAGING_STATUS.json',{'status':'REGISTERED_ADAPTER_GPU_GATE_AND_FRESH_SCREEN_STARTING','pid':os.getpid(),'time_unix':time.time(),
  'warm_weights_used':False,'loss_selected_before_probe_results':'original all-valid normalization'})
 with (REPORTS/'controller.log').open('a') as log:
  p=subprocess.Popen([sys.executable,'-u','controller.py'],cwd=str(HERE),stdout=log,stderr=log,env=env)
 write('CONTROLLER_DISPATCHED.json',{'pid':p.pid,'staging_pid':os.getpid(),'time_unix':time.time(),'server_kept_online':True})
 code=p.wait()
 write('STAGING_COMPLETE.json',{'exit_code':code,'time_unix':time.time(),'server_kept_online':True})
 assert code==0,'Adapter controller failed; saved error needs immediate autonomous repair'

if __name__=='__main__':
 try:main()
 except BaseException as e:
  write('STAGING_ERROR.json',{'error':str(e),'traceback':traceback.format_exc(),'time_unix':time.time(),
       'server_kept_online':True,'external_diagnostics_controlled':False})
  raise
