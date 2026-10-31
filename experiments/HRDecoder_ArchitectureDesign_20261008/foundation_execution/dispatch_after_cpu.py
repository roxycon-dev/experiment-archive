import json,os,subprocess,time,traceback
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parent;P=R/'reports/foundation';P.mkdir(parents=True,exist_ok=True)
env=dict(os.environ,PYTHONPATH=str(R/'vendor/champion_repo'),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='4')
try:
 assert not (R/'training_runs/foundation/last.pt').exists()
 ready=json.loads((R/'reports/sacd/RAM_CACHE_READY.json').read_text());assert ready['pid']==39434 and b'sacd_execution/ram_cache_owner.py' in Path('/proc/39434/cmdline').read_bytes()
 with open(P/'CPU_gate.log','a',buffering=1) as log:
  code=subprocess.call(['/root/miniconda3/bin/python','-u',str(H/'verify_adapter.py')],cwd=str(H),env=env,stdout=log,stderr=log)
 assert code==0,'Pinned CPU candidate invariants failed'
 v=json.loads((H/'CPU_ADAPTER_PASS.json').read_text());assert v['status']=='PASS' and v['torch']=='2.0.1+cu117'
 (P/'CPU_TO_GPU_DISPATCH.json').write_text(json.dumps({'status':'CPU_PASS_STARTING_REAL_GPU_GATE','controller_pid':os.getpid(),'time_unix':time.time(),'fresh_only':True},indent=2))
 os.execve('/root/miniconda3/bin/python',['/root/miniconda3/bin/python','-u',str(H/'controller.py')],env)
except BaseException as error:
 (P/'LAUNCH_ERROR.json').write_text(json.dumps({'error':str(error),'traceback':traceback.format_exc(),'time_unix':time.time(),'server_kept_running':True,'external_diagnostics_controlled':False},indent=2));raise
