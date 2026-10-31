"""Read this newly launched arm's smoke/progress and last diagnostic log lines once."""
import json
from pathlib import Path
import shlex
import sys
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(ROOT/'local_preparation'))
from remote import connect
client=connect()
try:
    code='''import json,time,subprocess
from pathlib import Path
root=Path('/root/autodl-tmp/HRDecoder_v3_20261008')
reports=root/'reports/sacd_micro'
out={'time_unix':time.time(),'server_online':True}
for name in ('STAGING_STATUS.json','CONTROLLER_ERROR.json','MICRO_GATE.json'):
 path=reports/name
 if path.exists():out[name]=json.loads(path.read_text())
for arm in ('control','feature_isolated'):
 for name in ('GPU_SMOKE_PASS.json','progress.json','PROBE_COMPLETE.json'):
  path=root/'training_runs/sacd_micro'/arm/name
  if path.exists():out[arm+'/'+name]=json.loads(path.read_text())
 path=reports/(arm+'.log')
 if path.exists():
  with path.open('rb') as log:
   log.seek(max(0,path.stat().st_size-5000));out[arm+'/log_tail']=log.read().decode(errors='replace')
out['GPU_status']=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,power.draw','--format=csv'],universal_newlines=True)
print(json.dumps(out))
'''
    _,out,err=client.exec_command('/root/miniconda3/bin/python -c '+shlex.quote(code))
    raw,error=out.read().decode(),err.read().decode()
    if out.channel.recv_exit_status():raise RuntimeError(error)
    value=json.loads(raw)
    (HERE/'PROBE_LAUNCH_VERIFIED.json').write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(value))
finally:client.close()
