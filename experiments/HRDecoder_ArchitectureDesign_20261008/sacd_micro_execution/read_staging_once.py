"""One launch-confirmation read; does not change the45min monitoring cadence."""
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
    code='''import json,time
from pathlib import Path
root=Path('/root/autodl-tmp/HRDecoder_v3_20261008')
reports=root/'reports/sacd_micro'
result={'time_unix':time.time(),'server_online':True}
for name in ('STAGING_STATUS.json','CONTROLLER_ERROR.json','MICRO_STARTED.json','MICRO_GATE.json'):
 path=reports/name
 if path.exists():result[name]=json.loads(path.read_text())
staged=json.loads((reports/'STAGING_DISPATCHED.json').read_text())
pid=staged['controller_pid']
result['staging_controller_cmdline']=Path('/proc/%d/cmdline'%pid).read_bytes().replace(b'\\0',b' ').decode()
result['parent_progress']=json.loads((root/'training_runs/sacd/progress.json').read_text())
print(json.dumps(result))
'''
    _,out,err=client.exec_command('/root/miniconda3/bin/python -c '+shlex.quote(code))
    raw,error=out.read().decode(),err.read().decode()
    if out.channel.recv_exit_status():raise RuntimeError(error)
    value=json.loads(raw)
    (HERE/'STAGING_LAUNCH_VERIFIED.json').write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(value))
finally:client.close()
