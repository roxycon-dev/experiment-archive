"""Current atomic experiment states and relevant data-disk reserve only."""
import sys,json,shlex
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from qlib.common import read_json,write_json,utc_now
from ops.remote import connect,command

def main():
    runtime=read_json(ROOT/'configs/runtime.local.json')
    if not runtime['server'].get('assignment_confirmed'):
        state={'stage':'LOCAL_PREPARATION_NO_SERVER_ASSIGNED','preparation':read_json(ROOT/'reports/preparation_status.json')}
    else:
        code="""
from pathlib import Path
import json,shutil
r=Path(%r)
def read(p):return json.loads(p.read_text(encoding='utf-8')) if p.exists() else None
print(json.dumps({'driver':read(r/'reports/driver_state.json'),'campaign':read(r/'reports/campaign.json'),
 'early_gpu_gates':read(r/'reports/early_gpu_gates.json'),'early_gpu_pid':read(r/'reports/early_gpu_gate_pid.json'),
 'gpu_readiness':read(r/'reports/gpu_readiness.json'),'runs':{p.parent.name:read(p) for p in (r/'runs').glob('*/status.json')},
 'free_disk_bytes':shutil.disk_usage(r if r.exists() else r.parent).free}))
"""%runtime['server']['remote_root']
        with connect() as client:state=json.loads(command(client,shlex.quote(runtime['legacy_python'])+" - <<'PY'\n"+code+"\nPY",30))
    preparation=ROOT/'reports/preparation_driver.json'
    if preparation.exists():state['local_preparation_driver']=read_json(preparation)
    state['observed_at']=utc_now();write_json(ROOT/'reports/live_status.json',state);print(json.dumps(state,ensure_ascii=False));return state

if __name__=='__main__':main()
