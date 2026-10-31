"""Read live atomic status separately from slower immutable evidence mirroring."""
import json
from pathlib import Path
import shlex
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from qlib.common import read_json,utc_now,write_json
from ops.remote import connect,command


def main():
    runtime=read_json(ROOT/'configs/runtime.local.json');remote=runtime['server']['remote_root']
    code='''
from pathlib import Path
import json,shutil
r=Path(%r)
def read(p):return json.loads(p.read_text()) if p.exists() else None
statuses={p.parent.name:read(p) for p in (r/'runs').glob('*/status.json')}
print(json.dumps({'driver':read(r/'reports/driver_state.json'),'campaign':read(r/'reports/campaign.json'),'runs':statuses,
 'last_checkpoint_completed':{p.parent.name: max([int(x.stem.split('_u')[-1]) for x in p.parent.glob('valid_audit_u*.json')],default=0) for p in (r/'runs').glob('*/status.json')},
 'free_disk_bytes':shutil.disk_usage(r).free}))
'''%remote
    with connect() as client:
        state=json.loads(command(client,shlex.quote(runtime['server_python'])+" - <<'PY'\n"+code+"\nPY",timeout=30))
    state['observed_at']=utc_now()
    write_json(ROOT/'reports/live_status.json',state)
    print(json.dumps(state,ensure_ascii=False))
    return state


if __name__=='__main__':main()
