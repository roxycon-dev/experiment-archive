set -eu
cd /root/autodl-tmp/HRDecoder_Q_20261003
/root/miniconda3/envs/hrdecoder_q_20261003/bin/python - <<'PY'
from pathlib import Path
import json,os
from qlib.common import read_json,write_json,utc_now
root=Path.cwd()
state=read_json('reports/final_evaluation_state.json')
alive=False
try:
    os.kill(state['pid'],0)
    alive=b'run_locked_test.py' in Path(f'/proc/{state["pid"]}/cmdline').read_bytes()
except (ProcessLookupError,FileNotFoundError):pass
r={'checked_at':utc_now(),'state':state,'process_alive':alive,'phases':{}}
for name in ('p0_manifest','seed42','seed2024','summary'):
    p=root/'reports/final_test'/(name+'.json')
    if p.exists():
        x=read_json(p)
        r['phases'][name]={'status':x.get('status'),'completed_images':len(x.get('images',{})),
            'lock_sha256':x.get('lock_sha256'),'selected':x.get('selected') if x.get('status')=='COMPLETE' else None}
log=root/'reports/final_evaluation.log'
with log.open('rb') as f:
    f.seek(max(0,log.stat().st_size-12000))
    r['recent_log']=f.read().decode(errors='replace').splitlines()[-12:]
write_json('reports/final_evaluation_snapshot.json',r)
print(json.dumps(r))
PY
