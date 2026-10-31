set -eu
BASE=/root/autodl-tmp/nativepyramid_sp_v3_20261016
"$BASE/venv/bin/python" - <<'PY'
import json,os,signal,time
from pathlib import Path
b=Path('/root/autodl-tmp/nativepyramid_sp_v3_20261016')
p=json.loads((b/'run_seed42/progress.json').read_text())
assert p['executed_updates']==0 and p['in_flight_update'] is None
assert not (b/'run_seed42/train.jsonl').exists()
pid=1752
command=Path(f'/proc/{pid}/cmdline').read_bytes()
assert b'train.py' in command and str(b/'run_seed42').encode() in command
assert os.getpgid(pid)==pid
receipt={'reason':'initial_forked_input_workers_stalled_before_first_batch',
         'executed_updates':0,'in_flight_update':None,'train_ledger_exists':False,
         'pid':pid,'action':'SIGINT owned training process; preserve server and all outputs'}
(b/'receipts/zero_step_loader_hang.json').write_text(json.dumps(receipt,indent=2)+'\n')
os.kill(pid,signal.SIGINT)
print(json.dumps(receipt),flush=True)
PY
