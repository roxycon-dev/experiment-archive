set -eu
BASE=/root/autodl-tmp/nativepyramid_sp_v3_20261016
"$BASE/venv/bin/python" - <<'PY'
import json,os,signal,time
from pathlib import Path
b=Path('/root/autodl-tmp/nativepyramid_sp_v3_20261016');pid=1752
p=json.loads((b/'run_seed42/progress.json').read_text())
assert p['executed_updates']==0 and p['in_flight_update'] is None and not (b/'run_seed42/train.jsonl').exists()
command=Path(f'/proc/{pid}/cmdline').read_bytes()
assert b'train.py' in command and str(b/'run_seed42').encode() in command and os.getpgid(pid)==pid
r={'reason':'SIGINT could not unwind stalled initial worker creation',
   'executed_updates':0,'in_flight_update':None,'additional_executed_updates_uncertain':[0,0],
   'evidence':'progress persisted before first attempted update; no execution ledger; no optimizer entry',
   'signal':'SIGTERM','owned_process_group':pid,'server_shutdown_requested':False}
(b/'receipts/zero_step_loader_forced_stop.json').write_text(json.dumps(r,indent=2)+'\n')
os.killpg(pid,signal.SIGTERM)
print(json.dumps(r),flush=True)
PY
sleep 6
cat "$BASE/job_status.json"
ps -eo pid,ppid,comm,args | grep -E '1752|pt_data_worker|controller.py' | grep -v grep || true
