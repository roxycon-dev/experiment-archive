set -eu
BASE=/root/autodl-tmp/nativepyramid_sp_v3_20261016
"$BASE/venv/bin/python" - <<'PY'
import json,os,signal,time
from pathlib import Path
b=Path('/root/autodl-tmp/nativepyramid_sp_v3_20261016');pid=2160
assert not (b/'run_seed42_launch02').exists()
assert json.loads((b/'preflight_attempt03/preflight.json').read_text())['status']=='PASS'
assert b'controller.py' in Path(f'/proc/{pid}/cmdline').read_bytes()
(b/'receipts/performance_revision_zero_formal.json').write_text(json.dumps({
 'reason':'adopt measured faster memory/compute schedule before any formal update',
 'controller_pid':pid,'formal_updates':0,'formal_output_created':False,
 'preflight_attempt03':'PASS; retained, not relabeled',
 'performance_benchmark_SGD_updates':0,'server_shutdown_requested':False},indent=2)+'\n')
os.kill(pid,signal.SIGINT);os.kill(pid,signal.SIGCONT)
print('OWNED_HELD_CONTROLLER_INTERRUPTED_BEFORE_FORMAL_START')
PY
sleep 6
cat "$BASE/job_status.json"
