set -eu
BASE=/root/autodl-tmp/nativepyramid_sp_v3_20261016
"$BASE/venv/bin/python" - <<'PY'
import os,signal,json,time
from pathlib import Path
b=Path('/root/autodl-tmp/nativepyramid_sp_v3_20261016')
s=json.loads((b/'job_status.json').read_text())
assert s['status']=='PREFLIGHT' and s['formal_training_started'] is False
assert not (b/'run_seed42_launch02').exists()
pid=2160;command=Path(f'/proc/{pid}/cmdline').read_bytes()
assert b'controller.py' in command
os.kill(pid,signal.SIGSTOP)
(b/'receipts/performance_hold_before_formal.json').write_text(json.dumps({
 'controller_pid':pid,'formal_updates':0,'reason':'user requested maximum performance; zero-SGD memory/speed comparison',
 'preflight_child_continues':2162,'server_shutdown_requested':False,'unix_time':time.time()},indent=2)+'\n')
print('OWNED_CONTROLLER_HELD; PREFLIGHT_CONTINUES; NO_FORMAL_SGD')
PY
