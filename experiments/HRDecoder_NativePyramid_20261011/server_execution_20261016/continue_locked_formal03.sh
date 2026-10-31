set -eu
BASE=/root/autodl-tmp/nativepyramid_sp_v3_20261016
"$BASE/venv/bin/python" - <<'PY'
import os,signal,json
from pathlib import Path
b=Path('/root/autodl-tmp/nativepyramid_sp_v3_20261016');pid=2562
assert b'train.py' in Path(f'/proc/{pid}/cmdline').read_bytes() and os.getpgid(pid)==pid
os.killpg(pid,signal.SIGCONT);os.kill(2160,signal.SIGCONT)
r={'reason':'SIGINT inherited as ignored by background controller; it resumed and launched validated source03',
 'observed_executed_updates':20,'observed_in_flight_update':21,
 'action':'continued same owned process group; no weight reload, rollback, source or runtime switch',
 'active_source_sha256':'d3032fc8b20f30a058adbcf3e6bc5a3485b83adafe9b7cdce0682a2d55d7f627',
 'performance04_deployed':False,'server_shutdown_requested':False}
(b/'receipts/formal_started_before_performance_switch.json').write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps(r))
PY
sleep 3
tail -n 5 "$BASE/training_attempt03.log"
