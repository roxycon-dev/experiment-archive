set -eu
BASE=/root/autodl-tmp/nativepyramid_sp_v3_20261016
tail -n 8 "$BASE/preflight_performance04.log"
"$BASE/venv/bin/python" - <<'PY'
import json
from pathlib import Path
b=Path('/root/autodl-tmp/nativepyramid_sp_v3_20261016')
for name in ('preflight_performance04/preflight.json','preflight_performance04/diagnostic_progress.json','run_seed42_launch02/progress.json'):
    p=b/name
    if p.exists():
        r=json.loads(p.read_text());print(name,json.dumps({k:r.get(k) for k in ('status','source_sha256','stage','completed_updates','completed','executed_updates','checkpoint_persisted_updates','in_flight_update','error')}))
PY
nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw --format=csv
