set -eu
BASE=/root/autodl-tmp/nativepyramid_sp_v3_20261016
"$BASE/venv/bin/python" - <<'PY'
import json
from pathlib import Path
b=Path('/root/autodl-tmp/nativepyramid_sp_v3_20261016')
for rel in ['job_status.json','job_result.json','receipts/shutdown_request.json','finalization_failure.json','run_seed42_launch02/progress.json','run_seed42_launch02/result.json','run_seed42_launch02/endpoint_lock.json','run_seed42_launch02/failure.json','run_seed42_launch02/test_result.json','run_seed42_launch02/test_failure.json']:
    p=b/rel
    if not p.is_file():
        print(json.dumps({'file':rel,'exists':False}));continue
    d=json.loads(p.read_text())
    out={k:d.get(k) for k in ['source_sha256','controller_version','status','outcome','observed_at','finished_at','shutdown_requested_at','requested_at','child_pid','child_exit_code','formal_progress','completed','executed_updates','checkpoint_persisted_updates','in_flight_update','checkpoint_sha256','test_permitted','test_accessed','primary','best_val_secondary','finalization_error','error'] if k in d}
    if 'val' in d: out['val']={k:d['val'].get(k) for k in ['update','mIoU_percent','IoU_percent','images']}
    out.update(file=rel,mtime=p.stat().st_mtime)
    print(json.dumps(out))
PY
ps -p 2160,2562 -o pid,ppid,stat,etime,wchan:28,args || true
if test -f "$BASE/shutdown.log"; then tail -n 8 "$BASE/shutdown.log"; fi
