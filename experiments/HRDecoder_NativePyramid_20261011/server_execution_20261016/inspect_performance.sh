set -eu
BASE=/root/autodl-tmp/nativepyramid_sp_v3_20261016
"$BASE/venv/bin/python" - <<'PY'
import os,json,hashlib
from pathlib import Path
b=Path('/root/autodl-tmp/nativepyramid_sp_v3_20261016')
r={'logical_cpu':os.cpu_count(),'affinity':sorted(os.sched_getaffinity(0)),
   'current_shutdown_script_sha256':hashlib.sha256(Path('/usr/bin/shutdown').read_bytes()).hexdigest(),
   'previous_shutdown_script_sha256':json.loads((b/'receipts/shutdown_wrapper_preparation.json').read_text())['source_sha256']}
for p in ('/sys/fs/cgroup/cpu.max','/sys/fs/cgroup/memory.max','/sys/fs/cgroup/cpu/cpu.cfs_quota_us','/sys/fs/cgroup/cpu/cpu.cfs_period_us','/sys/fs/cgroup/memory/memory.limit_in_bytes'):
    if Path(p).is_file():r[p]=Path(p).read_text().strip()
print(json.dumps(r,indent=2))
PY
nvidia-smi --query-gpu=name,pstate,temperature.gpu,power.draw,power.limit,utilization.gpu,memory.used,clocks.sm,clocks.mem --format=csv
ps -eo pid,comm,args | grep -E 'controller|bootstrap|train.py|verify.py|diagnose_backend' | grep -v grep || true
