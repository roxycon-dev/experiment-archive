set -eu
BASE=/root/autodl-tmp/nativepyramid_sp_v3_20261016
"$BASE/venv/bin/python" - <<'PY'
import json,time,subprocess,statistics
from pathlib import Path
b=Path('/root/autodl-tmp/nativepyramid_sp_v3_20261016');run=b/'run_seed42_launch02'
samples=[]
for i in range(10):
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,power.draw,temperature.gpu,clocks.sm','--format=csv,noheader,nounits'],text=True)
    samples.append([float(v.strip()) for v in raw.strip().split(',')]);time.sleep(1)
p=run/'train.jsonl'
with p.open('rb') as f:
    size=p.stat().st_size;offset=max(0,size-16*1024**2);f.seek(offset);lines=f.read().splitlines()
    if offset:lines=lines[1:]
rows=[json.loads(line) for line in lines][-51:]
deltas=[right['seconds_this_process']-left['seconds_this_process'] for left,right in zip(rows,rows[1:]) if left['update']>25]
progress=json.loads((run/'progress.json').read_text())
record={'completed':progress['completed'],'sampled_updates':len(deltas),
 'mean_seconds_per_update':statistics.mean(deltas) if deltas else None,
 'median_seconds_per_update':statistics.median(deltas) if deltas else None,
 'gpu_utilization_mean_percent':statistics.mean(s[0] for s in samples),
 'gpu_utilization_range_percent':[min(s[0] for s in samples),max(s[0] for s in samples)],
 'gpu_power_mean_watts':statistics.mean(s[2] for s in samples),
 'gpu_memory_max_MiB':max(s[1] for s in samples),'gpu_temperature_max_c':max(s[3] for s in samples),
 'gpu_clock_mean_MHz':statistics.mean(s[4] for s in samples),'raw_samples':samples,
 'wall_clock_unix':time.time(),'note':'short sample; excludes earlier control hold; evaluation/I/O can affect overall ETA'}
(b/'receipts/active_training_performance.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record),flush=True)
PY
