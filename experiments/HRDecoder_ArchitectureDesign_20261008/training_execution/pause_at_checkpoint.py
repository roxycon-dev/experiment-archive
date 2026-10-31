"""Wait for the next normal durable checkpoint before switching runtime helpers."""
import json
import os
from pathlib import Path
import signal
import time

ROOT=Path(__file__).resolve().parent.parent
reports=ROOT/'reports'; run=ROOT/'training_runs/v3'
running=json.loads((reports/'GPU_CAMPAIGN_RUNNING.json').read_text())
pid=running['pid']
initial=json.loads((run/'progress.json').read_text())['completed']
assert initial < 40000 and b'train_performance.py' in Path('/proc/%d/cmdline'%pid).read_bytes()
target=(initial//1000+1)*1000
(reports/'shutdown_hold.enabled').write_text('Authorized performance tuning at next saved checkpoint\n')
print(json.dumps({'waiting_for_saved_update':target,'pid':pid}),flush=True)
while True:
    saved=json.loads((run/'progress.json').read_text())['completed']
    if saved>=target: break
    os.kill(pid,0); time.sleep(.1)
os.kill(pid,signal.SIGTERM)
for unused in range(400):
    if (reports/'GPU_CAMPAIGN_REPAIR_HOLD.json').exists(): break
    time.sleep(.1)
assert (reports/'GPU_CAMPAIGN_REPAIR_HOLD.json').exists()
observed=json.loads((run/'performance.json').read_text())['completed_update']
stamp=str(int(time.time()))
record={'status':'SAVED_FOR_FINAL_PERFORMANCE_TUNING','saved_completed':saved,
    'last_measured_update':observed,'resume_model_optimizer_RNG':True,'time_unix':time.time()}
for source in ('v3_training.log','gpu_telemetry.csv','GPU_CAMPAIGN_OUTCOME.json','GPU_CAMPAIGN_REPAIR_HOLD.json'):
    if (reports/source).exists(): os.replace(reports/source,reports/('before_final_tuning_'+stamp+'_'+source))
os.replace(run/'performance.json',reports/('before_final_tuning_'+stamp+'_performance.json'))
(reports/'FINAL_TUNING_CHECKPOINT_SAVED.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record),flush=True)
