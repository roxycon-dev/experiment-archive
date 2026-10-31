import json, os, pathlib, signal, time
root=pathlib.Path('/root/autodl-tmp/HRDecoder_v3_20261008')
reports=root/'reports'; run=root/'training_runs/v3'
assert not (run/'last.pt').exists(), 'Existing formal checkpoint: do not discard'
running=json.loads((reports/'GPU_CAMPAIGN_RUNNING.json').read_text())
pid=running['pid']
assert b'train_performance.py' in pathlib.Path('/proc/%d/cmdline'%pid).read_bytes()
(reports/'shutdown_hold.enabled').write_text('Agent repairing startup data pipeline\n')
performance=json.loads((run/'performance.json').read_text())
assert performance['measured_batches'] < 1000
os.kill(pid, signal.SIGTERM)
for unused in range(30):
    if (reports/'GPU_CAMPAIGN_REPAIR_HOLD.json').exists(): break
    time.sleep(1)
assert (reports/'GPU_CAMPAIGN_REPAIR_HOLD.json').exists()
record={'reason':'Initial data-loading throughput bottleneck; tune workers before first checkpoint', 'last_observed_batches':performance['measured_batches'], 'weights_discarded':True, 'retained_training_updates':0, 'replacement_fresh_seed42':True, 'time_unix':time.time()}
(reports/'STARTUP_PIPELINE_REPLACEMENT.json').write_text(json.dumps(record,indent=2)+'\n')
for source, dest in [('v3_training.log','v3_startup_workers4.log'),('gpu_telemetry.csv','gpu_startup_workers4.csv'),('GPU_CAMPAIGN_OUTCOME.json','STARTUP_WORKERS4_OUTCOME.json'),('GPU_CAMPAIGN_REPAIR_HOLD.json','STARTUP_WORKERS4_REPAIR_HOLD.json')]:
    os.replace(reports/source,reports/dest)
os.replace(run/'performance.json',reports/'STARTUP_WORKERS4_PERFORMANCE.json')
(reports/'shutdown_hold.enabled').unlink()
print(json.dumps(record))
