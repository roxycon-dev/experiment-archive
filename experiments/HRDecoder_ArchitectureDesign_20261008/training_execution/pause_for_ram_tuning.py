"""Stop only the identified process, preserve last saved formal checkpoint and RNG."""
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'local_preparation'))
from remote import connect
code=r'''import json,os,pathlib,signal,time
root=pathlib.Path('/root/autodl-tmp/HRDecoder_v3_20261008')
reports=root/'reports'; run=root/'training_runs/v3'
saved=json.loads((run/'progress.json').read_text())['completed']
assert saved >= 1000 and (run/'last.pt').exists()
running=json.loads((reports/'GPU_CAMPAIGN_RUNNING.json').read_text()); pid=running['pid']
assert b'train_performance.py' in pathlib.Path('/proc/%d/cmdline'%pid).read_bytes()
observed=json.loads((run/'performance.json').read_text())
(reports/'shutdown_hold.enabled').write_text('RAM cache tuning from saved checkpoint; no reset of weights\n')
os.kill(pid,signal.SIGTERM)
for unused in range(40):
    if (reports/'GPU_CAMPAIGN_REPAIR_HOLD.json').exists(): break
    time.sleep(1)
assert (reports/'GPU_CAMPAIGN_REPAIR_HOLD.json').exists()
record={'status':'BUILDING_SHARED_RAM_CACHE','saved_completed':saved,
    'last_observed_completed':observed['completed_update'],
    'resume_saved_model_optimizer_RNG':True,'time_unix':time.time()}
(reports/'RAM_TUNING_CHECKPOINT_SAVED.json').write_text(json.dumps(record,indent=2)+'\n')
for source,dest in [('v3_training.log','v3_before_ram_cache.log'),('gpu_telemetry.csv','gpu_before_ram_cache.csv'),('GPU_CAMPAIGN_OUTCOME.json','BEFORE_RAM_CACHE_OUTCOME.json'),('GPU_CAMPAIGN_REPAIR_HOLD.json','BEFORE_RAM_CACHE_REPAIR_HOLD.json')]:
    os.replace(reports/source,reports/dest)
os.replace(run/'performance.json',reports/'BEFORE_RAM_CACHE_PERFORMANCE.json')
print(json.dumps(record))
'''
client=connect()
with client.open_sftp() as ftp:
    with ftp.open('/root/autodl-tmp/HRDecoder_v3_20261008/training_execution/pause_for_ram_remote.py','w') as stream: stream.write(code)
_,stdout,stderr=client.exec_command('/root/miniconda3/bin/python /root/autodl-tmp/HRDecoder_v3_20261008/training_execution/pause_for_ram_remote.py')
print(stdout.read().decode(),end=''); error=stderr.read().decode()
assert stdout.channel.recv_exit_status()==0,error
client.close()
