"""Archive current small receipts and fresh runtime evidence without cache hashes."""
from datetime import datetime,timezone,timedelta
import json
from pathlib import Path
import sys
import time
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'local_preparation'))
from remote import connect,HOST,PORT
ROOT='/root/autodl-tmp/HRDecoder_v3_20261008'
DEST=HERE.parent/'cache_execution/server_receipts'

def main():
    observed={}; client=connect()
    names=['reports/GPU_CAMPAIGN_RUNNING.json','reports/GPU_CAMPAIGN_OUTCOME.json',
        'reports/TRAINING_SHUTDOWN_COMMAND_ISSUED.json','reports/RAM_MIXED_READY_SUMMARY.json',
        'reports/GPU_ASYNC_PREFETCH_PASS.json','reports/RAM_PIPELINE_BENCHMARK_PASS.json',
        'reports/GPU_LAYOUT_BENCHMARK.json','reports/FINAL_TUNING_CHECKPOINT_SAVED.json',
        'reports/FINAL_PERFORMANCE_TUNING_PASS.json','reports/FINAL_PERFORMANCE_TUNING_ERROR.json',
        'reports/performance_config.json','training_runs/v3/progress.json',
        'training_runs/v3/performance.json','training_runs/v3/TRAINING_COMPLETE.json',
        'training_runs/v3/protocol.json']
    names += ['training_runs/v3/val_%05d.json'%step for step in range(4000,40001,4000)]
    with client.open_sftp() as ftp:
        for name in names:
            try:
                with ftp.open(ROOT+'/'+name,'r') as stream: raw=stream.read()
                observed[name]=json.loads(raw)
                target=DEST/name; target.parent.mkdir(parents=True,exist_ok=True); target.write_bytes(raw)
            except FileNotFoundError: observed[name]=None
        for name in ('reports/v3_training.log','reports/gpu_telemetry.csv','reports/finish_performance_tuning.log',
            'reports/benchmark_gpu_layout.py.log','reports/benchmark_ram_loader.py.log'):
            try:
                with ftp.open(ROOT+'/'+name,'rb') as stream:
                    size=stream.stat().st_size
                    if size>65536: stream.seek(size-65536)
                    raw=stream.read(65536)
                target=DEST/name; target.parent.mkdir(parents=True,exist_ok=True); target.write_bytes(raw)
            except FileNotFoundError: pass
    code="""import json,pathlib,os
r=pathlib.Path('/root/autodl-tmp/HRDecoder_v3_20261008')
a=json.loads((r/'reports/GPU_CAMPAIGN_RUNNING.json').read_text())
j=json.loads((r/'reports/RAM_CACHE_READY.json').read_text())
print(json.dumps({'training_pid_alive':pathlib.Path('/proc/%d/cmdline'%a['pid']).exists(),
'training_cmdline':pathlib.Path('/proc/%d/cmdline'%a['pid']).read_text().replace('\\0',' ') if pathlib.Path('/proc/%d/cmdline'%a['pid']).exists() else None,
'RAM_owner_alive':pathlib.Path('/proc/%d'%j['pid']).exists(),'RAM_files':len(j['mapping']),
'RAM_bytes':j['bytes'],'memory_current_bytes':int(pathlib.Path('/sys/fs/cgroup/memory.current').read_text()),
'shutdown_hold':(r/'reports/shutdown_hold.enabled').exists(),'last_checkpoint_bytes':(r/'training_runs/v3/last.pt').stat().st_size}))
"""
    with client.open_sftp() as ftp:
        with ftp.open(ROOT+'/training_execution/current_snapshot_remote.py','w') as stream: stream.write(code)
    _,out,err=client.exec_command('/root/miniconda3/bin/python '+ROOT+'/training_execution/current_snapshot_remote.py')
    raw=out.read().decode(); error=err.read().decode(); assert out.channel.recv_exit_status()==0,error
    runtime=json.loads(raw); client.close()
    snapshot={'time_unix':time.time(),'server':{'host':HOST,'port':PORT,'remote_root':ROOT,'password_stored':False},
        'runtime':runtime,'receipts':observed,'hash_rescan':False}
    (DEST/'GPU_CURRENT_SNAPSHOT.json').write_text(json.dumps(snapshot,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    state_path=HERE.parent/'cache_execution/handover_state.json'
    state=json.loads(state_path.read_text(encoding='utf-8'))
    performance=observed['training_runs/v3/performance.json']
    completed=observed['training_runs/v3/TRAINING_COMPLETE.json']
    tuning=observed['reports/FINAL_PERFORMANCE_TUNING_PASS.json']
    outcome=observed['reports/GPU_CAMPAIGN_OUTCOME.json']
    verified=completed and outcome and outcome.get('status')=='COMPLETE' and outcome.get('persistent_checkpoint_verification',{}).get('status')=='PASS'
    status='RUNNING' if runtime['training_pid_alive'] else ('COMPLETE' if verified else ('COMPLETION_VERIFYING' if completed else ('FAILED' if outcome and outcome.get('status')=='FAILED' else 'PERFORMANCE_TUNING')))
    state.update(stage='V3_'+status,formal_training_status=status,server=snapshot['server'],
        latest_performance=performance,current_runtime=runtime,performance_tuning_in_progress=tuning is None,
        latest_user_instruction='All teacher caches in RAM; fastest measured training; preserve v3 2048 batch2 40k; shut down after completion.',
        server_controller=observed['reports/GPU_CAMPAIGN_RUNNING.json'],
        full_RAM_cache=observed['reports/RAM_MIXED_READY_SUMMARY.json'],
        last_monitor_time_unix=snapshot['time_unix'],
        resume_first='Check current PID and latest receipts before any launch. Explicitly resume last.pt; never duplicate a running process.',
        current_completion_receipt=completed,current_outcome=observed['reports/GPU_CAMPAIGN_OUTCOME.json'])
    state_path.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':status,'runtime':runtime,'performance':performance,'tuning':tuning},ensure_ascii=False))

if __name__=='__main__':main()
