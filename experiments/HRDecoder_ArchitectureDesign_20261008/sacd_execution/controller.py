"""Keep SACD alive across SSH disconnects; record outcome and keep server on."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
import traceback
from datetime import datetime, timezone, timedelta

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
REPORTS=ROOT/'reports/sacd'
RUN=ROOT/'training_runs/sacd'

def write(name,value):
    path=REPORTS/name;tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)

def terminate_own_ram():
    try:
        ready=json.loads((REPORTS/'RAM_CACHE_READY.json').read_text())
        cmd=Path('/proc/%d/cmdline'%ready['pid']).read_bytes()
        if str(HERE/'ram_cache_owner.py').encode() in cmd:
            os.kill(ready['pid'],15)
            return {'own_RAM_owner_termination_requested':True,'pid':ready['pid']}
    except (OSError,ValueError):pass
    return {'own_RAM_owner_termination_requested':False}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    REPORTS.mkdir(parents=True,exist_ok=True)
    with open(ROOT/'reports/gpu.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        # Existing unrelated work is never terminated to claim the GPU.
        gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],universal_newlines=True).strip()
        assert not gpu,'GPU already belongs to another process; refusing duplicate launch'
        assert args.resume or not (RUN/'last.pt').exists(),'Explicit SACD resume required'
        env=dict(os.environ,PYTHONPATH=str(ROOT/'vendor/champion_repo'),
                 OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='4')
        command=['/root/miniconda3/bin/python','-u','train.py','--train']
        if args.resume:command.append('--resume')
        started=time.time()
        with open(REPORTS/'gpu_telemetry.csv','a',buffering=1) as telemetry,open(REPORTS/'sacd_training.log','a',buffering=1) as log:
            monitor=subprocess.Popen(['nvidia-smi',
                '--query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,power.limit,temperature.gpu,clocks.sm',
                '--format=csv,noheader,nounits','--loop=2'],stdout=telemetry,stderr=log)
            training=subprocess.Popen(command,cwd=str(HERE),env=env,stdout=log,stderr=log)
            running={'status':'RUNNING','name':'HRDecoder-SACD','pid':training.pid,
                     'controller_pid':os.getpid(),'telemetry_pid':monitor.pid,'start_unix':started,
                     'updates':40000,'batch':2,'canvas':2048,'resume':args.resume,
                     'shutdown_after_completion':False,'shutdown_after_failure':False,
                     'do_not_control_other_AI_diagnostics':True}
            write('GPU_CAMPAIGN_RUNNING.json',running)
            cutoff=datetime(2026,10,10,2,0,tzinfo=timezone(timedelta(hours=8))).timestamp()-180
            deadline_stop=False
            try:
                code=training.wait(timeout=max(1,cutoff-time.time()))
            except subprocess.TimeoutExpired:
                # Only the subprocess launched here is controlled. Existing atomic
                # checkpoints remain intact; no external diagnosis is touched.
                deadline_stop=True;training.terminate()
                try:code=training.wait(timeout=20)
                except subprocess.TimeoutExpired:training.kill();code=training.wait()
                write('DEADLINE_STOP.json',{'status':'STOPPED_AT_USER_DEADLINE',
                    'time_unix':time.time(),'training_pid':training.pid,'checkpoint_preserved':True,
                    'server_kept_running':True})
            monitor.terminate()
            try:monitor.wait(timeout=10)
            except subprocess.TimeoutExpired:monitor.kill();monitor.wait()
        marker=json.loads((RUN/'TRAINING_COMPLETE.json').read_text()) if (RUN/'TRAINING_COMPLETE.json').exists() else None
        final=json.loads((RUN/'val_40000.json').read_text()) if (RUN/'val_40000.json').exists() else None
        passed=not deadline_stop and code==0 and marker is not None and marker.get('completed')==40000 and marker.get('durable_checkpoint_loaded_and_verified') and final is not None and final.get('images')==76
        test_status={'status':'NOT_RUN','reason':'Training incomplete'}
        if passed:
            if final['mIoU_percent']>45.32183122738841:
                # Evaluate the pre-registered fixed endpoint once, never best-val.
                with open(REPORTS/'test_evaluation.log','a',buffering=1) as test_log:
                    test_code=subprocess.call(['/root/miniconda3/bin/python','-u','test_eval.py'],
                        cwd=str(HERE),env=env,stdout=test_log,stderr=test_log)
                test_status={'status':'COMPLETE' if test_code==0 else 'FAILED','exit_code':test_code}
            else:
                test_status={'status':'SKIPPED','reason':'Fixed40k val did not exceed fusion; combined goal not achieved'}
        write('TEST_EVALUATION_OUTCOME.json',test_status)
        outcome={'status':'COMPLETE' if passed else 'FAILED','exit_code':code,
                 'time_unix':time.time(),'start_unix':started,'completion':marker,
                 'final_validation':final,'shutdown_command_issued':False,'server_kept_running':True,
                 'checkpoint_download_requested':False,'test_evaluation':test_status,**terminate_own_ram()}
        write('GPU_CAMPAIGN_OUTCOME.json',outcome)
        running.update(status=outcome['status'],end_unix=outcome['time_unix'],processes_alive=False)
        write('GPU_CAMPAIGN_RUNNING.json',running)
        print(json.dumps(outcome),flush=True)
        os.sync()
        if not passed:raise SystemExit(1)

if __name__=='__main__':
    try:main()
    except Exception as error:
        REPORTS.mkdir(parents=True,exist_ok=True)
        write('CONTROLLER_ERROR.json',{'status':'FAILED','error':str(error),'traceback':traceback.format_exc(),
                                      'time_unix':time.time(),'shutdown_command_issued':False})
        raise
