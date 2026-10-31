"""Wait for owned CET to finish, then run SACD gates and one fresh launch."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
REPORTS=ROOT/'reports/sacd'
PYTHON='/root/miniconda3/bin/python'
created_own_ram=False

def write(value):
    value['time_unix']=time.time()
    p=REPORTS/'TRANSITION_STATUS.json';tmp=p.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(p)

def alive(pid,expected=None):
    try:
        proc=Path('/proc/%d'%pid)
        state=(proc/'stat').read_text().split(') ',1)[1].split()[0]
        cmd=(proc/'cmdline').read_bytes()
        return state!='Z' and (expected is None or expected.encode() in cmd)
    except OSError:return False

def main():
    global created_own_ram
    with open(REPORTS/'transition.lock','a') as own_lock:
        fcntl.flock(own_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not (REPORTS/'FINAL_ROUND_STARTED.json').exists()
        start=time.monotonic()
        while True:
            outcome_path=ROOT/'reports/cet/GPU_CAMPAIGN_OUTCOME.json'
            if outcome_path.exists():
                outcome=json.loads(outcome_path.read_text())
                assert outcome['status']=='COMPLETE','CET did not complete; manual investigation required'
                marker=json.loads((ROOT/'training_runs/cet/TRAINING_COMPLETE.json').read_text())
                val=json.loads((ROOT/'training_runs/cet/val_40000.json').read_text())
                assert marker['completed']==40000 and marker['durable_checkpoint_loaded_and_verified'] and val['images']==76
                running=json.loads((ROOT/'reports/cet/GPU_CAMPAIGN_RUNNING.json').read_text())
                ram=json.loads((ROOT/'reports/cet/RAM_CACHE_READY.json').read_text())
                if not alive(running['pid']) and not alive(running['controller_pid']) and not alive(ram['pid'],str(ROOT/'cet_execution/ram_cache_owner.py')):
                    break
            assert time.monotonic()-start<900,'CET transition wait exceeded 15 minutes'
            write({'status':'WAITING_CET_COMPLETION','server_shutdown':False})
            time.sleep(5)
        cpu=json.loads((HERE/'CPU_SACD_PASS.json').read_text())
        resume=json.loads((HERE/'CHECKPOINT_RESUME_PASS.json').read_text())
        assert cpu['status']==resume['status']=='PASS' and cpu['torch']==resume['torch']=='2.0.1+cu117'
        ready_path=REPORTS/'RAM_CACHE_READY.json'
        if ready_path.exists():
            previous=json.loads(ready_path.read_text())
            assert not alive(previous['pid'],str(HERE/'ram_cache_owner.py')),'Own cache already alive; do not duplicate'
            ready_path.unlink()
        write({'status':'BUILDING_OWN_RAM','CET_terminal_verified':True,'CET_own_RAM_exited':True})
        with open(REPORTS/'ram_cache.log','ab',buffering=0) as log:
            ram_process=subprocess.Popen([PYTHON,'-u',str(HERE/'ram_cache_owner.py')],cwd=str(HERE),
                stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
            created_own_ram=True
        while not (REPORTS/'RAM_CACHE_READY.json').exists():
            assert ram_process.poll() is None,'SACD RAM owner failed'
            assert time.monotonic()-start<1500,'RAM build timed out'
            time.sleep(2)
        with open(ROOT/'reports/gpu.lock','a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],universal_newlines=True).strip()
            assert not gpu,'Unrelated GPU work active; do not control it'
            from train import identity
            _,digest=identity()
            smoke_path=ROOT/'training_runs/sacd/GPU_SMOKE_PASS.json'
            smoke=json.loads(smoke_path.read_text()) if smoke_path.exists() else {}
            valid=smoke.get('status')=='PASS' and smoke.get('identity_SHA')==digest and smoke.get('discarded_optimizer_steps')==12
            write({'status':'REUSING_VERIFIED_GPU_SMOKE' if valid else 'GPU_SMOKE',
                   'CET_terminal_verified':True,'own_RAM_pid':ram_process.pid,'identity_SHA':digest})
            if not valid:
                env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1')
                with open(REPORTS/'gpu_smoke.log','ab',buffering=0) as log:
                    subprocess.check_call([PYTHON,'-u','train.py','--smoke'],cwd=str(HERE),env=env,stdout=log,stderr=log)
        subprocess.check_call([PYTHON,'start_controller.py'],cwd=str(HERE))
        write({'status':'FORMAL_TRAINING_LAUNCHED','CET_terminal_verified':True,
               'GPU_smoke_steps_discarded':12,'full_training_runs':1,'server_shutdown':False})

if __name__=='__main__':
    try:main()
    except Exception as error:
        write({'status':'FAILED','error':str(error),'traceback':traceback.format_exc(),'server_shutdown':False})
        if created_own_ram:
            from controller import terminate_own_ram
            terminate_own_ram()
        raise
