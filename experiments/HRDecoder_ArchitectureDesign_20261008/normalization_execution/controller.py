"""Bounded paired diagnostics only; no new full training, no shutdown."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
REPORTS=ROOT/'reports/residual_normalization'
WORK=ROOT/'training_runs/residual_normalization'
STOP=1791568620


def write(name,value):
    path=REPORTS/name
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,indent=2)+'\n')
    tmp.replace(path)


def main():
    REPORTS.mkdir(parents=True,exist_ok=True)
    WORK.mkdir(parents=True,exist_ok=True)
    assert not (REPORTS/'PROBE_STARTED.json').exists()
    env=dict(os.environ,PYTHONPATH=str(ROOT/'vendor/champion_repo'),
             OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='4')
    subprocess.run([sys.executable,'verify_normalization.py'],cwd=str(HERE),env=env,check=True)
    cpu=json.loads((HERE/'NORMALIZATION_CPU_PASS.json').read_text())
    assert cpu['status']=='PASS' and cpu['torch']=='2.0.1+cu117'
    # Conservative two-arm compute plus validation/startup/checkpoint allowance.
    assert time.time()+1000*1.1+480<STOP,'Cannot finish meaningful paired diagnostic before deadline'
    with open(ROOT/'reports/gpu.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid',
                   '--format=csv,noheader'],universal_newlines=True).strip(),'Foreign GPU process; do not interfere'
        with (REPORTS/'PROBE_STARTED.json').open('x') as file:
            json.dump({'kind':'paired warm diagnostic, not fresh screen','time_unix':time.time(),
              'controller_pid':os.getpid(),'own_stop_unix':STOP,'automatic_extension':False,
              'automatic_fresh_training':False,'server_kept_running':True},file,indent=2)
        for arm in ('control','gt_stratified'):
            with open(REPORTS/(arm+'.log'),'a',buffering=1) as log:
                child=subprocess.Popen([sys.executable,'-u','probe.py','--arm',arm],cwd=str(HERE),
                    stdout=log,stderr=log,env=env)
                write('RUNNING.json',{'status':'PAIRED_DIAGNOSTIC_RUNNING','arm':arm,
                      'pid':child.pid,'controller_pid':os.getpid(),'time_unix':time.time(),
                      'own_stop_unix':STOP,'server_kept_running':True})
                while child.poll() is None:
                    if time.time()>=STOP:
                        child.terminate()
                        try:child.wait(timeout=15)
                        except subprocess.TimeoutExpired:child.kill();child.wait()
                        raise RuntimeError('Own paired diagnostic reached original deadline reserve')
                    time.sleep(2)
                assert child.returncode==0,arm+' probe failed'
        values={}
        for arm in ('control','gt_stratified'):
            marker=json.loads((WORK/arm/'PROBE_COMPLETE.json').read_text())
            assert marker['probe_updates']==500 and marker['parent_completed']==8000
            values[arm]={str(n):json.loads((WORK/arm/('val_probe_%05d.json'%n)).read_text()) for n in (0,250,500)}
        write('OUTCOME.json',{'status':'COMPLETE','time_unix':time.time(),'values':values,
            'only_KL_normalization_changed':True,'parent_completed':8000,'warm_updates_per_arm':500,
            'fresh_screen_pass':False,'deadline_extension':False,'automatic_fresh_training':False,
            'test_accessed':False,'goal_achieved':False,'server_kept_running':True,
            'shutdown_command_issued':False,'external_diagnostics_controlled':False,
            'own_RAM_retained':True})
        write('RUNNING.json',{'status':'COMPLETE','time_unix':time.time(),'controller_pid':os.getpid(),
                             'processes_alive':False,'server_kept_running':True})
        os.sync()


if __name__=='__main__':
    try:main()
    except BaseException as error:
        write('ERROR.json',{'status':'FAILED','time_unix':time.time(),'error':str(error),
             'traceback':traceback.format_exc(),'server_kept_running':True,
             'shutdown_command_issued':False,'external_diagnostics_controlled':False})
        raise
