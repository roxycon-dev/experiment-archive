"""Checkpoint-safe measured tuning, then immediate explicit resume of the main run."""
import json
import os
from pathlib import Path
import subprocess
import time
ROOT=Path(__file__).resolve().parent.parent
HERE=ROOT/'training_execution'; REPORTS=ROOT/'reports'
PYTHON='/root/miniconda3/bin/python'

def main():
    while not (REPORTS/'FINAL_TUNING_CHECKPOINT_SAVED.json').exists(): time.sleep(1)
    ready=json.loads((REPORTS/'RAM_CACHE_READY.json').read_text())
    assert ready.get('all_teachers_in_RAM') and len(ready['mapping'])==1362
    saved=json.loads((REPORTS/'FINAL_TUNING_CHECKPOINT_SAVED.json').read_text())['saved_completed']
    env=dict(os.environ,PYTHONPATH=str(ROOT/'vendor/champion_repo'),OMP_NUM_THREADS='4',
        OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='4',HRDECODER_BENCH_WORKERS='8,12',
        HRDECODER_BENCH_FRESH='1',HRDECODER_BENCH_START=str(saved))
    for name in ('benchmark_ram_loader.py','benchmark_gpu_layout.py'):
        with open(REPORTS/(name+'.log'),'w',buffering=1) as log:
            subprocess.run([PYTHON,'-u',str(HERE/name)],env=env,cwd=str(HERE),stdout=log,stderr=log,check=True)
    layout=json.loads((REPORTS/'GPU_LAYOUT_BENCHMARK.json').read_text())
    config=json.loads((REPORTS/'performance_config.json').read_text())
    config.update(channels_last=layout['adopt_channels_last'],CUDA_prefetch_slots=1)
    (REPORTS/'performance_config.json').write_text(json.dumps(config,indent=2)+'\n')
    (REPORTS/'shutdown_hold.enabled').unlink()
    with open(REPORTS/'gpu_controller.log','ab',buffering=0) as log:
        p=subprocess.Popen([PYTHON,'-u',str(HERE/'gpu_campaign_controller.py')],cwd=str(HERE),
            stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
    (REPORTS/'FINAL_PERFORMANCE_TUNING_PASS.json').write_text(json.dumps({'status':'PASS',
        'resumed_from_saved_update':saved,'config':config,'controller_pid':p.pid,
        'time_unix':time.time(),'full_RAM_cache_bytes':ready['bytes']},indent=2)+'\n')
    print('FINAL_PERFORMANCE_TUNING_PASS',flush=True)

if __name__=='__main__':
    try: main()
    except BaseException as error:
        (REPORTS/'FINAL_PERFORMANCE_TUNING_ERROR.json').write_text(json.dumps({'error':repr(error),'time_unix':time.time()},indent=2)+'\n')
        raise
