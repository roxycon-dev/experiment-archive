"""Wait for the already-running evaluation, then perform bounded authorized validation."""
import json
import argparse
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
BASE = Path('C:/Users/GuanXingRen/Miniconda3/python.exe')
LEGACY = Path('C:/Users/GuanXingRen/Miniconda3/envs/M2MRF-Lesion-Segmentation/python.exe')
DDR = ROOT/'results/ddr_test_local_v1'


def run(python, script, args, log):
    with (ROOT/'logs'/log).open('w',encoding='utf-8') as f:
        p = subprocess.run([str(python),str(ROOT/'scripts'/script),*args],cwd=str(ROOT.parent),
                           stdout=f,stderr=subprocess.STDOUT,creationflags=0x08000000)
    if p.returncode:raise RuntimeError(script+' failed; see '+log)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--local-only',action='store_true');args=parser.parse_args()
    state = ROOT/'results/queue_progress.json'
    def status(stage):
        state.write_text(json.dumps({'stage':stage,'updated_unix':time.time()},indent=2),encoding='utf-8')
        print(stage,flush=True)
    status('WAITING_FOR_DDR_COMPLETION')
    while not (DDR/'completion.json').exists():
        if (DDR/'failure.json').exists():raise RuntimeError('DDR run failed; stop dependent queue')
        time.sleep(10)
    # Give the completed driver time to shut down its own worker before GPU reuse.
    time.sleep(15)
    status('ANALYZING_DDR')
    run(BASE,'analyze_fixed_run.py',[str(DDR)],'ddr_analysis.log')
    status('PROFILING_FULL_SYSTEM')
    run(LEGACY,'run_cohort_system.py',['--mode','profile','--run-name','full_system_profile_v1'],'full_system_profile.log')
    run(BASE,'analyze_cohort_run.py',[str(ROOT/'results/full_system_profile_v1')],'profile_analysis.log')
    if args.local_only:
        status('LOCAL_COMPLETE_REMOTE_HOLDOUT_PENDING');return
    status('EVALUATING_IDRID_HOLDOUT')
    run(LEGACY,'run_cohort_system.py',['--mode','external','--run-name','idrid_holdout_v1','--max-estimated-seconds','3600'],'idrid_holdout.log')
    status('ANALYZING_IDRID_HOLDOUT')
    run(BASE,'analyze_cohort_run.py',[str(ROOT/'results/idrid_holdout_v1')],'idrid_analysis.log')
    status('ALL_QUEUED_EVALUATIONS_COMPLETE')


if __name__ == '__main__':
    try:main()
    except BaseException as error:
        (ROOT/'results/queue_failure.json').write_text(json.dumps({'type':type(error).__name__,'message':str(error)},indent=2),encoding='utf-8')
        raise
