"""Finite serial acceptance/import/G queue, with explicit process status."""
from pathlib import Path
import json
import subprocess
import os
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from nextlib.common import write_json
env=dict(os.environ,PATH='/root/miniconda3/bin:'+os.environ.get('PATH',''),
         OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONUNBUFFERED='1')
steps=[('relocate',[sys.executable,'configure.py','--ddr-root','/root/HRDecoder/data/DDR_split_613',
                   '--prefer-existing','/root/autodl-tmp/HRDecoder_Ready_20261002']),
       ('acceptance',[sys.executable,'preflight.py','--server']),
       ('import',[sys.executable,'ops/import_existing.py']),
       ('G',[sys.executable,'run_campaign.py','--stage','G'])]
for stage,cmd in steps:
    write_json(ROOT/'reports/server_stage.json',dict(stage=stage,status='running',time=time.time(),pid=os.getpid()))
    with (ROOT/f'reports/{stage}.log').open('w') as log:
        result=subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
    if result.returncode:
        write_json(ROOT/'reports/server_stage.json',dict(stage=stage,status='failed',exit_code=result.returncode,time=time.time()))
        sys.exit(result.returncode)
write_json(ROOT/'reports/server_stage.json',dict(stage='G',status='complete',time=time.time(),
    next='Inspect locked geometry/calibration and start paired M'))
