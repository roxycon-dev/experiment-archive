"""Launch only this CET controller, with PID and cmdline duplicate protection."""
import json
import os
from pathlib import Path
import subprocess
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
REPORTS=ROOT/'reports/cet'
SCRIPT=str(HERE/'controller.py')
for p in Path('/proc').iterdir():
    if p.name.isdigit() and int(p.name)!=os.getpid():
        try:
            if SCRIPT.encode() in (p/'cmdline').read_bytes():
                raise RuntimeError('Existing CET controller PID '+p.name+'; do not launch duplicate')
        except OSError:pass
assert not (ROOT/'training_runs/cet/last.pt').exists(),'Existing checkpoint requires explicit resume, not fresh launch'
log=open(REPORTS/'controller.log','ab',buffering=0)
child=subprocess.Popen(['/root/miniconda3/bin/python','-u',SCRIPT],cwd=str(HERE),
                       stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
time.sleep(2)
if child.poll() is not None:
    raise RuntimeError('Controller exited: '+(REPORTS/'controller.log').read_text()[-3000:])
print(json.dumps({'status':'CONTROLLER_LAUNCHED','pid':child.pid,'server_shutdown':False}))
