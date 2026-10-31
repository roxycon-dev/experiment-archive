"""Hidden, detached local backup launcher with explicit start/exit evidence.

Launch via Win32_Process.Create so it does not inherit the tool shell job lifetime.
No server processes are started/stopped by this launcher.
"""
import datetime
import json
import os
from pathlib import Path
import runpy
import sys
import traceback

ROOT=Path(__file__).resolve().parents[1]
os.chdir(ROOT)
controller=ROOT/'results/terminal_backup_controller.json'
previous=json.loads(controller.read_text(encoding='utf-8-sig')) if controller.exists() else None
out=ROOT/'logs/terminal_backup_detached_20261024.log'
err=ROOT/'logs/terminal_backup_detached_20261024.err.log'
sys.stdout=out.open('a',encoding='utf-8',buffering=1)
sys.stderr=err.open('a',encoding='utf-8',buffering=1)
record={'pid':os.getpid(),'previous_pid':previous.get('pid') if previous else None,
        'reason':'Previous33400 absent without traceback; detached WMI launcher restores local backup only',
        'launch_method':'Win32_Process.Create/pythonw,hidden,not inherited tool-shell job',
        'stdout':str(out),'stderr':str(err),'no_shutdown':True,
        'updated_at':datetime.datetime.now(datetime.timezone.utc).isoformat()}
controller.write_text(json.dumps(record,indent=2),encoding='utf-8')
exit_path=ROOT/'logs/terminal_backup_detached_exit.json'
exit_path.unlink(missing_ok=True)
sys.argv=[str(ROOT/'scripts/backup_terminal_assets.py'),'--watch']
sys.path.insert(0,str(ROOT/'scripts'))
try:
    runpy.run_path(sys.argv[0],run_name='__main__')
except BaseException:
    traceback.print_exc()
    exit_path.write_text(json.dumps({'status':'EXCEPTION','pid':os.getpid(),
        'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat()}),encoding='utf-8')
    raise
else:
    exit_path.write_text(json.dumps({'status':'NORMAL_EXIT','pid':os.getpid(),
        'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat()}),encoding='utf-8')
