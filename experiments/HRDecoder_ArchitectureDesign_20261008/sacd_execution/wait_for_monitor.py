"""One local scheduled observation; leaves the paid GPU and all processes alone."""
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
ledger=json.loads((HERE/'validation_reporting.json').read_text(encoding='utf-8'))
last=datetime.fromisoformat(ledger['last_checked_at'])
due=last+timedelta(minutes=45)
receipt={'status':'WAITING','pid':os.getpid(),'last_checked_at':last.isoformat(),
         'next_check_at':due.isoformat(),'poll_interval_minutes':45,
         'remote_training_PID':25190,'current_model':'sacd'}
(HERE/'scheduled_observation.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt),flush=True)
while time.time()<due.timestamp():
    time.sleep(min(5,max(.01,due.timestamp()-time.time())))
result=subprocess.run([sys.executable,str(HERE/'monitor_sacd.py')],stdout=subprocess.PIPE,
                      stderr=subprocess.STDOUT,encoding='utf-8')
print(result.stdout,flush=True)
receipt.update(status='OBSERVED' if result.returncode==0 else 'OBSERVATION_FAILED',
               returned_at=datetime.now(timezone(timedelta(hours=8))).isoformat(),exit_code=result.returncode)
(HERE/'scheduled_observation.json').write_text(json.dumps(receipt,indent=2)+'\n')
raise SystemExit(result.returncode)
