"""Local completion listener; agent heartbeat stays at45min. Starts after deployment."""
import json
from pathlib import Path
import shlex
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from qlib.common import read_json,utc_now,write_json
from ops.remote import connect,command
from ops.mirror import main as mirror
from ops.shutdown import main as shutdown


def main():
    runtime=read_json(ROOT/'configs/runtime.local.json');remote=runtime['server']['remote_root']
    # Keep a bounded-to-terminal-state remote waiter, so final backup starts when
    # the campaign ends without waiting for the next45min agent wakeup.
    waiter='''
import json,time
from pathlib import Path
r=Path(%r)
while True:
 p=r/'reports/driver_state.json'
 if p.exists():
  s=json.loads(p.read_text())
  if s.get('stage') in ('READY_FOR_LOCAL_BACKUP_AND_SHUTDOWN','TECHNICAL_PAUSE'):
   print(json.dumps(s),flush=True);break
 time.sleep(30)
'''%remote
    with connect() as client:
        result=json.loads(command(client,shlex.quote(runtime['server_python'])+" - <<'PY'\n"+waiter+"\nPY",timeout=None))
    if result['stage']=='TECHNICAL_PAUSE':
        mirror(False)
        write_json(ROOT/'reports/attendant_state.json',dict(result,automatic_retry=False));return
    mirror(True);shutdown()
    write_json(ROOT/'reports/attendant_state.json',{'stage':'BACKUP_COMPLETE_SHUTDOWN_DISPATCHED_OFF_CHECK_PENDING','time':utc_now()})


if __name__=='__main__':
    try:main()
    except BaseException as error:
        write_json(ROOT/'reports/attendant_state.json',{'stage':'TECHNICAL_PAUSE','reason':str(error),'automatic_retry':False,'time':utc_now()})
        raise
