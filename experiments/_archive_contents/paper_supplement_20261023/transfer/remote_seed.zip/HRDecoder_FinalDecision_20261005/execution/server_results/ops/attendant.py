"""Quiet background transfers, full verified recovery, one shutdown dispatch."""
import sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from qlib.common import read_json,write_json,utc_now
from ops.status import main as status
from ops.mirror import main as mirror

def main():
    while True:
        if (ROOT/'reports/shutdown_receipt.json').exists():return
        try:
            state=status();stage=(state.get('driver') or {}).get('stage')
            if stage=='READY_FOR_LOCAL_BACKUP_AND_SHUTDOWN':
                receipt=mirror(full=True)
                if receipt['stage']=='EXISTING_RECEIVER_LOCK':time.sleep(30);continue
                assert receipt['stage']=='FULL_BACKUP_VERIFIED'
                from ops.shutdown import main as shutdown
                shutdown();write_json(ROOT/'reports/attendant_state.json',{'stage':'AWAITING_PLATFORM_OFF_CONFIRMATION','time':utc_now()});return
            if stage in ('TECHNICAL_PAUSE','CURVE_REVIEW_REQUIRED'):
                mirror(endpoints=True);write_json(ROOT/'reports/attendant_state.json',{'stage':'PAUSED_AWAITING_DIAGNOSIS','driver':state['driver'],'time':utc_now()});return
            mirror(endpoints=True)
            write_json(ROOT/'reports/attendant_state.json',{'stage':'WAITING','time':utc_now()})
        except BaseException as e:
            write_json(ROOT/'reports/attendant_state.json',{'stage':'TRANSFER_OR_CONNECTION_ERROR','reason':str(e),'automatic_restart_training':False,'time':utc_now()});traceback.print_exc()
        time.sleep(30) # Immutable endpoint custody requests; user-facing heartbeat stays45min.

if __name__=='__main__':main()
