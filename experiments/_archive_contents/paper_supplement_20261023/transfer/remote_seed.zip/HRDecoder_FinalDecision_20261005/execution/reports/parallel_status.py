"""Read both explicitly assigned nodes without touching receiver ownership."""
import sys,json,contextlib,io
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from qlib.common import read_json,write_json,utc_now
from ops import status,remote

def main():
    states={}
    for name,profile in (('primary',ROOT),('second',ROOT/'parallel/node2')):
        old_status,old_remote=status.ROOT,remote.ROOT
        try:
            status.ROOT=remote.ROOT=profile
            with contextlib.redirect_stdout(io.StringIO()):live=status.main()
            states[name]={'driver':live.get('driver'),'campaign':live.get('campaign'),'runs':live.get('runs'),'free_disk_bytes':live.get('free_disk_bytes'),'observed_at':live.get('observed_at')}
        finally:status.ROOT=old_status;remote.ROOT=old_remote
    states['authorized_seed_scope']=[42]
    states['assigned_runs']={'primary':['C_seed42','S_seed42','D_seed42'],'second':['SD_seed42']}
    states['old_primary_loaded_queue_is_superseded_by_boundary_amendment']=True
    states['time']=utc_now();write_json(ROOT/'reports/live_parallel_status.json',states)
    print(json.dumps(states,ensure_ascii=False));return states

if __name__=='__main__':main()
