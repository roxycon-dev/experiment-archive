"""Stop only the identified zero-progress local receiver; preserve all evidence."""
import sys,time,shutil,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import psutil
from qlib.common import read_json,write_json,utc_now,sha

expected=read_json(ROOT/'reports/attendant_pid.json');pid=expected['pid']
assert pid==30340
assert not (ROOT/'reports/shutdown_receipt.json').exists()
process=psutil.Process(pid)
assert Path(process.cwd()).resolve()==ROOT.resolve()
assert any(arg.replace('\\','/').endswith('reports/attendant_persistent_receivers.py') for arg in process.cmdline())
lock=ROOT/'reports/mirror.lock';assert read_json(lock)['pid']==pid
partial=ROOT/'server_results/runs/C_seed42/checkpoint_u20000.pth.download'
info=partial.stat();assert info.st_size==0 and time.time()-info.st_mtime>3600
progress=read_json(ROOT/'reports/transfer_progress.json')
from datetime import datetime
assert time.time()-datetime.fromisoformat(progress['updated_at']).timestamp()>1800
archive=ROOT/'reports/preparation_repairs/receiver_large_file_20261005';archive.mkdir(parents=True,exist_ok=False)
originals=[]
for name in ['attendant_pid.json','attendant_state.json','transfer_progress.json','persistent_receiver_state.json','mirror.lock','attendant_persistent.log','attendant_persistent_receivers.py']:
    p=ROOT/'reports'/name
    if p.exists():
        target=archive/('original_'+name.replace('.lock','.json'));shutil.copy2(p,target)
        originals.append({'path':target.relative_to(ROOT).as_posix(),'sha256':sha(target)})
receipt={'stage':'STALLED_RECEIVER_STOPPED_FOR_DIAGNOSIS','time':utc_now(),'pid':pid,
         'concrete_failure':'checkpoint_u20000.pth.download remained zero bytes for over one hour; last verified progress older than30min',
         'partial_bytes':info.st_size,'partial_mtime_epoch':info.st_mtime,'previous_progress':progress,
         'originals':originals,'all_verified_files_retained':True,'training_not_interrupted':True}
process.terminate();process.wait(timeout=15);assert not process.is_running()
assert read_json(lock)['pid']==pid;lock.unlink()
write_json(ROOT/'reports/receiver_large_file_diagnosis.json',receipt)
print(json.dumps(receipt,ensure_ascii=False))
