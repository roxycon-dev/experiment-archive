"""Idle-boundary receiver handoff for audited two-node final custody union."""
import sys,time,subprocess,shutil,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import psutil
from qlib.common import read_json,write_json,utc_now,sha
expected=read_json(ROOT/'reports/attendant_pid.json');pid=expected['pid'];assert pid==30340
assert not (ROOT/'reports/shutdown_receipt.json').exists()
process=psutil.Process(pid);assert Path(process.cwd()).resolve()==ROOT.resolve()
assert any(a.replace('\\','/').endswith('reports/attendant_persistent_receivers.py') for a in process.cmdline())
lock=ROOT/'reports/mirror.lock'
while lock.exists():time.sleep(.1)
assert read_json(ROOT/'reports/attendant_state.json')['stage']=='WAITING'
archive=ROOT/'reports/preparation_repairs/parallel_custody_handoff';archive.mkdir(parents=True,exist_ok=False)
for name in ['attendant_pid.json','attendant_state.json','persistent_receiver_state.json','attendant_persistent_receivers.py']:
 p=ROOT/'reports'/name
 if p.exists():shutil.copy2(p,archive/('original_'+name))
assert not lock.exists();process.terminate();process.wait(timeout=15)
assert not process.is_running() and not lock.exists()
helper=ROOT/'reports/primary_parallel_attendant.py'
with (ROOT/'reports/primary_parallel_attendant.log').open('ab') as log:
 new=subprocess.Popen([sys.executable,'-u',str(helper)],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=log,creationflags=subprocess.CREATE_NO_WINDOW)
receipt={'stage':'IDLE_RECEIVER_HANDOFF','old_pid':pid,'pid':new.pid,'time':utc_now(),'helper':'reports/primary_parallel_attendant.py','helper_sha256':sha(helper),'all_verified_files_preserved':True,'training_unchanged':True,'reason':'Explicitly authorized second node and single-seed scope; final SHA union accepts only assigned SD42 source custody.'}
write_json(ROOT/'reports/primary_parallel_receiver_handoff.json',receipt);write_json(ROOT/'reports/attendant_pid.json',{'pid':new.pid,'time':utc_now(),'helper':receipt['helper']});print(json.dumps(receipt))
