"""One guarded receiver replacement; no training interruption, no verified-file loss."""
import sys,time,subprocess,shutil,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import psutil
from qlib.common import read_json,write_json,sha,utc_now
expected=read_json(ROOT/'reports/attendant_pid.json');pid=expected['pid'];assert pid==35684
assert not (ROOT/'reports/shutdown_receipt.json').exists()
helper=ROOT/'reports/attendant_persistent_receivers.py';assert helper.exists()
process=psutil.Process(pid)
assert Path(process.cwd()).resolve()==ROOT.resolve()
assert any(arg.replace('\\','/').endswith('ops/attendant.py') for arg in process.cmdline())
temporary=list((ROOT/'server_results').rglob('*.download'))
assert all(p.resolve().is_relative_to((ROOT/'server_results').resolve()) for p in temporary)
assert all(p.stat().st_size<10*1024**2 for p in temporary),'Let the current large immutable receiver finish first'
lock=ROOT/'reports/mirror.lock'
if lock.exists():assert read_json(lock)['pid']==pid
archive=ROOT/'reports/preparation_repairs/receiver_transport_20261005';archive.mkdir(parents=True,exist_ok=True)
for name in ('attendant_pid.json','attendant_state.json','transfer_progress.json','mirror.lock'):
    p=ROOT/'reports'/name
    if p.exists():shutil.copy2(p,archive/('original_'+name.replace('.lock','.json')))
before=read_json(ROOT/'reports/transfer_progress.json')
partial=[{'path':p.relative_to(ROOT).as_posix(),'bytes':p.stat().st_size} for p in temporary]
process.terminate();process.wait(timeout=15)
assert not process.is_running()
if lock.exists():
    assert read_json(lock)['pid']==pid
    lock.unlink() # Only a lock whose recorded owning receiver has exited.
flags=subprocess.CREATE_NO_WINDOW if sys.platform=='win32' else 0
with (ROOT/'reports/attendant_persistent.log').open('ab') as log:
    new=subprocess.Popen([sys.executable,'-u',str(helper)],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=log,creationflags=flags)
receipt={'stage':'ONE_RECEIVER_HANDOFF','time':utc_now(),'old_pid':pid,'pid':new.pid,
    'helper':'reports/attendant_persistent_receivers.py','helper_sha256':sha(helper),'original_transfer_progress':before,
    'small_unverified_partials_retained_for_original_SHA_checked_get':partial,'all_verified_files_retained':True,
    'frozen_source_and_training_unchanged':True,'max_file_receivers':2,
    'change':'Reuse two worker SSH/SFTP connections, lossless transport compression; compact duplicate polling stdout. Original mirror, atomic SHA, custody and exactly-once shutdown checks unchanged.'}
write_json(ROOT/'reports/receiver_transport_revision.json',receipt)
write_json(ROOT/'reports/attendant_pid.json',{'pid':new.pid,'time':utc_now(),'helper':receipt['helper']})
print(json.dumps(receipt,ensure_ascii=False))
