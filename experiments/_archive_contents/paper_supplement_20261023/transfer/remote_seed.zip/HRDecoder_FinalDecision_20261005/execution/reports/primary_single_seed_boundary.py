"""Retire the original controller AFTER D42 completes; never run seed2024."""
import sys,os,time,signal,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from qlib.common import read_json,write_json,utc_now,sha
assert read_json(ROOT/'reports/single_seed_parallel_amendment.json')['seeds']==[42]
write_json(ROOT/'reports/single_seed_boundary_state.json',{'stage':'ARMED_AFTER_D42_BOUNDARY','pid':os.getpid(),'time':utc_now()})
while True:
    p=ROOT/'runs/D_seed42/status.json'
    state=read_json(p) if p.exists() else {}
    if state.get('stage') not in ('COMPLETE','SCIENTIFIC_STOP_INCOMPLETE'):
        time.sleep(.05);continue
    lock=ROOT/'reports/driver.lock'
    if not lock.exists():time.sleep(.05);continue
    owner=read_json(lock)['pid'];args=Path(f'/proc/{owner}/cmdline').read_bytes()
    assert b'HRDecoder_FinalDecision_20261005' in Path(f'/proc/{owner}/cwd').resolve().as_posix().encode()
    assert os.getpgid(owner)==owner
    os.killpg(owner,signal.SIGSTOP)
    # A complete run is sealed before the campaign can initialize the next arm.
    assert read_json(p)['stage'] in ('COMPLETE','SCIENTIFIC_STOP_INCOMPLETE')
    assert not (ROOT/'runs/PARENT_F12_seed2024/identity.json').exists()
    sd=ROOT/'runs/SD_seed42/status.json'
    assert not sd.exists() or read_json(sd)['stage'] in ('COMPLETE','SCIENTIFIC_STOP_INCOMPLETE'), 'Unexpected duplicate SD; preserve stopped controller for diagnosis'
    archive=ROOT/'reports/primary_controller_boundary_revision';archive.mkdir(exist_ok=True)
    for name in ['driver_state.json','campaign.json','driver.lock','campaign.lock']:
        q=ROOT/'reports'/name
        if q.exists():shutil.copy2(q,archive/('original_'+name.replace('.lock','.json')))
    os.killpg(owner,signal.SIGKILL)
    for _ in range(100):
        if not Path(f'/proc/{owner}').exists():break
        time.sleep(.05)
    for name in ['driver.lock','campaign.lock']:
        q=ROOT/'reports'/name
        if q.exists():assert read_json(q)['pid']==owner;q.unlink()
    write_json(ROOT/'reports/single_seed_boundary_state.json',{'stage':'COMPLETED_RUN_BOUNDARY_HANDOFF','old_pid':owner,'boundary':'D_seed42 sealed terminal status','lost_updates':0,'source_unchanged':True,'time':utc_now()})
    break
write_json(ROOT/'reports/driver_state.json',{'stage':'WAITING_FOR_PARALLEL_SD_IMPORT','pid':os.getpid(),'time':utc_now(),'seed_scope':[42]})
write_json(ROOT/'reports/campaign.json',{'stage':'WAITING_FOR_PARALLEL_SD_IMPORT','queue':[['C',42],['S',42],['D',42],['SD',42]],'cancelled_seed2024':True,'time':utc_now()})
while not (ROOT/'reports/parallel_import_receipt_SD_seed42.json').exists():time.sleep(5)
assert read_json(ROOT/'reports/parallel_import_receipt_SD_seed42.json')['stage']=='FULL_RUN_IMPORT_VERIFIED'
from single_seed_finalizer import main
main()
