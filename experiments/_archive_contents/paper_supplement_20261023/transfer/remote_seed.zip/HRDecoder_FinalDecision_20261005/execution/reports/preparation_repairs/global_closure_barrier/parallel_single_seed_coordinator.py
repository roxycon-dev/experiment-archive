"""One canonical physical result copy; admit completed SD with authentic custody."""
import sys,time,json,os,shutil,zipfile,hashlib,shlex,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'reports'))
from qlib.common import read_json,write_json,sha,utc_now
from ops.remote import connect,command
from parallel_remote import connect_node2
PROFILE=ROOT/'parallel/node2';N2=PROFILE/'server_results';OUT=ROOT/'server_results';REMOTE='/root/autodl-tmp/HRDecoder_FinalDecision_20261005'

def link_endpoints():
    for u in (20000,32000):
        receipt=PROFILE/f'reports/storage_offload/SD_seed42_u{u}.json'
        if not receipt.exists():continue
        r=read_json(receipt);assert r['stage']=='LOCAL_SHA_VERIFIED_REMOTE_RELEASED' and r['server_instance']=='connect.nmb1.seetacloud.com:40920' and r['run']=='SD_seed42'
        for row in r['files']:
            src=(N2/row['path']).resolve();target=(OUT/row['path']).resolve()
            assert src.is_relative_to(N2/'runs/SD_seed42') and target.is_relative_to(OUT/'runs/SD_seed42')
            target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists():assert sha(target)==row['sha256']
            else:assert sha(src)==row['sha256'];os.link(src,target)

def import_complete():
    receipts=[PROFILE/f'reports/storage_offload/SD_seed42_u{u}.json' for u in (20000,32000)]
    assert all(p.exists() for p in receipts)
    link_endpoints()
    run=N2/'runs/SD_seed42';status=read_json(run/'status.json');assert status['stage']=='COMPLETE' and status['completed']==32000
    mapping={}
    for p in run.rglob('*'):
        if not p.is_file() or p.suffix in ('.download','.tmp','.pth','.npz') or p.name=='status.json':continue
        rel=p.relative_to(N2).as_posix();dst=OUT/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dst);mapping[rel]=p
    for p in (N2/'reports').rglob('*'):
        if not p.is_file() or p.suffix in ('.download','.tmp','.pth','.npz','.zip') or p.name.startswith('stream_'):continue
        rel='reports/parallel_node2_evidence/'+p.relative_to(N2/'reports').as_posix();mapping[rel]=p
    for p in receipts:
        rel='reports/storage_offload/'+p.name;mapping[rel]=p
        (ROOT/'reports/storage_offload').mkdir(exist_ok=True);shutil.copy2(p,ROOT/rel)
    for rel,p in mapping.items():
        dst=OUT/rel;dst.parent.mkdir(parents=True,exist_ok=True)
        if dst.resolve()!=p.resolve():shutil.copy2(p,dst)
    package=ROOT/'reports/parallel_SD_import_meta.zip'
    with zipfile.ZipFile(package,'w',zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        for rel,p in mapping.items():z.write(p,rel)
    manifest={rel:sha(p) for rel,p in mapping.items()};pack_sha=sha(package)
    with connect() as c:
        with c.open_sftp() as f:f.put(str(package),REMOTE+'/parallel_SD_import_meta.zip')
        code="""
from pathlib import Path
import zipfile,hashlib,json,os
r=Path(%r);p=r/'parallel_SD_import_meta.zip'
assert hashlib.sha256(p.read_bytes()).hexdigest()==%r
assert not (r/'runs/SD_seed42/status.json').exists(),'Unexpected primary SD owner: diagnose before import'
with zipfile.ZipFile(p) as z:
 for n in z.namelist():assert (r/n).resolve().is_relative_to(r)
 z.extractall(r)
for rel,h in %r.items():assert hashlib.sha256((r/rel).read_bytes()).hexdigest()==h
status=%r
(r/'runs/SD_seed42/status.json').write_text(json.dumps(status))
receipt={'stage':'FULL_RUN_IMPORT_VERIFIED','run':'SD_seed42','source_instance':'connect.nmb1.seetacloud.com:40920','destination_instance':'connect.nmb2.seetacloud.com:36431','all_binaries_in_authentic_foreign_SHA_custody':True,'metadata_sha256':%r,'no_duplicate_training':True}
(r/'reports/parallel_import_receipt_SD_seed42.json').write_text(json.dumps(receipt));p.unlink();print(json.dumps(receipt))
"""%(REMOTE,pack_sha,manifest,status,manifest)
        result=json.loads(command(c,"/root/miniconda3/envs/hrdecoder_q_20261003/bin/python - <<'PY'\n"+code+"\nPY",120))
    package.unlink();shutil.copy2(run/'status.json',OUT/'runs/SD_seed42/status.json');write_json(ROOT/'reports/parallel_import_receipt_SD_seed42.json',result)

def finish_node2_global():
    with connect() as a,connect_node2() as b:
        with a.open_sftp() as af,b.open_sftp() as bf:
            for name in ['最终实验报告.md','decision.json','final_test.json','final_selection_lock.json']:
                source=REMOTE+'/reports/'+name
                try:
                    with af.file(source,'rb') as f:data=f.read()
                except FileNotFoundError:continue
                with bf.file(REMOTE+'/reports/global_primary_'+name,'wb') as f:f.write(data)
            with af.file(REMOTE+'/reports/最终实验报告.md','rb') as f:report=f.read()
            with bf.file(REMOTE+'/reports/最终实验报告.md','wb') as f:f.write(report)
        code="""
from pathlib import Path
import json
r=Path(%r);assert not (r/'reports/driver.lock').exists()
assert json.loads((r/'runs/SD_seed42/status.json').read_text())['stage']=='COMPLETE'
assert (r/'reports/最终实验报告.md').exists()
state={'stage':'READY_FOR_LOCAL_BACKUP_AND_SHUTDOWN','queue':[['SD',42]],'global_single_seed_scope_complete':True,'single_seed_exploratory':True,'shutdown_owned_by_node2_profile_attendant':True}
(r/'reports/campaign.json').write_text(json.dumps(state));(r/'reports/driver_state.json').write_text(json.dumps(state));print('NODE2_GLOBAL_CLOSURE_READY')
"""%REMOTE
        print(command(b,"/root/miniconda3/envs/hrdecoder_q_20261003/bin/python - <<'PY'\n"+code+"\nPY",30),flush=True)

def main():
    admitted=(ROOT/'reports/parallel_import_receipt_SD_seed42.json').exists()
    while True:
        try:
            link_endpoints()
            if not admitted:
                p=N2/'runs/SD_seed42/status.json';s=read_json(p) if p.exists() else {}
                if s.get('stage')=='COMPLETE' and all((PROFILE/f'reports/storage_offload/SD_seed42_u{u}.json').exists() for u in (20000,32000)):
                    import_complete();admitted=True
                elif s.get('stage')=='SCIENTIFIC_STOP_INCOMPLETE':
                    write_json(ROOT/'reports/parallel_coordinator_state.json',{'stage':'INCOMPLETE_STATE_BACKUP_AND_IMPORT_REVIEW_REQUIRED','run':'SD_seed42','automatic_training_restart':False,'time':utc_now()});time.sleep(30);continue
            if admitted:
                from ops.status import main as status
                import contextlib,io
                with contextlib.redirect_stdout(io.StringIO()):live=status()
                if (live.get('driver') or {}).get('stage')=='READY_FOR_LOCAL_BACKUP_AND_SHUTDOWN':
                    finish_node2_global();write_json(ROOT/'reports/parallel_coordinator_state.json',{'stage':'GLOBAL_READY_BOTH_ATTENDANTS_OWN_RECOVERY_AND_SHUTDOWN','time':utc_now()});return
            write_json(ROOT/'reports/parallel_coordinator_state.json',{'stage':'SD_IMPORTED_WAITING_PRIMARY' if admitted else 'WAITING_SD_COMPLETE_SHA_CUSTODY','time':utc_now()})
        except BaseException as e:
            write_json(ROOT/'reports/parallel_coordinator_state.json',{'stage':'TECHNICAL_PAUSE_REVIEW','reason':str(e),'automatic_training_restart':False,'time':utc_now()});traceback.print_exc()
            return
        time.sleep(30)

if __name__=='__main__':main()
