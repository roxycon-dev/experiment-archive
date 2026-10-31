"""Release only immutable endpoints after primary receiver SHA verification."""
import sys,json,shlex
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from qlib.common import read_json,write_json,sha,utc_now,digest
from ops.remote import connect,command

def release_verified():
    runtime=read_json(ROOT/'configs/runtime.local.json');remote=runtime['server']['remote_root'];output=ROOT/'server_results'
    receipts=ROOT/'reports/storage_offload';receipts.mkdir(parents=True,exist_ok=True)
    applied=[]
    for path in sorted((output/'reports/storage_requests').glob('*.json')):
        request=read_json(path);assert request['request_sha256']==digest({k:v for k,v in request.items() if k!='request_sha256'})
        run=request['run'];update=request['update']
        is_parent=run=='PARENT_F12_seed2024' and update==12000
        assert is_parent or (run in {f'{a}_seed{s}' for a in ('C','S','D','SD') for s in (42,2024)} and update in (20000,32000))
        receiptfile=receipts/path.name
        if receiptfile.exists():
            if update==32000:remove_completed_duplicate(read_json(receiptfile),remote,runtime)
            continue
        files=request['files'];assert len(files)==(76 if is_parent else 77)
        for row in files:
            target=(output/row['path']).resolve();assert target.is_relative_to((output/'runs'/run).resolve())
            assert (not is_parent and row['path']==f'runs/{run}/checkpoint_u{update}.pth') or (target.parent==output.resolve()/f'runs/{run}/val_u{update}' and target.suffix=='.npz')
        if not all((output/r['path']).exists() and sha(output/r['path'])==r['sha256'] for r in files):continue
        result={'stage':'LOCAL_SHA_VERIFIED_REMOTE_RELEASED','files':files,'run':run,'update':update,
                'request_sha256':request['request_sha256'],'server_instance':runtime['server']['instance_id'],
                'local_directory':str(output),'time':utc_now(),'preserved_last_optimizer_rng_and_all_metadata':True}
        code="""
from pathlib import Path
import hashlib,json,os
r=Path(%r);request=json.loads(%r);result=json.loads(%r)
original=json.loads((r/'reports/storage_requests'/%r).read_text());assert original==request
def h(p):
 d=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):d.update(b)
 return d.hexdigest()
for row in result['files']:
 p=(r/row['path']).resolve();assert (r/'runs'/result['run']).resolve() in p.parents
 assert p.name!='last.pth'
 if p.exists():assert p.stat().st_size==row['bytes'] and h(p)==row['sha256'],str(p)
receipt=r/'reports/storage_offload'/%r;receipt.parent.mkdir(exist_ok=True)
# Persist the verified custody receipt before removing anything. A partial
# unlink can be retried from the same local SHA proof after concrete diagnosis.
tmp=receipt.with_suffix('.tmp');tmp.write_text(json.dumps(result));os.replace(tmp,receipt)
for row in result['files']:
 p=r/row['path']
 if p.exists():p.unlink()
print(json.dumps({'stage':result['stage'],'released_bytes':sum(x['bytes'] for x in result['files'])}))
"""%(remote,json.dumps(request),json.dumps(result),path.name,path.name)
        with connect() as c:confirmation=json.loads(command(c,shlex.quote(runtime['server_python'])+" - <<'PY'\n"+code+"\nPY",300))
        result.update(confirmation);write_json(receiptfile,result);applied.append(result)
        if update==32000:remove_completed_duplicate(result,remote,runtime)
    return applied

def remove_completed_duplicate(receipt,remote,runtime):
    if receipt.get('redundant_completed_last_checked'):return
    checkpoint=next(x for x in receipt['files'] if x['path'].endswith('.pth'))
    assert sha(ROOT/'server_results'/checkpoint['path'])==checkpoint['sha256']
    code="""
from pathlib import Path
import json,hashlib
r=Path(%r);run=r/'runs'/%r;state=json.loads((run/'status.json').read_text());p=run/'last.pth'
complete=state['stage']=='COMPLETE' and state['completed']==32000
if complete and p.exists():
 d=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):d.update(b)
 assert d.hexdigest()==%r,'Completed last differs from its saved endpoint: preserve and diagnose'
 p.unlink()
print(json.dumps({'complete_duplicate_checked':complete}))
"""%(remote,receipt['run'],checkpoint['sha256'])
    with connect() as c:result=json.loads(command(c,shlex.quote(runtime['server_python'])+" - <<'PY'\n"+code+"\nPY",120))
    if result['complete_duplicate_checked']:
        receipt['redundant_completed_last_checked']=True
        write_json(ROOT/f'reports/storage_offload/{receipt["run"]}_u{receipt["update"]}.json',receipt)

def restore_selected():
    output=ROOT/'server_results';requestfile=output/'reports/selected_restore_request.json'
    if not requestfile.exists():return
    request=read_json(requestfile);lock=read_json(output/'reports/final_selection_lock.json')
    assert lock['lock_sha256']==request['lock_sha256'] and read_json(output/'reports/decision.json')['test_authorized']
    if (output/'reports/selected_restore_complete.json').exists():return
    runtime=read_json(ROOT/'configs/runtime.local.json');remote=runtime['server']['remote_root']
    for row in request['files']:
        assert row in list(lock['checkpoints'].values())
        p=(output/row['path']).resolve();assert p.is_relative_to((output/'runs').resolve()) and sha(p)==row['sha256']
        check="from pathlib import Path;import hashlib;p=Path(%r);print(hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else '')"%(remote+'/'+row['path'])
        with connect() as c:
            existing=command(c,shlex.quote(runtime['server_python'])+" - <<'PY'\n"+check+"\nPY",60).strip()
            if existing==row['sha256']:continue
            assert not existing,'Conflicting selected checkpoint: diagnose before overwriting'
            with c.open_sftp() as s:
                temporary=remote+'/'+row['path']+'.restore';s.put(str(p),temporary)
            verify="from pathlib import Path;import hashlib,os;p=Path(%r);assert hashlib.sha256(p.read_bytes()).hexdigest()==%r;os.replace(p,%r)"%(temporary,row['sha256'],remote+'/'+row['path'])
            command(c,shlex.quote(runtime['server_python'])+" - <<'PY'\n"+verify+"\nPY",120)
    write_json(ROOT/'reports/selected_restore_dispatch.json',{'stage':'ONLY_LOCKED_WEIGHTS_SHA_RESTORED','lock_sha256':request['lock_sha256'],'time':utc_now()})
