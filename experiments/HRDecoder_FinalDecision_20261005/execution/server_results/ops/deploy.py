"""Upload only to newly assigned running server, verify all files; no allocation."""
import sys,shlex,json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from qlib.common import read_json,write_json,sha,utc_now
from ops.remote import connect,command

def main(launch=False,source_only=False):
    local=read_json(ROOT/'configs/runtime.local.json');remote=local['server']['remote_root']
    assert remote=='/root/autodl-tmp/HRDecoder_FinalDecision_20261005'
    assert read_json(ROOT/'reports/preparation_status.json')['stage']=='LOCAL_READY_SERVER_ASSIGNED_GPU_GATE_PENDING'
    if not source_only:assert read_json(ROOT/'reports/required_assets_receipt.json')['stage']=='ALL_REQUIRED_ASSETS_SHA_VERIFIED'
    plan=read_json(ROOT/'reports/storage_plan.json');assert plan['initial_local_free_bytes']>=100*1024**3 and plan['user_authorized_local_cold_archive']
    receipt=read_json(ROOT/'reports/package_identity.json');package=ROOT/'reports/HRDecoder_20261005_prepared.zip'
    assert sha(package)==receipt['sha256']
    def py(code,timeout=60):return shlex.quote(local['server_python'])+" - <<'PY'\n"+code+"\nPY"
    with connect() as client:
        # The ONLY machine check here is the experiment's required data-disk reserve.
        code="from pathlib import Path;import shutil,json;r=Path(%r);r.mkdir(parents=True,exist_ok=True);assert not (r/'reports/deployment.json').exists(),'Existing deployment: explicit recovery required';p=r/'reports/source_deployment.json';print(json.dumps({'free_disk_bytes':shutil.disk_usage(r).free,'source_deployment':json.loads(p.read_text()) if p.exists() else None}))"%remote
        space=json.loads(command(client,py(code)));write_json(ROOT/'reports/server_space.json',space)
        assert space['free_disk_bytes']>=12*1024**3,'Bounded active storage reserve insufficient; diagnose relevant assets before launch'
        staged=space['source_deployment']
        if staged:assert staged['package_sha256']==receipt['sha256'],'Staged source changed: diagnose before any gate or launch'
        else:
            with client.open_sftp() as sftp:sftp.put(str(package),remote+'/prepared.zip')
        code="""
from pathlib import Path
import zipfile,hashlib,json
r=Path(%r)
assert hashlib.sha256((r/'prepared.zip').read_bytes()).hexdigest()==%r
assert not (r/'reports/driver.lock').exists() and not (r/'reports/campaign.lock').exists()
with zipfile.ZipFile(r/'prepared.zip') as z:
 for name in z.namelist():assert (r/name).resolve().is_relative_to(r)
 z.extractall(r)
print('SOURCE_PACKAGE_SHA_VERIFIED')
"""%(remote,receipt['sha256'])
        if not staged:print(command(client,py(code)))
        # Runtime assignment is mutable only BEFORE GPU acceptance and run identity.
        if not staged:
            with client.open_sftp() as sftp:sftp.put(str(ROOT/'configs/runtime.server.json'),remote+'/configs/runtime.server.json')
            staged={'stage':'SOURCE_STAGED','package_sha256':receipt['sha256'],'time':utc_now()}
            write_json(ROOT/'reports/source_deployment.json',staged)
            with client.open_sftp() as sftp:sftp.put(str(ROOT/'reports/source_deployment.json'),remote+'/reports/source_deployment.json')
        if source_only:print(staged);return staged
        # Dedicated bounded upload already restored all896 train files and the
        # missing immutable models. Do not upload duplicate copies again.
        verify="""
import json,hashlib
from pathlib import Path
r=Path(%r);m=json.loads((r/'reports/TJDR_manifest.json').read_text())
for row in m['samples']:
 for directory,key in (('image','image_sha256'),('annotation','label_sha256')):
  p=r/'inputs/TJDR/train'/directory/(row['sid']+'.png')
  assert hashlib.sha256(p.read_bytes()).hexdigest()==row[key],str(p)
print('OFFICIAL_TRAIN_ALL896_FILES_SHA_VERIFIED')
"""%remote
        print(command(client,py(verify),300))
        with client.open_sftp() as sftp:
            for filename in ('required_assets_receipt.json','required_assets_inventory.json'):
                sftp.put(str(ROOT/'reports'/filename),remote+'/reports/'+filename)
        result={'stage':'DEPLOYED_SHA_VERIFIED','package_sha256':receipt['sha256'],'time':utc_now()}
        write_json(ROOT/'reports/deployment.json',result)
        with client.open_sftp() as sftp:sftp.put(str(ROOT/'reports/deployment.json'),remote+'/reports/deployment.json')
        if launch:
            code="""
import subprocess,json
from pathlib import Path
r=Path(%r)
assert not (r/'reports/driver_state.json').exists(),'Never duplicate driver'
with (r/'reports/server_driver.log').open('ab') as log:
 p=subprocess.Popen([%r,'-u','server_driver.py'],cwd=r,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
print(json.dumps({'pid':p.pid,'stage':'DRIVER_LAUNCHED'}))
"""%(remote,local['server_python'])
            result.update(json.loads(command(client,py(code),30)));write_json(ROOT/'reports/deployment.json',result)
            flags=subprocess.CREATE_NO_WINDOW if sys.platform=='win32' else 0
            with (ROOT/'reports/attendant.log').open('ab') as log:
                p=subprocess.Popen([sys.executable,'-u','ops/attendant.py'],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=log,creationflags=flags)
            write_json(ROOT/'reports/attendant_pid.json',{'pid':p.pid,'time':utc_now()})
        command(client,py("from pathlib import Path;p=Path(%r);p.unlink() if p.exists() else None"%(remote+'/prepared.zip')),30)
        package.unlink() # All source and data are verified; no redundant D-drive archive.
    print(result)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--launch',action='store_true');p.add_argument('--source-only',action='store_true');a=p.parse_args();main(a.launch,a.source_only)
