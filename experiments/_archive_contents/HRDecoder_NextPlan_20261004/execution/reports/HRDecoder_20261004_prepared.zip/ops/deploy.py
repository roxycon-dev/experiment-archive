"""Run only after the user starts this assigned server. No paid allocation API."""
import argparse
import json
from pathlib import Path
import shlex
import sys
import subprocess
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from qlib.common import read_json,sha,utc_now,write_json
from ops.remote import connect,command


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--launch',action='store_true');args=ap.parse_args()
    local=read_json(ROOT/'configs/runtime.local.json');remote=local['server']['remote_root']
    assert remote=='/root/autodl-tmp/HRDecoder_Finetune_20261004'
    package=ROOT/'reports/HRDecoder_20261004_prepared.zip';receipt=read_json(ROOT/'reports/package_identity.json')
    assert sha(package)==receipt['sha256']
    py=local['server_python']
    with connect() as client:
        # Free the one explicitly identified unrelated cache BEFORE writing an upload
        # to a full data disk. Run the same reviewed cleanup source in memory.
        space_code=(ROOT/'ops/prepare_space.py').read_text(encoding='utf-8')
        space_code=space_code.replace("ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))",
            "ROOT=Path("+repr(remote)+");sys.path.insert(0,'/root/autodl-tmp/HRDecoder_Q_20261003')")
        print(command(client,shlex.quote(py)+" - <<'PY'\n"+space_code+"\nPY"))
        command(client,'mkdir -p '+shlex.quote(remote))
        with client.open_sftp() as sftp:sftp.put(str(package),remote+'/prepared.zip')
        script='''
from pathlib import Path
import hashlib,zipfile,json,os
r=Path(%r)
assert hashlib.sha256((r/'prepared.zip').read_bytes()).hexdigest()==%r
assert not (r/'reports/driver.lock').exists() and not (r/'reports/campaign.lock').exists(),'Experiment already active'
assert not (r/'runs').exists(),'Existing results: use explicit recovery, never redeploy over them'
with zipfile.ZipFile(r/'prepared.zip') as z:
 for name in z.namelist():
  assert (r/name).resolve().is_relative_to(r)
 z.extractall(r)
print('PACKAGE_EXTRACTED_SHA_VERIFIED')
'''%(remote,receipt['sha256'])
        print(command(client,shlex.quote(py)+" - <<'PY'\n"+script+"\nPY"))
        if args.launch:
            launch='''
import subprocess,json
from pathlib import Path
r=Path(%r)
with (r/'reports/server_driver.log').open('ab') as log:
 p=subprocess.Popen([%r,'-u','server_driver.py'],cwd=r,stdout=log,stderr=log,stdin=subprocess.DEVNULL,start_new_session=True)
(r/'reports/driver.pid').write_text(str(p.pid))
print(json.dumps({'stage':'LAUNCHED','pid':p.pid}))
'''%(remote,py)
            result=json.loads(command(client,shlex.quote(py)+" - <<'PY'\n"+launch+"\nPY"))
        else:result={'stage':'DEPLOYED_NOT_LAUNCHED'}
    result.update(package_sha256=receipt['sha256'],time=utc_now(),server=local['server']['host'])
    write_json(ROOT/'reports/deployment.json',result);print(result)
    if args.launch:
        # Windows helper stays hidden; it waits only for terminal job state and then
        # backs up/offlines the server. Agent monitoring remains every45minutes.
        flags=subprocess.CREATE_NO_WINDOW if sys.platform=='win32' else 0
        with (ROOT/'reports/attendant.log').open('ab') as log:
            child=subprocess.Popen([sys.executable,'-u',str(ROOT/'ops/attendant.py')],cwd=ROOT,
                stdin=subprocess.DEVNULL,stdout=log,stderr=log,creationflags=flags)
        write_json(ROOT/'reports/attendant_pid.json',{'pid':child.pid,'time':utc_now()})


if __name__=='__main__':main()
