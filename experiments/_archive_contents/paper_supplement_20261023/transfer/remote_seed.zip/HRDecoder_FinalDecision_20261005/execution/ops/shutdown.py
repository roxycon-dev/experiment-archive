"""Official platform shutdown after verified local full-state backup, exactly once."""
import json
from pathlib import Path
import shlex
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from qlib.common import read_json,sha,utc_now,write_json
from ops.remote import connect,command


def main():
    receipt_file=ROOT/'reports/shutdown_receipt.json'
    assert not receipt_file.exists(),'Already dispatched: inspect platform, do not blindly repeat'
    backup=read_json(ROOT/'reports/final_backup_receipt.json')
    assert backup['stage']=='FULL_BACKUP_VERIFIED'
    assert backup['campaign']['stage']=='READY_FOR_LOCAL_BACKUP_AND_SHUTDOWN'
    for item in backup['files']:
        assert sha(ROOT/'server_results'/item['path'])==item['sha256']
    runtime=read_json(ROOT/'configs/runtime.local.json');remote=runtime['server']['remote_root']
    code='''
from pathlib import Path
import hashlib,json,subprocess,sys,time
r=Path(%r)
state=json.loads((r/'reports/campaign.json').read_text())
assert state['stage']=='READY_FOR_LOCAL_BACKUP_AND_SHUTDOWN'
assert not (r/'reports/campaign.lock').exists() and not (r/'reports/driver.lock').exists()
control=Path('/usr/bin/shutdown')
assert hashlib.sha256(control.read_bytes()).hexdigest()=='0358e83eeeaf542aa98f64ba9e339c91df46f1e025892d52dba159f4fb1cf027','Platform control changed; review before executing'
request=r/'reports/shutdown_request.json'
assert not request.exists(),'Already requested'
request.write_text(json.dumps({'stage':'SCHEDULED_AFTER_VERIFIED_LOCAL_BACKUP','backup_receipt_sha256':%r,'platform_OFF_verified':False}))
worker="import time,subprocess;time.sleep(5);subprocess.run(['/bin/bash','/usr/bin/shutdown'],check=True)"
with (r/'reports/shutdown_worker.private.log').open('wb') as log:
 p=subprocess.Popen([sys.executable,'-c',worker],stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
print(json.dumps({'stage':'SHUTDOWN_DISPATCHED','worker_pid':p.pid,'platform_OFF_verified':False}))
'''%(remote,sha(ROOT/'reports/final_backup_receipt.json'))
    with connect() as client:
        with client.open_sftp() as sftp:sftp.put(str(ROOT/'reports/final_backup_receipt.json'),remote+'/reports/local_backup_receipt.json')
        result=json.loads(command(client,shlex.quote(runtime['server_python'])+" - <<'PY'\n"+code+"\nPY",timeout=30))
    result.update(dispatched_at=utc_now(),backup_SHA_verified=True,
                  note='Verify platformOFF through available authenticated platform UI/API or direct user observation. SSH loss is not proof.')
    write_json(receipt_file,result);print(result)


if __name__=='__main__':main()
