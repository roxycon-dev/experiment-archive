"""One-shot bounded launch queue; no training starts until accepted asset SHA receipt exists."""
import json
import pathlib
import sys
import time
from datetime import datetime,timezone

ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from ops.remote import connect

def main():
    output=ROOT/'reports/qmix_launch_queue_receipt.json'
    if output.exists():raise RuntimeError('A launch request already has a receipt; inspect instead of repeating')
    for _ in range(300):
        p=ROOT/'reports/qmix_deployment_progress.json'
        state=json.loads(p.read_text()) if p.exists() else {}
        if state.get('stage')=='REMOTE_ASSETS_READY':break
        errors=ROOT/'reports/qmix_deployment_error.log'
        if errors.exists() and errors.stat().st_size:raise RuntimeError('Asset deployment reported failure; do not launch')
        time.sleep(3)
    else:raise TimeoutError('Bounded deployment wait expired; no automatic retry')
    command="/root/miniconda3/envs/hrdecoder_q_20261003/bin/python - <<'PY'\n"+r'''
import json,pathlib,subprocess,os
root=pathlib.Path('/root/autodl-tmp/HRDecoder_Q_20261003')
os.chdir(root)
assert json.loads((root/'reports/qmix_remote_asset_receipt.json').read_text())['status']=='ALL_REMOTE_ASSET_SHA_VERIFIED'
assert not (root/'runs/Q_MIX_seed42/last.pth').exists()
for name in ('formal.pid','qmix_launcher.pid'):
 p=root/'reports'/name
 if p.exists():
  proc=pathlib.Path('/proc')/p.read_text().strip()/'cmdline'
  if proc.exists():
   value=proc.read_bytes()
   assert b'train_run.py' not in value and b'start_qmix_remote.py' not in value
with (root/'reports/qmix_launcher.log').open('wb') as out:
 p=subprocess.Popen(['/root/miniconda3/envs/hrdecoder_q_20261003/bin/python','-u','reports/start_qmix_remote.py','--arm','Q_MIX','--seed','42','--endpoint','16000'],
   stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT,cwd=root,start_new_session=True,
   env={**os.environ,'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4'})
(root/'reports/qmix_launcher.pid').write_text(str(p.pid)+'\n')
print(json.dumps({'launcher_pid':p.pid,'stage':'AUTHORIZED_Q_MIX_LAUNCHER_QUEUED','arm':'Q_MIX','seed':42,'endpoint':16000}))
PY'''
    with connect() as c:
        _,out,err=c.exec_command(command,timeout=60)
        data=out.read().decode();error=err.read().decode();code=out.channel.recv_exit_status()
        if code:raise RuntimeError(error)
    receipt=json.loads(data);receipt['created_at']=datetime.now(timezone.utc).isoformat()
    output.write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print(json.dumps(receipt),flush=True)

if __name__=='__main__':main()
