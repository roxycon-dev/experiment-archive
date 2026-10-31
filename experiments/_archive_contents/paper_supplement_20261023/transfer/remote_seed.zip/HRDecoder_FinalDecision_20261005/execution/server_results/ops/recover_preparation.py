"""Explicit diagnosed pre-training recovery; archive all failed source/gate evidence."""
import sys,json,shlex,zipfile,shutil,argparse,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from qlib.common import read_json,write_json,sha,utc_now
from ops.remote import connect,command

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',default='device_call_20261005');ap.add_argument('--failed-pid',type=int,default=5899);ap.add_argument('--reason',default='early GPU test and engine omitted required device argument of registered forward_rng; pass current x.device. No scientific setting or seed formula changes. Also include ops/tests in full source backup inventory.');a=ap.parse_args()
    assert re.fullmatch('[a-z0-9_]+',a.tag) and len(a.reason)>20
    rt=read_json(ROOT/'configs/runtime.local.json');remote=rt['server']['remote_root'];archive=ROOT/'reports/preparation_repairs'/a.tag
    archive.mkdir(parents=True,exist_ok=False)
    package=ROOT/'reports/HRDecoder_20261005_prepared.zip'
    with zipfile.ZipFile(package) as z:
        for rel in ('engine.py','tests/gpu_acceptance.py','ops/mirror.py','reports/source_identity.json'):
            dest=archive/rel;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(z.read(rel))
    old=read_json(archive/'reports/source_identity.json')
    result={'time':utc_now(),'old_source_sha256':old['sha256'],'diagnosis':a.reason,'formal_training_started':False,'failed_early_pid':a.failed_pid,'preserve_original_logs':True,'archive':str(archive)}
    code="""
from pathlib import Path
import json,shutil
r=Path(%r);assert not (r/'reports/driver_state.json').exists();assert not (r/'reports/campaign.json').exists();assert not any((r/'runs').glob('*/status.json'))
old=json.loads((r/'reports/source_identity.json').read_text());assert old['sha256']==%r
pid=json.loads((r/'reports/early_gpu_gate_pid.json').read_text());assert pid['pid']==%r and not Path('/proc/'+str(pid['pid'])).exists()
archive=r/'reports/preparation_repairs'/%r;archive.mkdir(parents=True,exist_ok=False)
for rel in ('engine.py','tests/gpu_acceptance.py','ops/mirror.py','reports/source_identity.json','reports/early_gpu_gate_pid.json','reports/early_gpu_gates.json','reports/early_gpu_gates.log','reports/source_deployment.json'):
 p=r/rel;target=archive/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
for rel in ('reports/early_gpu_gate_pid.json','reports/early_gpu_gates.json','reports/early_gpu_gates.log','reports/source_deployment.json'):(r/rel).unlink()
print('FAILED_PREPARATION_ARCHIVED_NO_LEARNED_STATE')
"""%(remote,old['sha256'],a.failed_pid,a.tag)
    with connect() as c:print(command(c,shlex.quote(rt['server_python'])+" - <<'PY'\n"+code+"\nPY",30))
    p=ROOT/'reports/source_deployment.json';shutil.copy2(p,archive/'source_deployment.local.json');p.unlink()
    for name in ('preparation_driver.log','preparation_driver.err.log','preparation_driver.json'):
        p=ROOT/'reports'/name
        if p.exists():shutil.copy2(p,archive/name)
    write_json(ROOT/'reports/preparation_recovery.json',result);print(result)

if __name__=='__main__':main()
