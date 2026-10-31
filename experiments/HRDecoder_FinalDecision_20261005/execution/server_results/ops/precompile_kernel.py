"""Overlap the pinned compiler build with the immutable input upload; no training."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from qlib.common import read_json,write_json,sha,utc_now
from ops.remote import connect,command
rt=read_json(ROOT/'configs/runtime.local.json');remote=rt['server']['remote_root']
files=[ROOT/'kernel.py',ROOT/'qlib/common.py',ROOT/'qlib/__init__.py',ROOT/'reports/kernel_source_identity.json',*[p for p in (ROOT/'vendor/msda_compat/src').rglob('*') if p.is_file()]]
with connect() as c:
    command(c,"/root/miniconda3/bin/python - <<'PY'\nfrom pathlib import Path\nr=Path(%r)\nassert not (r/'reports/driver_state.json').exists()\nfor sub in ('reports','qlib','vendor/msda_compat/src/cpu','vendor/msda_compat/src/cuda'): (r/sub).mkdir(parents=True,exist_ok=True)\nPY"%remote,30)
    with c.open_sftp() as s:
        for p in files:s.put(str(p),remote+'/'+p.relative_to(ROOT).as_posix())
    code="""
from pathlib import Path
import subprocess,json,hashlib
r=Path(%r);files=json.loads(%r)
for rel,expected in files.items():assert hashlib.sha256((r/rel).read_bytes()).hexdigest()==expected
pidfile=r/'reports/kernel_precompile_pid.json'
if pidfile.exists():
 assert %r,'Existing compiler requires concrete diagnosis'
 old=json.loads(pidfile.read_text());assert not Path('/proc/'+str(old['pid'])).exists(),'Compiler still healthy/running'
 pidfile.rename(r/'reports'/('kernel_precompile_failed_'+str(old['pid'])+'.json'))
with (r/'reports/kernel_precompile.log').open('ab') as log:
 p=subprocess.Popen([%r,'-u','-c',"import kernel;kernel.load_kernel();print('PINNED_KERNEL_BUILD_PASS',flush=True)"],cwd=r,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
result={'pid':p.pid,'stage':'PINNED_COMPILE_STARTED_NO_TRAINING'};pidfile.write_text(json.dumps(result));print(json.dumps(result))
"""%(remote,json.dumps({p.relative_to(ROOT).as_posix():sha(p) for p in files}),'--diagnosed-retry' in sys.argv,rt['server_python'])
    result=json.loads(command(c,"/root/miniconda3/bin/python - <<'PY'\n"+code+"\nPY",30))
result['time']=utc_now();write_json(ROOT/'reports/kernel_precompile_launch.json',result);print(result)
