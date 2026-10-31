from pathlib import Path
import sys,json
sys.path.insert(0,str(Path(__file__).resolve().parent))
from mirror_remote import ROOT,REMOTE,connect,remote_exec,write_json
c=connect()
try:
 with c.open_sftp() as sftp:sftp.put(str(ROOT/'ops/preserve_remaining_g.py'),REMOTE+'/ops/preserve_remaining_g.py')
 code="""import subprocess,os,json,time
from pathlib import Path
r=Path(REMOTE)
with (r/'reports/persistent_nonwinner_G.log').open('ab') as log:
 p=subprocess.Popen(['/root/miniconda3/bin/python','ops/preserve_remaining_g.py'],cwd=r,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUNBUFFERED='1'))
print(json.dumps(dict(pid=p.pid,time=time.time())))
""".replace('REMOTE',repr(REMOTE))
 data=json.loads(remote_exec(c,"/root/miniconda3/bin/python - <<'PY'\n"+code+"\nPY",20))
 write_json(ROOT/'reports/persistent_nonwinner_G_pid.json',data);print(json.dumps(data))
finally:c.close()
