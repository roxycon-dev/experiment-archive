"""Start the checkpoint watcher and cache owner; no duplicate training process."""
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'local_preparation'))
from remote import connect
ROOT='/root/autodl-tmp/HRDecoder_v3_20261008'
c=connect()
with c.open_sftp() as ftp:
    for name in ('ram_cache_mixed.py','pause_at_checkpoint.py','ram_loader.py'):
        ftp.put(str(HERE/name),ROOT+'/training_execution/'+name)
code="""import subprocess,json
root='/root/autodl-tmp/HRDecoder_v3_20261008'
results={}
for name,python in [('pause_at_checkpoint.py','/root/miniconda3/bin/python'),('ram_cache_mixed.py','/root/miniconda3/envs/hrdecoder_q_20261003/bin/python')]:
 f=open(root+'/reports/'+name+'.log','ab',buffering=0)
 p=subprocess.Popen([python,'-u',root+'/training_execution/'+name],stdin=subprocess.DEVNULL,stdout=f,stderr=f,start_new_session=True)
 results[name]=p.pid
print(json.dumps(results))
"""
with c.open_sftp() as ftp:
    with ftp.open(ROOT+'/training_execution/launch_final_tuning.py','w') as f:f.write(code)
_,out,err=c.exec_command('/root/miniconda3/bin/python '+ROOT+'/training_execution/launch_final_tuning.py')
print(out.read().decode(),end=''); error=err.read().decode()
assert out.channel.recv_exit_status()==0,error
c.close()
