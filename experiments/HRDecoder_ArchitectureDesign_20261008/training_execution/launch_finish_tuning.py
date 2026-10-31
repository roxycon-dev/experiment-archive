from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'local_preparation'))
from remote import connect
ROOT='/root/autodl-tmp/HRDecoder_v3_20261008'
c=connect()
with c.open_sftp() as ftp:
    for name in ('train_performance.py','channels_last_runtime.py','benchmark_gpu_layout.py',
        'benchmark_ram_loader.py','ram_loader.py','finish_performance_tuning.py'):
        ftp.put(str(HERE/name),ROOT+'/training_execution/'+name)
code="""import subprocess
root='/root/autodl-tmp/HRDecoder_v3_20261008'
f=open(root+'/reports/finish_performance_tuning.log','ab',buffering=0)
p=subprocess.Popen(['/root/miniconda3/bin/python','-u',root+'/training_execution/finish_performance_tuning.py'],stdin=subprocess.DEVNULL,stdout=f,stderr=f,start_new_session=True)
print(p.pid)
"""
with c.open_sftp() as ftp:
    with ftp.open(ROOT+'/training_execution/launch_finish_remote.py','w') as f:f.write(code)
_,out,err=c.exec_command('/root/miniconda3/bin/python '+ROOT+'/training_execution/launch_finish_remote.py')
print(out.read().decode(),end=''); error=err.read().decode(); assert out.channel.recv_exit_status()==0,error
c.close()
