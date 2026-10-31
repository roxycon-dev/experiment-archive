from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/"local_preparation"))
from remote import connect
ROOT='/root/autodl-tmp/HRDecoder_v3_20261008'
client=connect()
with client.open_sftp() as ftp:
    ftp.put(str(HERE/'ram_probability_cache.py'),ROOT+'/training_execution/ram_probability_cache.py')
command="""cd /root/autodl-tmp/HRDecoder_v3_20261008/training_execution
/root/miniconda3/bin/python -c 'import subprocess; f=open("../reports/ram_probability_cache.log","ab",buffering=0); p=subprocess.Popen(["/root/miniconda3/envs/hrdecoder_q_20261003/bin/python","-u","ram_probability_cache.py"],stdin=subprocess.DEVNULL,stdout=f,stderr=f,start_new_session=True); print(p.pid)'
"""
_,stdout,stderr=client.exec_command(command)
print(stdout.read().decode(),end='')
error=stderr.read().decode()
assert stdout.channel.recv_exit_status()==0,error
client.close()
