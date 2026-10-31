"""Password-in-memory bridge to the user's LAN server through the authorized reverse tunnel."""
import base64
import getpass
import hashlib
import json
from pathlib import Path
import sys
import time
import paramiko

def emit(value):print(json.dumps(value,ensure_ascii=False),flush=True)
root=paramiko.SSHClient();root.set_missing_host_key_policy(paramiko.AutoAddPolicy())
root.connect('connect.nmb1.seetacloud.com',port=40411,username='root',
             key_filename='C:/Users/GuanXingRen/.ssh/codex_hrdecoder_next_20261003',
             look_for_keys=False,allow_agent=False,timeout=15)
fingerprint=base64.b64encode(hashlib.sha256(root.get_transport().get_remote_server_key().asbytes()).digest()).decode().rstrip('=')
assert fingerprint=='yswdgtshuc+hy/m0jR576i7Nu0Es7lZ71jIrFKXnkE0'
auth=json.loads(getpass.getpass('LAN_AUTH (hidden): '))
channel=root.get_transport().open_channel('direct-tcpip',('127.0.0.1',40559),('127.0.0.1',0))
client=paramiko.SSHClient();client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('192.168.3.59',username=auth['user'],password=auth.pop('password'),sock=channel,
               look_for_keys=False,allow_agent=False,timeout=20,auth_timeout=20,banner_timeout=20)
emit({'connected':True,'target':'authorized .59 LAN server','host_key_SHA256':
    base64.b64encode(hashlib.sha256(client.get_transport().get_remote_server_key().asbytes()).digest()).decode().rstrip('=')})
sftp=paramiko.SFTPClient.from_transport(client.get_transport(),window_size=128*1024**2,max_packet_size=1024**2)
for line in sys.stdin:
    try:
        job=json.loads(line);action=job['action']
        if action in ['exec','jump_exec']:
            selected=root if action=='jump_exec' else client
            _,out,err=selected.exec_command(job['command'],timeout=job.get('timeout',60))
            stdout=out.read().decode('utf-8','replace');stderr=err.read().decode('utf-8','replace')
            emit({'action':action,'status':out.channel.recv_exit_status(),'stdout':stdout,'stderr':stderr})
        elif action in ['put','get']:
            prior=[0.]
            def progress(done,total):
                if time.monotonic()-prior[0]>10 or done==total:
                    emit({'action':action,'bytes':done,'total':total});prior[0]=time.monotonic()
            if action=='put':sftp.put(job['local'],job['remote'],callback=progress)
            else:
                Path(job['local']).parent.mkdir(parents=True,exist_ok=True)
                sftp.get(job['remote'],job['local'],callback=progress)
            emit({'action':action,'status':'COMPLETE','remote':job['remote']})
        elif action=='close':break
        else:raise ValueError('Unsupported action')
    except Exception as exc:emit({'error':type(exc).__name__,'message':str(exc)})
sftp.close();client.close();root.close()
