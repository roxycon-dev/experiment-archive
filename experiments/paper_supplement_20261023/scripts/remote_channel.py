"""Ephemeral SSH/SFTP bridge. Credentials arrive only on stdin, never logged or saved."""
import base64
import hashlib
import getpass
import json
import sys
import time
from pathlib import Path
import paramiko


def emit(value):print(json.dumps(value,ensure_ascii=False),flush=True)


def main():
    auth=json.loads(getpass.getpass('SSH_AUTH (hidden): '))
    client=paramiko.SSHClient();client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(auth['host'],port=auth['port'],username=auth['user'],password=auth.pop('password'),
                   look_for_keys=False,allow_agent=False,timeout=15,auth_timeout=20,banner_timeout=20)
    key=client.get_transport().get_remote_server_key()
    emit({'connected':True,'host_key_type':key.get_name(),'host_key_SHA256':base64.b64encode(hashlib.sha256(key.asbytes()).digest()).decode().rstrip('=')})
    sftp=paramiko.SFTPClient.from_transport(client.get_transport(),window_size=128*1024**2,max_packet_size=1024**2)
    for line in sys.stdin:
        try:
            job=json.loads(line);action=job['action']
            if action=='exec':
                _,stdout,stderr=client.exec_command(job['command'],timeout=job.get('timeout',60))
                out=stdout.read().decode('utf-8','replace');err=stderr.read().decode('utf-8','replace')
                emit({'action':action,'status':stdout.channel.recv_exit_status(),'stdout':out,'stderr':err})
            elif action in ['put','get']:
                previous=[0.0]
                def progress(done,total):
                    if time.monotonic()-previous[0]>10 or done==total:
                        emit({'action':action,'bytes':done,'total':total});previous[0]=time.monotonic()
                if action=='put':sftp.put(job['local'],job['remote'],callback=progress)
                else:
                    Path(job['local']).parent.mkdir(parents=True,exist_ok=True)
                    sftp.get(job['remote'],job['local'],callback=progress)
                emit({'action':action,'status':'COMPLETE','remote':job['remote']})
            elif action=='close':break
            else:raise ValueError('Unsupported action')
        except Exception as error:emit({'error':type(error).__name__,'message':str(error)})
    sftp.close();client.close()


if __name__=='__main__':main()
