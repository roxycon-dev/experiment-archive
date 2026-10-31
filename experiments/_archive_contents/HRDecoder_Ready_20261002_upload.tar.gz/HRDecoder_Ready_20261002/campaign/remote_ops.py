"""Interactive SSH/SFTP bridge. Password remains in process memory only."""
import argparse
import json
import sys
import getpass
import paramiko

ap=argparse.ArgumentParser()
ap.add_argument('--host',required=True)
ap.add_argument('--port',type=int,required=True)
a=ap.parse_args()
print('PASSWORD_REQUIRED',flush=True)
password=getpass.getpass('SSH password: ')
client=paramiko.SSHClient()
client.load_system_host_keys()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(a.host,port=a.port,username='root',password=password,
               timeout=25,allow_agent=False,look_for_keys=False)
password=None
print('SSH_CONNECTED',flush=True)
for line in sys.stdin:
    try:
        req=json.loads(line)
        op=req.get('op','exec')
        if op=='exec':
            _,out,err=client.exec_command(req['cmd'],timeout=req.get('timeout',120))
            stdout=out.read().decode('utf-8','replace')
            stderr=err.read().decode('utf-8','replace')
            print(json.dumps(dict(exit_code=out.channel.recv_exit_status(),stdout=stdout,stderr=stderr),ensure_ascii=True),flush=True)
        elif op in ('upload','download'):
            with client.open_sftp() as sftp:
                if op=='upload':
                    sftp.put(req['local'],req['remote'])
                else:
                    sftp.get(req['remote'],req['local'])
            print(json.dumps(dict(ok=True,op=op)),flush=True)
        elif op=='close':
            break
        else:
            raise ValueError('Unknown operation')
    except Exception as e:
        print(json.dumps(dict(error=f'{type(e).__name__}: {e}')),flush=True)
client.close()
