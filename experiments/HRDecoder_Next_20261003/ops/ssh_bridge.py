"""SSH/SFTP control channel; credential only in process memory, never in reports."""
import argparse
import getpass
import json
import sys
import os
from pathlib import Path
import paramiko

ap=argparse.ArgumentParser();ap.add_argument('--host',required=True);ap.add_argument('--port',type=int,required=True)
ap.add_argument('--key')
a=ap.parse_args()
if not a.key:
    print('PASSWORD_REQUIRED',flush=True)
password=None if a.key else getpass.getpass('SSH password: ')
client=paramiko.SSHClient();client.load_system_host_keys()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(a.host,port=a.port,username='root',password=password,key_filename=a.key,timeout=25,
               allow_agent=False,look_for_keys=False)
client.save_host_keys(str(Path(__file__).resolve().parent/'known_hosts'))
password=None
client.get_transport().set_keepalive(30)
print('SSH_CONNECTED',flush=True)
reports=Path(__file__).resolve().parents[1]/'reports/remote_responses';reports.mkdir(parents=True,exist_ok=True)
existing=[int(p.stem.split('_')[1]) for p in reports.glob('reply_*.json')]
counter=max(existing,default=0)
def reply(value):
    global counter
    counter+=1
    path=reports/f'reply_{counter:04d}_{os.getpid()}.json'
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(reply_file=str(path),exit_code=value.get('exit_code'),ok=value.get('ok'),error=value.get('error'))),flush=True)
for line in sys.stdin:
    try:
        req=json.loads(line);op=req.get('op','exec')
        if op=='exec':
            _,out,err=client.exec_command(req['cmd'],timeout=req.get('timeout',120))
            stdout=out.read().decode('utf-8','replace');stderr=err.read().decode('utf-8','replace')
            reply(dict(exit_code=out.channel.recv_exit_status(),stdout=stdout,stderr=stderr))
        elif op in ('upload','download'):
            with client.open_sftp() as sftp:
                if op=='upload':
                    last=[-1]
                    def progress(sent,total):
                        p=int(sent/total*100)//25*25
                        if p!=last[0]:last[0]=p;print(f'SFTP_PROGRESS {p}%',flush=True)
                    sftp.put(req['local'],req['remote'],callback=progress)
                else:
                    Path(req['local']).parent.mkdir(parents=True,exist_ok=True)
                    sftp.get(req['remote'],req['local'])
            reply(dict(ok=True,op=op))
        elif op=='close':break
        else:raise ValueError('Unknown operation')
    except Exception as exc:
        reply(dict(error=f'{type(exc).__name__}: {exc}'))
client.close()
