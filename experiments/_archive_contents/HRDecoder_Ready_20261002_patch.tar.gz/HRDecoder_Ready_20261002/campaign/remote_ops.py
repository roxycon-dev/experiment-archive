"""Interactive SSH/SFTP bridge. Password remains in process memory only."""
import argparse
import json
import sys
import getpass
from pathlib import Path
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
responses=Path(__file__).resolve().parents[1]/'reports/remote_responses'
responses.mkdir(parents=True,exist_ok=True)
reply_id=0
def reply(result):
    global reply_id
    reply_id+=1
    path=responses/f'reply_{reply_id:04d}.json'
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(reply_file=str(path),exit_code=result.get('exit_code'),ok=result.get('ok'),error=result.get('error'))),flush=True)
for line in sys.stdin:
    try:
        req=json.loads(line)
        op=req.get('op','exec')
        if op=='exec':
            _,out,err=client.exec_command(req['cmd'],timeout=req.get('timeout',120))
            stdout=out.read().decode('utf-8','replace')
            stderr=err.read().decode('utf-8','replace')
            reply(dict(exit_code=out.channel.recv_exit_status(),stdout=stdout,stderr=stderr))
        elif op in ('upload','download'):
            with client.open_sftp() as sftp:
                if op=='upload':
                    last=[-1]
                    def progress(sent,total):
                        pc=int(sent/total*100)//20*20
                        if pc!=last[0]:
                            last[0]=pc;print(f'SFTP_PROGRESS {pc}%',flush=True)
                    sftp.put(req['local'],req['remote'],callback=progress)
                else:
                    sftp.get(req['remote'],req['local'])
            reply(dict(ok=True,op=op))
        elif op=='close':
            break
        else:
            raise ValueError('Unknown operation')
    except Exception as e:
        reply(dict(error=f'{type(e).__name__}: {e}'))
client.close()
