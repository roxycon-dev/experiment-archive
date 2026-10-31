"""SACD-scoped SSH deployment; never alter v3 or diagnosis files."""
import argparse
import json
from pathlib import Path
import shlex
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'local_preparation'))
from remote import connect
REMOTE = '/root/autodl-tmp/HRDecoder_v3_20261008'

def execute(client, command):
    _, stdout, stderr = client.exec_command(command)
    output, error = stdout.read().decode(), stderr.read().decode()
    code = stdout.channel.recv_exit_status()
    if output: print(output, end='', flush=True)
    if code: raise RuntimeError(error or output)

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--command'); args=parser.parse_args()
    client=connect()
    try:
        execute(client, 'mkdir -p '+REMOTE+'/sacd_execution '+REMOTE+'/reports/sacd '+REMOTE+'/training_runs/sacd')
        with client.open_sftp() as ftp:
            for path in sorted(HERE.iterdir()):
                if path.is_file() and path.suffix in ('.py','.json') and path.name not in ('deploy.py','CPU_SACD_PASS.json','CHECKPOINT_RESUME_PASS.json','validation_reporting.json','validation_series.json'):
                    ftp.put(str(path),REMOTE+'/sacd_execution/'+path.name)
        if args.prepare:
            code = '''import json,os,subprocess
from pathlib import Path
root=Path('/root/autodl-tmp/HRDecoder_v3_20261008')
ready=root/'reports/sacd/RAM_CACHE_READY.json'
if ready.exists():
 r=json.loads(ready.read_text())
 try:
  cmd=Path('/proc/%d/cmdline'%r['pid']).read_bytes()
  if b'ram_cache_owner.py' in cmd: print('SACD_RAM_ALREADY_READY');raise SystemExit(0)
 except FileNotFoundError: pass
for proc in Path('/proc').iterdir():
 if proc.name.isdigit() and int(proc.name)!=os.getpid():
  try:
   if str(root/'sacd_execution/ram_cache_owner.py').encode() in (proc/'cmdline').read_bytes(): print('SACD_RAM_ALREADY_BUILDING');raise SystemExit(0)
  except OSError: pass
log=open(root/'reports/sacd/ram_cache.log','ab',buffering=0)
p=subprocess.Popen(['/root/miniconda3/bin/python','-u',str(root/'sacd_execution/ram_cache_owner.py')],cwd=str(root/'sacd_execution'),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps({'SACD_RAM_owner_launched_pid':p.pid}))
'''
            execute(client,'/root/miniconda3/bin/python -c '+shlex.quote(code))
            execute(client,'cd '+REMOTE+'/sacd_execution && /root/miniconda3/bin/python verify_sacd.py')
        if args.command:
            execute(client,args.command)
    finally: client.close()

if __name__=='__main__': main()
