"""Run a named, reviewed CPU diagnostic through the existing authenticated route."""
import argparse
from pathlib import Path
import shlex
import sys

here=Path(__file__).resolve().parent
sys.path.insert(0,str(here.parent/'local_preparation'))
from remote import connect

p=argparse.ArgumentParser(); p.add_argument('script'); p.add_argument('extra', nargs=argparse.REMAINDER); args=p.parse_args()
source=(here/args.script).resolve()
assert source.parent==here and source.suffix=='.py'
remote_root='/root/autodl-tmp/HRDecoder_v3_20261008'
remote_file=remote_root+'/reports/diagnostics/'+source.name
c=connect()
try:
    with c.open_sftp() as s: s.put(str(source),remote_file)
    command='OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /root/miniconda3/bin/python -u '+shlex.quote(remote_file)+' '+ ' '.join(shlex.quote(v) for v in args.extra)
    _,stdout,stderr=c.exec_command(command)
    for line in stdout: print(line,end='',flush=True)
    error=stderr.read().decode()
    if error: print(error[-4000:],file=sys.stderr)
    status=stdout.channel.recv_exit_status()
    print('REMOTE_EXIT_STATUS',status,flush=True)
    destination=here/'evidence'; destination.mkdir(exist_ok=True)
    with c.open_sftp() as s:
        try:
            for item in s.listdir(remote_root+'/reports/diagnostics/v4'):
                if item.endswith('.json'):
                    s.get(remote_root+'/reports/diagnostics/v4/'+item,str(destination/item))
        except FileNotFoundError: pass
    sys.exit(status)
finally:c.close()
