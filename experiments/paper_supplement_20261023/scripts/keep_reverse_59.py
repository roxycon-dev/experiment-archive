"""Task-scoped loopback-only reverse SSH, independent of the UU/FinalShell window."""
from pathlib import Path
import subprocess
import time

ROOT=Path('/home/studentuser001/paper_supplement_20261023')
key=Path.home()/'.ssh/codex_supp_reverse_20261023'
command=['ssh','-N','-T','-i',str(key),'-p','40411','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes',
         '-o','ExitOnForwardFailure=yes','-o','ServerAliveInterval=20','-o','ServerAliveCountMax=3',
         '-R','127.0.0.1:40560:127.0.0.1:22','root@connect.nmb1.seetacloud.com']
while not (ROOT/'logs/stop_reverse_tunnel').exists():
    with (ROOT/'logs/reverse_tunnel.log').open('a') as log:
        result=subprocess.run(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        log.write(f'Connection ended with code{result.returncode}; retrying in10seconds\n')
    time.sleep(10)
