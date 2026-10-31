"""Task-scoped existing SSH key. No configuration scans or password storage."""
from pathlib import Path
import sys
import paramiko
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from qlib.common import read_json


def connect():
    server=read_json(ROOT/'configs/runtime.local.json')['server']
    client=paramiko.SSHClient()
    known=ROOT/'ops/known_hosts'
    if known.exists():client.load_host_keys(str(known))
    client.connect(server['host'],port=server['port'],username='root',key_filename=server['ssh_key'],
                   allow_agent=False,look_for_keys=False,timeout=15,banner_timeout=15,auth_timeout=15)
    client.get_transport().set_keepalive(30)
    return client


def command(client,text,timeout=60):
    _,out,err=client.exec_command(text,timeout=timeout)
    stdout=out.read().decode('utf-8','replace');stderr=err.read().decode('utf-8','replace')
    code=out.channel.recv_exit_status()
    if code:raise RuntimeError(f'Remote exit {code}: {stderr[-2000:]} {stdout[-500:]}')
    return stdout
