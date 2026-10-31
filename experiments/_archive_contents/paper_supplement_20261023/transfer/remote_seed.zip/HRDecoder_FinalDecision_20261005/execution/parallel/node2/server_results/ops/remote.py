"""Never connect to a stale assignment, and never save/use a password."""
import sys
from pathlib import Path
import paramiko
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from qlib.common import read_json

def connect():
    server=read_json(ROOT/'configs/runtime.local.json')['server']
    assert server.get('host') and server.get('port') and server.get('assignment_confirmed'), 'New server assignment not supplied by user'
    client=paramiko.SSHClient();known=ROOT/'ops/known_hosts'
    assert known.exists(),'Known host identity must be established for this assigned server'
    client.load_host_keys(str(known));client.set_missing_host_key_policy(paramiko.RejectPolicy())
    client.connect(server['host'],port=int(server['port']),username='root',key_filename=server['ssh_key'],
        allow_agent=False,look_for_keys=False,timeout=15,banner_timeout=15,auth_timeout=15)
    client.get_transport().set_keepalive(30);return client

def command(client,text,timeout=60):
    # Send scripts through stdin; a complete896-file SHA manifest exceeds SSH
    # exec-packet and shell single-argument limits when embedded in a command.
    inp,out,err=client.exec_command('/bin/bash -s',timeout=timeout)
    inp.write(text+'\n');inp.flush();inp.channel.shutdown_write()
    stdout=out.read().decode('utf-8','replace');stderr=err.read().decode('utf-8','replace')
    code=out.channel.recv_exit_status()
    if code:raise RuntimeError(f'Remote exit{code}: {stderr[-2000:]} {stdout[-500:]}')
    return stdout
