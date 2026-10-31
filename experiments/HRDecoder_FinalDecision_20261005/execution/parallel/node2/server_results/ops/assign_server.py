"""User-supplied server; ephemeral getpass only if configured key is not installed."""
import sys,argparse,getpass,shlex,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import paramiko
from qlib.common import read_json,write_json,utc_now

def main(host,port):
    runtime=read_json(ROOT/'configs/runtime.local.json');key=runtime['server']['ssh_key']
    known=ROOT/'ops/known_hosts';client=paramiko.SSHClient()
    if known.exists():client.load_host_keys(str(known))
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:client.connect(host,port=port,username='root',key_filename=key,allow_agent=False,look_for_keys=False,timeout=15,auth_timeout=15)
    except paramiko.AuthenticationException:
        client.close();client=paramiko.SSHClient()
        if known.exists():client.load_host_keys(str(known))
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        secret=getpass.getpass('Temporary assigned-server SSH password (not saved): ')
        client.connect(host,port=port,username='root',password=secret,allow_agent=False,look_for_keys=False,timeout=15,auth_timeout=15)
        del secret
    client.save_host_keys(str(known))
    key_object=paramiko.PKey.from_path(key)
    public=key_object.get_name()+' '+key_object.get_base64()
    command='mkdir -p /root/.ssh && chmod 700 /root/.ssh && touch /root/.ssh/authorized_keys && chmod 600 /root/.ssh/authorized_keys && (grep -qxF -- '+shlex.quote(public)+' /root/.ssh/authorized_keys || printf "%s\\n" '+shlex.quote(public)+' >> /root/.ssh/authorized_keys)'
    _,out,err=client.exec_command(command);assert out.channel.recv_exit_status()==0,err.read().decode()
    server=dict(runtime['server'],host=host,port=port,assignment_confirmed=True,assigned_at=utc_now(),
        instance_id=f'{host}:{port}',password_stored=False)
    for name in ('runtime.local.json','runtime.server.json'):
        config=read_json(ROOT/'configs'/name);config['server']=server;config['stage']='SERVER_ASSIGNED_LOCAL_PREPARATION_PENDING'
        config['test_p0_cache']='/dev/shm/hrdecoder_final_p0_20261005'
        write_json(ROOT/'configs'/name,config)
    print(json.dumps({'stage':'ASSIGNED_KEY_AUTHORIZED','host':host,'port':port,'known_host_saved':True,'password_stored':False}))
    client.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--host',required=True);p.add_argument('--port',type=int,required=True);a=p.parse_args();main(a.host,a.port)
