"""Task-scoped SSH channel. Password only on stdin/in memory, never on disk."""
import argparse
import getpass
import json
import sys
from pathlib import Path

import paramiko


ap = argparse.ArgumentParser()
ap.add_argument('--host', required=True)
ap.add_argument('--port', type=int, required=True)
ap.add_argument('--key')
args = ap.parse_args()
root = Path(__file__).resolve().parents[1]
known = root/'ops/known_hosts'
client = paramiko.SSHClient()
if known.exists():
    client.load_host_keys(str(known))
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
password = None
if not args.key:
    print('PASSWORD_STDIN_REQUIRED', flush=True)
    password = getpass.getpass('SSH password: ')
client.connect(args.host, port=args.port, username='root', password=password, key_filename=args.key,
               allow_agent=False, look_for_keys=False, timeout=20, banner_timeout=20, auth_timeout=20)
password = None
client.save_host_keys(str(known))
client.get_transport().set_keepalive(30)
print('SSH_CONNECTED', flush=True)
for line in sys.stdin:
    try:
        req = json.loads(line)
        if req['op'] == 'close':
            break
        if req['op'] == 'exec':
            _, out, err = client.exec_command(req['cmd'], timeout=req.get('timeout', 60))
            response = {'exit_code': out.channel.recv_exit_status(), 'stdout': out.read().decode('utf-8', 'replace'),
                        'stderr': err.read().decode('utf-8', 'replace')}
        elif req['op'] in ('upload', 'download'):
            with client.open_sftp() as sftp:
                if req['op'] == 'upload':
                    sftp.put(req['local'], req['remote'])
                else:
                    Path(req['local']).parent.mkdir(parents=True, exist_ok=True)
                    sftp.get(req['remote'], req['local'])
            response = {'ok': True}
        else:
            raise ValueError('Unknown SSH operation')
        print(json.dumps(response, ensure_ascii=True), flush=True)
    except Exception as exc:
        print(json.dumps({'error': str(exc)}, ensure_ascii=True), flush=True)
client.close()
