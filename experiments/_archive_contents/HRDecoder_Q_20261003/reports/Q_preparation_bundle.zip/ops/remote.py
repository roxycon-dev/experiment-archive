"""Bounded operations on the one user-assigned instance, using existing authorized key."""
import argparse
import json
import sys
from pathlib import Path

import paramiko

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qlib.common import read_json, write_json


def connect():
    root = Path(__file__).resolve().parents[1]
    config = read_json(root/'configs/runtime.local.json')['server']
    if (config['host'], config['port']) != ('connect.nmb1.seetacloud.com', 26494):
        raise ValueError('Server identity differs from this user-authorized instance')
    client = paramiko.SSHClient()
    client.load_host_keys(str(root/'ops/known_hosts'))
    client.connect(config['host'], port=config['port'], username='root', key_filename=config['ssh_key'],
        allow_agent=False, look_for_keys=False, timeout=20, banner_timeout=20, auth_timeout=20)
    return client


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--command-file')
    ap.add_argument('--output', default='reports/remote_response.json')
    ap.add_argument('--upload', nargs=2, metavar=('LOCAL', 'REMOTE'))
    ap.add_argument('--download', nargs=2, metavar=('REMOTE', 'LOCAL'))
    ap.add_argument('--timeout', type=int, default=60)
    args = ap.parse_args()
    with connect() as client:
        if args.upload or args.download:
            with client.open_sftp() as sftp:
                if args.upload:
                    sftp.put(*args.upload)
                else:
                    Path(args.download[1]).parent.mkdir(parents=True, exist_ok=True)
                    sftp.get(*args.download)
            result = {'ok': True}
        else:
            cmd = Path(args.command_file).read_text(encoding='utf-8')
            _, out, err = client.exec_command(cmd, timeout=args.timeout)
            stdout, stderr = out.read().decode('utf-8', 'replace'), err.read().decode('utf-8', 'replace')
            result = {'exit_code': out.channel.recv_exit_status(), 'stdout': stdout, 'stderr': stderr}
        write_json(args.output, result)
        print(json.dumps({'output': args.output, 'ok': result.get('ok'), 'exit_code': result.get('exit_code')}, ensure_ascii=True))
        if result.get('exit_code', 0):
            sys.exit(1)


if __name__ == '__main__':
    main()
