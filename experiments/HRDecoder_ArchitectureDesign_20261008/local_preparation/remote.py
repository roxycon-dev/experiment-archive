"""Campaign-scoped SSH operations; authenticate with the local task key."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import paramiko

HOST = "connect.nmb1.seetacloud.com"
PORT = 11347
KEY = Path.home() / ".ssh/hrdecoder_v3_20261008"
KNOWN = Path.home() / ".ssh/hrdecoder_v3_known_hosts"

def connect():
    client = paramiko.SSHClient()
    client.load_host_keys(str(KNOWN))
    client.set_missing_host_key_policy(paramiko.RejectPolicy())
    client.connect(HOST, port=PORT, username="root", key_filename=str(KEY),
                   timeout=20, auth_timeout=20, look_for_keys=False, allow_agent=False)
    return client

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["cmd", "put", "get"])
    parser.add_argument("source")
    parser.add_argument("destination", nargs="?")
    args = parser.parse_args()
    client = connect()
    try:
        if args.action == "cmd":
            stdin, stdout, stderr = client.exec_command(args.source)
            channel = stdout.channel
            while not channel.exit_status_ready() or channel.recv_ready() or channel.recv_stderr_ready():
                import select
                select.select([channel], [], [], 1)
                if channel.recv_ready():
                    sys.stdout.buffer.write(channel.recv(65536)); sys.stdout.buffer.flush()
                if channel.recv_stderr_ready():
                    sys.stderr.buffer.write(channel.recv_stderr(65536)); sys.stderr.buffer.flush()
            sys.exit(channel.recv_exit_status())
        else:
            with client.open_sftp() as sftp:
                if args.action == "put":
                    sftp.put(args.source, args.destination)
                else:
                    Path(args.destination).parent.mkdir(parents=True, exist_ok=True)
                    sftp.get(args.source, args.destination)
            print(json.dumps({"action": args.action, "source": args.source, "destination": args.destination}))
    finally:
        client.close()

if __name__ == "__main__":
    main()
