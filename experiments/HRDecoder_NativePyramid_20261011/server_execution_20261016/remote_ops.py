"""Task SSH transport. Password is read from process environment, never saved."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import paramiko


def connect():
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(os.environ["HR_V3_SSH_HOST"], port=int(os.environ["HR_V3_SSH_PORT"]),
                   username="root", password=os.environ["HR_V3_SSH_PASSWORD"],
                   timeout=25, auth_timeout=30, banner_timeout=30,
                   look_for_keys=False, allow_agent=False)
    client.get_transport().set_keepalive(20)
    return client


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="action", required=True)
    run = sub.add_parser("run"); run.add_argument("--command-file", required=True)
    run.add_argument("--timeout", type=int, default=300)
    upload = sub.add_parser("put"); upload.add_argument("local"); upload.add_argument("remote")
    download = sub.add_parser("get"); download.add_argument("remote"); download.add_argument("local")
    args = parser.parse_args()
    client = connect()
    try:
        if args.action == "run":
            command = Path(args.command_file).read_text(encoding="utf-8")
            _, stdout, stderr = client.exec_command(command, timeout=args.timeout)
            channel = stdout.channel
            while not channel.exit_status_ready() or channel.recv_ready() or channel.recv_stderr_ready():
                if channel.recv_ready():
                    sys.stdout.write(channel.recv(65536).decode("utf-8", errors="replace")); sys.stdout.flush()
                if channel.recv_stderr_ready():
                    sys.stderr.write(channel.recv_stderr(65536).decode("utf-8", errors="replace")); sys.stderr.flush()
                if not channel.recv_ready() and not channel.recv_stderr_ready():
                    time.sleep(.1)
            raise SystemExit(channel.recv_exit_status())
        sftp = client.open_sftp()
        try:
            started = time.monotonic()
            def progress(done, total):
                if total and done == total:
                    print(json.dumps({"bytes": total, "seconds": round(time.monotonic()-started, 2)}), flush=True)
            if args.action == "put":
                sftp.put(args.local, args.remote, callback=progress, confirm=True)
            else:
                Path(args.local).parent.mkdir(parents=True, exist_ok=True)
                sftp.get(args.remote, args.local, callback=progress)
        finally:
            sftp.close()
    finally:
        client.close()


if __name__ == "__main__":
    main()
