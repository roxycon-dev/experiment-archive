"""paramiko helper: upload local script -> run on the allowed RTX3090 box."""
import os
import sys
import paramiko

HOST = "connect.nmb2.seetacloud.com"
PORT = 48301
USER = "root"
PWD = "[REDACTED_USE_ENVIRONMENT]"


def connect():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, port=PORT, username=USER, password=PWD,
              timeout=30, banner_timeout=90, auth_timeout=90)
    tr = c.get_transport()
    if tr is not None:
        tr.set_keepalive(30)
    return c


def run(cmd, timeout=3600, client=None):
    own = client is None
    c = client or connect()
    try:
        stdin, stdout, stderr = c.exec_command(cmd, timeout=timeout, get_pty=False)
        out = stdout.read().decode("utf-8", "replace")
        err = stderr.read().decode("utf-8", "replace")
        rc = stdout.channel.recv_exit_status()
        return rc, out, err
    finally:
        if own:
            c.close()


def put_bytes(data, remote_path, client=None):
    own = client is None
    c = client or connect()
    try:
        sftp = c.open_sftp()
        with sftp.open(remote_path, "wb") as fh:
            fh.write(data)
        sftp.close()
    finally:
        if own:
            c.close()


def get_bytes(remote_path, client=None):
    own = client is None
    c = client or connect()
    try:
        sftp = c.open_sftp()
        with sftp.open(remote_path, "rb") as fh:
            data = fh.read()
        sftp.close()
        return data
    finally:
        if own:
            c.close()


def run_script(local_path, remote_path=None, timeout=3600, client=None):
    """Upload a local bash script and execute it remotely, streaming output."""
    remote_path = remote_path or ("/root/_agent_" + os.path.basename(local_path))
    own = client is None
    c = client or connect()
    try:
        sftp = c.open_sftp()
        sftp.put(local_path, remote_path)
        sftp.close()
        c.exec_command(f"sed -i 's/\\r$//' {remote_path}")[1].channel.recv_exit_status()
        return run(f"bash {remote_path}", timeout=timeout, client=c)
    finally:
        if own:
            c.close()


def main():
    if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]):
        rc, out, err = run_script(sys.argv[1], timeout=7200)
    else:
        rc, out, err = run(" ".join(sys.argv[1:]))
    sys.stdout.write(out)
    if err.strip():
        sys.stderr.write("--- STDERR ---\n" + err)
    print(f"\n[remote_exit={rc}]")


if __name__ == "__main__":
    main()
