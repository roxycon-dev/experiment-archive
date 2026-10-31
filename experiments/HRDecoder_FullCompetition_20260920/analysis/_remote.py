"""Minimal paramiko helper for the RTX3090 box (single allowed server)."""
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
              timeout=30, banner_timeout=60, auth_timeout=60)
    return c


def run(cmd, timeout=1800, client=None):
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


def main():
    cmd = " ".join(sys.argv[1:])
    rc, out, err = run(cmd)
    sys.stdout.write(out)
    if err.strip():
        sys.stderr.write("--- STDERR ---\n" + err)
    print(f"\n[remote_exit={rc}]")


if __name__ == "__main__":
    main()
