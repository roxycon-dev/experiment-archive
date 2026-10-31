#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""HRDecoder 远程执行助手：通过 paramiko 在 AutoDL 实例上执行命令。

用法:
  python rssh.py "命令"                 # 执行单条命令
  python rssh.py -f script.sh           # 上传并执行本地脚本
  python rssh.py -t 60 "命令"           # 指定超时
"""
import os
import sys
import time
import argparse

import paramiko

HOST = "connect.nmb1.seetacloud.com"
PORT = 37047
USER = "root"
PWD = os.environ.get("HR_PWD", "")
KEY = os.environ.get("HR_KEY", "")


def connect():
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    pwd = PWD or open(KEY, encoding="utf-8").read().strip()
    cli.connect(HOST, port=PORT, username=USER, password=pwd,
                timeout=30, banner_timeout=30, auth_timeout=30,
                allow_agent=False, look_for_keys=False)
    return cli


def run(cli, cmd, timeout=300):
    chan = cli.get_transport().open_session()
    chan.settimeout(timeout)
    chan.exec_command(cmd)
    out, err = [], []
    while True:
        if chan.recv_ready():
            out.append(chan.recv(65536).decode("utf-8", "replace"))
        if chan.recv_stderr_ready():
            err.append(chan.recv_stderr(65536).decode("utf-8", "replace"))
        if chan.exit_status_ready() and not chan.recv_ready() \
                and not chan.recv_stderr_ready():
            break
        time.sleep(0.05)
    rc = chan.recv_exit_status()
    chan.close()
    return rc, "".join(out), "".join(err)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", default="")
    ap.add_argument("-f", "--file")
    ap.add_argument("-t", "--timeout", type=int, default=300)
    a = ap.parse_args()

    cli = connect()
    try:
        if a.file:
            sftp = cli.open_sftp()
            remote = "/root/_run.sh"
            sftp.put(a.file, remote)
            sftp.close()
            cmd = "bash " + remote
        else:
            cmd = a.cmd
        rc, out, err = run(cli, cmd, a.timeout)
        print(f"### exit={rc}")
        if out:
            print(out)
        if err.strip():
            print("--- stderr ---")
            print(err)
    finally:
        cli.close()


if __name__ == "__main__":
    main()
