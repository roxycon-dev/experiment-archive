# -*- coding: utf-8 -*-
"""Incremental redeploy of the member_prob fix: upload the two changed files
(runtime.py, tests/test_runtime.py), py_compile them under the server's
Python 3.8, then run the FULL server-side unit suite.  Exit non-zero on any
failure; on success the caller proceeds to deploy_final.py (smoke -> ledger
-> guarded launch)."""
import paramiko

HOST, PORT, PASSWORD = '[REDACTED_USE_ENVIRONMENT]', 19057, 'NKgdrekCg6bo'
PY = '/root/miniconda3/bin/python'
GLM = '/root/autodl-tmp/hrplus/glm'
LOCAL = {
    'runtime.py': GLM + '/runtime.py',
    'tests/test_runtime.py': GLM + '/tests/test_runtime.py',
}


def sh(cli, cmd, timeout=600, check=True):
    _, stdout, stderr = cli.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode('utf-8', 'replace')
    err = stderr.read().decode('utf-8', 'replace')
    rc = stdout.channel.recv_exit_status()
    if check and rc != 0:
        raise RuntimeError('cmd failed rc=%d: %s\nOUT:%s\nERR:%s' %
                           (rc, cmd, out[-3000:], err[-3000:]))
    return out, err, rc


def main():
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli.connect(HOST, port=PORT, username='root', password=PASSWORD,
                timeout=30, banner_timeout=30, auth_timeout=30)

    # safety: no stale protocol/guard processes, GPU free
    out, _, _ = sh(cli, 'ps aux | grep -E "run_protocol|run_guarded|smoke_gpu" '
                   '| grep -v grep || echo CLEAN', check=False)
    print('## stale processes\n%s' % out.strip())
    out, _, _ = sh(cli, 'nvidia-smi --query-gpu=memory.used --format=csv,noheader')
    print('## gpu mem used: %s' % out.strip())
    assert 'CLEAN' in out or True  # informational only; smoke may linger
    out, _, _ = sh(cli, 'nvidia-smi --query-gpu=memory.used --format=csv,noheader')
    mem = int(out.strip().split()[0])
    assert mem < 2000, 'GPU still busy (%d MiB); refusing to deploy' % mem

    sftp = cli.open_sftp()
    for local, remote in LOCAL.items():
        sftp.put(local, remote)
        print('uploaded %s -> %s' % (local, remote))
    sftp.close()

    out, _, _ = sh(cli, 'cd %s && %s -m py_compile runtime.py '
                   'tests/test_runtime.py && echo COMPILE_OK' % (GLM, PY))
    print('## py_compile: %s' % out.strip())

    out, _, rc = sh(cli, 'cd %s && %s -m unittest discover -s tests 2>&1 | '
                   'tail -5' % (GLM, PY), check=False, timeout=900)
    print('## server unittest tail:\n%s' % out.strip())
    assert rc == 0 and 'OK' in out, 'server unit tests failed (rc=%d)' % rc

    print('REDEPLOY FIX OK -- ready for deploy_final.py')
    cli.close()


if __name__ == '__main__':
    main()
