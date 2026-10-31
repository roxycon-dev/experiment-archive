# -*- coding: utf-8 -*-
"""Round-6 recovery: diagnose, clean partial trap backups, re-upload the two
patched files, and relaunch the guarded protocol (state is idempotent: P0 is
cached, the HE feature cache is complete on disk, so P1 resumes straight into
head training).  Ledger deadline (19:17) is preserved."""
import os
import sys
import time

import paramiko

HOST, PORT, PASSWORD = '[REDACTED_USE_ENVIRONMENT]', 19057, 'NKgdrekCg6bo'
PY = '/root/miniconda3/bin/python'
GLM = '/root/autodl-tmp/hrplus/glm'
RUN = GLM + '/runs/r6'
LOCAL_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD = {
    'r6_head.py': GLM + '/r6_head.py',
    'r6_lib.py': GLM + '/r6_lib.py',
    'r6_features.py': GLM + '/r6_features.py',
    'r6_run.py': GLM + '/r6_run.py',
    'run_guarded.sh': GLM + '/run_guarded.sh',
}


def sh(cli, cmd, timeout=300, check=False):
    _, o, e = cli.exec_command(cmd, timeout=timeout)
    out = o.read().decode('utf-8', 'replace')
    err = e.read().decode('utf-8', 'replace')
    rc = o.channel.recv_exit_status()
    if check and rc != 0:
        raise RuntimeError('rc=%d cmd=%s\n%s' % (rc, cmd, err[-1500:]))
    return out, err, rc


def main():
    cli = None
    for i in range(8):
        try:
            cli = paramiko.SSHClient()
            cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            cli.connect(HOST, port=PORT, username='root', password=PASSWORD,
                        timeout=20, banner_timeout=20, auth_timeout=20)
            break
        except Exception as ex:
            print('probe %d (%s): %r' % (i + 1, time.strftime('%H:%M:%S'), ex))
            cli = None
            time.sleep(30)
    if cli is None:
        print('STATUS: STILL_DOWN')
        sys.exit(2)

    print('== time ==\n%s' % sh(cli, 'date -Is; uptime')[0])
    print('== df ==\n%s' % sh(cli, 'df -h /root/autodl-tmp | tail -1')[0])
    print('== processes ==\n%s'
          % sh(cli, "ps aux | grep -E 'r6_run|run_guarded|ops.py watchdog' "
                    "| grep -v grep || echo CLEAN")[0])
    print('== r6.log tail ==\n%s' % sh(cli, 'tail -12 %s/r6.log' % RUN)[0])
    print('== guard.log tail ==\n%s'
          % sh(cli, 'tail -20 %s/guard.log' % GLM)[0])
    print('== DEVIATIONS ==\n%s'
          % sh(cli, 'cat %s/DEVIATIONS.jsonl 2>/dev/null || echo none' % RUN)[0])
    print('== backups ==\n%s'
          % sh(cli, 'ls -la %s/backups 2>/dev/null || echo none' % GLM)[0])
    print('== caches ==\n%s'
          % sh(cli, 'du -sh %s/featcache_r6/*/* 2>/dev/null || echo none'
               % RUN)[0])

    out, _, _ = sh(cli, "ps aux | grep -E 'r6_run|run_guarded' | grep -v grep "
                        "|| echo CLEAN")
    if 'CLEAN' not in out:
        print('REFUSING: stale processes present'); sys.exit(3)

    print('== removing partial/old trap backups ==')
    print(sh(cli, 'rm -f %s/backups/*.tar; ls -la %s/backups; '
                  'df -h /root/autodl-tmp | tail -1' % (GLM, GLM))[0])

    sftp = cli.open_sftp()
    for local, remote in UPLOAD.items():
        sftp.put(os.path.join(LOCAL_DIR, local), remote)
        print('uploaded %s' % local)
    sftp.close()
    out, err, rc = sh(cli, 'cd %s && %s -m py_compile r6_head.py r6_run.py '
                      'r6_features.py && bash -n run_guarded.sh && echo OK' % (GLM, PY))
    print('== server compile: %s' % out.strip())
    if rc != 0:
        print(err[-1500:]); print('REFUSING: compile failed'); sys.exit(3)
    out, err, rc = sh(cli, 'cd %s && %s -m unittest discover -s tests 2>&1'
                      % (GLM, PY), timeout=900)
    print('== server unittest: %s'
          % (out.strip().splitlines()[-1] if out.strip() else '?'))
    if rc != 0 or '\nOK' not in out[-4000:]:
        print(out[-2500:]); print('REFUSING: server tests failed'); sys.exit(3)

    out, err, rc = sh(cli, 'cd %s && %s -u r6_run.py --settings settings.json '
                      '--run-dir %s --rehearse 2>&1' % (GLM, PY, RUN),
                      timeout=2400)
    print('== rehearsal tail ==\n%s' % out[-2500:])
    if rc != 0 or 'REHEARSE OK' not in out:
        print('REFUSING: rehearsal failed (rc=%d) -- machine left UP for '
              'debugging' % rc)
        sys.exit(4)

    sh(cli, 'cd %s && (SCRIPT=r6_run.py PYTHON=%s nohup bash run_guarded.sh '
            'settings.json %s %s/backups > guard.log 2>&1 < /dev/null &) ; '
            'echo LAUNCHED' % (GLM, PY, RUN, GLM), timeout=30)
    time.sleep(30)
    out, _, _ = sh(cli, "ps aux | grep -E 'r6_run|run_guarded|ops.py watchdog' "
                        "| grep -v grep || echo NOT_ALIVE")
    print('== processes after relaunch\n%s' % out)
    print('== r6.log tail\n%s' % sh(cli, 'tail -8 %s/r6.log' % RUN)[0])
    if 'r6_run' not in out:
        print('RELAY FAILED'); sys.exit(3)
    print('RELAUNCH OK -- resumes at P1 (HE cache present on disk)')
    cli.close()


if __name__ == '__main__':
    main()
