# -*- coding: utf-8 -*-
"""Relaunch the guarded round-5 protocol after the disk-full crash.
Preserves the crash guard.log, verifies free space, relaunches with
SCRIPT=run_round5.py, and confirms the process tree + logs."""
import time

import paramiko

HOST, PORT, PASSWORD = '[REDACTED_USE_ENVIRONMENT]', 19057, 'NKgdrekCg6bo'
PY = '/root/miniconda3/bin/python'
GLM = '/root/autodl-tmp/hrplus/glm'
RUN = GLM + '/runs/r5'


def main():
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli.connect(HOST, port=PORT, username='root', password=PASSWORD,
                timeout=30, banner_timeout=30, auth_timeout=30)

    def sh(cmd, timeout=180):
        _, o, e = cli.exec_command(cmd, timeout=timeout)
        return (o.read().decode('utf-8', 'replace') +
                e.read().decode('utf-8', 'replace')).strip()

    out = sh("ps aux | grep -E 'run_round5|run_guarded|ops.py watchdog' "
             "| grep -v grep || echo CLEAN")
    print('## stale processes: %s' % out)
    if 'CLEAN' not in out:
        print('REFUSING: stale processes'); return
    print('## free space: %s' % sh('df -h /root/autodl-tmp | tail -1'))
    # preserve the crash log before guard.log is overwritten
    sh('cp -n %s/guard.log %s/guard.log.1000crash 2>/dev/null; echo ok'
       % (GLM, GLM))
    print('## crash log preserved: %s'
          % sh('ls -la %s/guard.log.1000crash 2>/dev/null || echo none' % GLM))

    sh('cd %s && (PYTHON=%s SCRIPT=run_round5.py nohup bash run_guarded.sh '
       'settings.json %s %s/backups > guard.log 2>&1 < /dev/null &) ; '
       'echo LAUNCHED' % (GLM, PY, RUN, GLM), timeout=30)
    time.sleep(30)
    out = sh("ps aux | grep -E 'run_round5|run_guarded|ops.py watchdog' "
             "| grep -v grep || echo NOT_ALIVE")
    print('\n## processes after relaunch\n%s' % out)
    print('\n## guard.log tail\n%s' % sh('tail -8 %s/guard.log' % GLM))
    print('\n## protocol.log tail\n%s' % sh('tail -6 %s/protocol.log' % RUN))
    if 'run_round5' not in out:
        print('RELAY FAILED: run_round5 not alive')
    else:
        print('RELAUNCH OK -- guard + watchdog armed, deadline preserved '
              '(2026-09-18T06:47:49Z / 北京 14:47)')
    cli.close()


if __name__ == '__main__':
    main()
