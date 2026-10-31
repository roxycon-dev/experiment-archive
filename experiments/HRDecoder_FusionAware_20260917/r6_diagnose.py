# -*- coding: utf-8 -*-
"""Round-6 incident diagnosis (read-only) + optional partial-tar cleanup.

Prints: r6.log tail, DEVIATIONS.jsonl, guard.log tail, df, backups listing,
feature-cache presence, and the r6 state.  With --clean-partials it also
removes any *.tar in glm/backups (a truncated EXIT-trap backup would be
holding the disk hostage); nothing else is touched.
"""
import sys

import paramiko

HOST, PORT, PASSWORD = '[REDACTED_USE_ENVIRONMENT]', 19057, 'NKgdrekCg6bo'
GLM = '/root/autodl-tmp/hrplus/glm'
RUN = GLM + '/runs/r6'


def main():
    clean = '--clean-partials' in sys.argv
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli.connect(HOST, port=PORT, username='root', password=PASSWORD,
                timeout=25, banner_timeout=25, auth_timeout=25)

    def sh(cmd, timeout=180):
        _, o, e = cli.exec_command(cmd, timeout=timeout)
        return (o.read().decode('utf-8', 'replace') +
                e.read().decode('utf-8', 'replace')).strip()

    print('== time ==\n%s' % sh('date -Is; uptime'))
    print('== df ==\n%s' % sh('df -h /root/autodl-tmp | tail -1'))
    print('== processes ==\n%s'
          % sh("ps aux | grep -E 'r6_run|run_guarded|ops.py watchdog' "
               "| grep -v grep || echo CLEAN"))
    print('== r6.log tail ==\n%s' % sh('tail -20 %s/r6.log' % RUN))
    print('== DEVIATIONS ==\n%s'
          % sh('cat %s/DEVIATIONS.jsonl 2>/dev/null || echo none' % RUN))
    print('== guard.log tail ==\n%s'
          % sh('tail -25 %s/guard.log 2>/dev/null || echo none' % GLM))
    print('== backups ==\n%s'
          % sh('ls -la %s/backups 2>/dev/null || echo none' % GLM))
    print('== r6 state (keys) ==\n%s'
          % sh('cat %s/r6_state.json 2>/dev/null | head -40 || echo none'
               % RUN))
    print('== feature caches ==\n%s'
          % sh('du -sh %s/featcache_r6/*/* 2>/dev/null || echo none' % RUN))
    print('== runs/r6 total ==\n%s' % sh('du -sh %s' % RUN))

    if clean:
        print('== cleanup: removing backups/*.tar ==')
        print(sh('ls -la %s/backups/*.tar 2>/dev/null; rm -f %s/backups/*.tar; '
                 'echo done' % (GLM, GLM)))
        print('== df after ==\n%s' % sh('df -h /root/autodl-tmp | tail -1'))
    cli.close()


if __name__ == '__main__':
    main()
