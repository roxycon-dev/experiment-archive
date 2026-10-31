# -*- coding: utf-8 -*-
"""READ-ONLY reachability probe for the round-6 boot.

Deliberately does NOTHING but report: no relaunch, no shutdown, no file
changes.  (The round-5 auto-resume tree must NOT be used now -- it would see
round5_summary.json and shut the freshly booted machine down again.)

Exit codes: 0 = reachable (prints status), 2 = still down.
"""
import sys
import time

import paramiko

HOST, PORT, PASSWORD = '[REDACTED_USE_ENVIRONMENT]', 19057, 'NKgdrekCg6bo'
MAX_WAIT = 540


def main():
    t_end = time.time() + MAX_WAIT
    while time.time() < t_end:
        try:
            cli = paramiko.SSHClient()
            cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            cli.connect(HOST, port=PORT, username='root', password=PASSWORD,
                        timeout=20, banner_timeout=20, auth_timeout=20)
            _, o, _ = cli.exec_command(
                'date -Is; df -h /root/autodl-tmp | tail -1; '
                'nvidia-smi --query-gpu=memory.used --format=csv,noheader; '
                'ls -d /root/autodl-tmp/hrplus/glm/runs/r5 2>/dev/null; '
                'echo READY', timeout=40)
            print('REACHABLE:\n%s' % o.read().decode('utf-8', 'replace'),
                  flush=True)
            cli.close()
            sys.exit(0)
        except Exception as ex:
            print('down (%s): %r' % (time.strftime('%H:%M:%S'), ex),
                  flush=True)
            time.sleep(40)
    print('STATUS: STILL_DOWN', flush=True)
    sys.exit(2)


if __name__ == '__main__':
    main()
