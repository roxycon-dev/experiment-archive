# -*- coding: utf-8 -*-
"""Wait-poll: block until a pattern appears in protocol.log (or timeout),
then print the key evidence lines.  Usage:
    python wait_poll.py "iter 3000 val" 900
(arg1 = grep pattern, arg2 = max seconds to wait server-side)."""
import sys
import time

import paramiko

HOST, PORT, PASSWORD = '[REDACTED_USE_ENVIRONMENT]', 19057, 'NKgdrekCg6bo'
LOG = '/root/autodl-tmp/hrplus/glm/runs/r1/protocol.log'


def connect():
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli.connect(HOST, port=PORT, username='root', password=PASSWORD,
                timeout=25, banner_timeout=25, auth_timeout=30)
    return cli


def main():
    pattern = sys.argv[1] if len(sys.argv) > 1 else 'val mIoU'
    max_wait = int(sys.argv[2]) if len(sys.argv) > 2 else 900
    cli = connect()

    def sh(cmd, timeout=90):
        nonlocal cli
        for attempt in range(3):
            try:
                _, o, e = cli.exec_command(cmd, timeout=timeout)
                return o.read().decode('utf-8', 'replace').strip()
            except Exception:
                if attempt == 2:
                    raise
                try:
                    cli.close()
                except Exception:
                    pass
                time.sleep(5)
                cli = connect()
        return ''

    t0 = time.time()
    while True:
        hit = sh("grep -F %r %s | tail -3" % (pattern, LOG))
        if hit:
            print('## HIT %r after %.0fs' % (pattern, time.time() - t0))
            break
        if time.time() - t0 > max_wait:
            print('## TIMEOUT waiting for %r (%.0fs)' % (pattern, max_wait))
            break
        time.sleep(45)

    print('## time\n%s' % sh('date -Is'))
    print('## key lines')
    print(sh("grep -E 'iter [0-9]+ val mIoU|starting D val|champion val|pilot|"
             "COMPLETE|deadline|gate|seed|final|lock|test|shutdown' %s | tail -25" % LOG))
    print('## gpu\n%s' % sh('nvidia-smi --query-gpu=memory.used,utilization.gpu '
                            '--format=csv,noheader'))
    print('## run_protocol alive: %s' %
          sh("ps aux | grep run_protocol | grep -v grep | wc -l"))
    print('## summary.json: %s' %
          sh('test -f /root/autodl-tmp/hrplus/glm/runs/r1/summary.json '
             '&& echo EXISTS || echo no'))
    cli.close()


if __name__ == '__main__':
    main()
