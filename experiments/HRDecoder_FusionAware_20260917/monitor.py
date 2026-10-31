# -*- coding: utf-8 -*-
"""Training monitor: one-shot status pull (run on demand, ~10min cadence).

Exit codes tell the wrapper what state we are in:
  0  = protocol still running
  1  = protocol finished (summary.json exists) but machine still up
  2  = SSH unreachable (machine shut down -- task complete)
"""
import sys

import paramiko

HOST, PORT, PASSWORD = '[REDACTED_USE_ENVIRONMENT]', 19057, 'NKgdrekCg6bo'
PY = '/root/miniconda3/bin/python'
GLM = '/root/autodl-tmp/hrplus/glm'
RUN = GLM + '/runs/r1'


def main():
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        cli.connect(HOST, port=PORT, username='root', password=PASSWORD,
                    timeout=25, banner_timeout=25, auth_timeout=25)
    except Exception as e:
        print('SSH UNREACHABLE (%r) -> machine is OFF; task complete' % e)
        sys.exit(2)

    def sh(cmd, timeout=60):
        _, o, e = cli.exec_command(cmd, timeout=timeout)
        return o.read().decode('utf-8', 'replace').strip()

    print('## time on server\n%s' % sh('date -Is'))
    print('## gpu\n%s' % sh('nvidia-smi --query-gpu=memory.used,utilization.gpu '
                            '--format=csv,noheader'))
    print('## disk (autodl-tmp)\n%s' % sh("df -h /root/autodl-tmp"))
    print('## processes')
    print(sh('ps aux | grep -E "run_protocol|run_guarded|watchdog" '
             '| grep -v grep || echo NONE-RUNNING'))
    print('## ledger')
    print(sh('cat %s/budget_ledger.json 2>/dev/null' % RUN))
    print('## protocol.log (tail 45)')
    print(sh('tail -45 %s/protocol.log 2>/dev/null || echo NO-LOG-YET' % RUN))
    print('## run dir')
    print(sh('ls -la %s/ 2>/dev/null' % RUN))
    print('## key results')
    print(sh('cat %s/summary.json 2>/dev/null || echo no-summary-yet' % RUN))
    print(sh('cat %s/test_report.json 2>/dev/null | head -60 || true' % RUN))
    print(sh('cat %s/guard.log 2>/dev/null | tail -12 || true' % GLM))
    print('## watchdog log')
    print(sh('tail -6 %s/watchdog.log 2>/dev/null || true' % RUN))
    finished = sh('test -f %s/summary.json && echo YES || echo NO' % RUN)
    cli.close()
    sys.exit(1 if finished == 'YES' else 0)


if __name__ == '__main__':
    main()
