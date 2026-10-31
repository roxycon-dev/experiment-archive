# -*- coding: utf-8 -*-
"""Diagnose-then-relaunch after an instance restart.

Default (no args): DIAGNOSE ONLY -- collect where the protocol stopped,
whether the guard EXIT trap ran (backup+shutdown), checkpoint state, and
kernel/platform clues.  No state is modified.

With --launch: additionally relaunch run_guarded.sh (run_protocol resumes
arm B from its last.pth).  Use only after reviewing the diagnostics.

Exit codes: 0 = OK; 2 = machine unreachable; 3 = sanity failed.
"""
import sys
import time

import paramiko

HOST, PORT, PASSWORD = '[REDACTED_USE_ENVIRONMENT]', 19057, 'NKgdrekCg6bo'
PY = '/root/miniconda3/bin/python'
GLM = '/root/autodl-tmp/hrplus/glm'
RUN = GLM + '/runs/r1'


def connect():
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli.connect(HOST, port=PORT, username='root', password=PASSWORD,
                timeout=25, banner_timeout=25, auth_timeout=30)
    return cli


def sh(cli, cmd, timeout=90):
    _, o, e = cli.exec_command(cmd, timeout=timeout)
    return o.read().decode('utf-8', 'replace').strip()


def main():
    launch = '--launch' in sys.argv
    try:
        cli = connect()
    except Exception as ex:
        print('UNREACHABLE: %r' % ex)
        sys.exit(2)

    def sec(t):
        print('\n===== %s =====' % t)

    sec('time / uptime')
    print(sh(cli, 'date -Is && uptime'))

    sec('processes (stale?)')
    print(sh(cli, "ps aux | grep -E 'run_protocol|run_guarded|ops.py watchdog' "
                  "| grep -v grep || echo CLEAN"))
    print('gpu mem: %s' % sh(cli, 'nvidia-smi --query-gpu=memory.used '
                                  '--format=csv,noheader'))

    sec('ledger + summary')
    print(sh(cli, 'cat %s/budget_ledger.json' % RUN))
    print('summary.json: %s' % sh(cli, 'test -f %s/summary.json && echo EXISTS '
                                       '|| echo no' % RUN))

    sec('protocol.log tail (where it stopped)')
    print(sh(cli, 'tail -20 %s/protocol.log' % RUN))

    sec('guard.log (EXIT trap / backup / shutdown evidence)')
    print(sh(cli, 'tail -25 %s/guard.log 2>/dev/null || echo NO-GUARD-LOG' % GLM))

    sec('watchdog.log')
    print(sh(cli, 'cat %s/watchdog.log 2>/dev/null' % RUN))

    sec('backups dir (did the trap complete a backup?)')
    print(sh(cli, 'ls -la %s/backups/ 2>/dev/null' % GLM))

    sec('B-arm checkpoints (resume point)')
    print(sh(cli, 'ls -la %s/ckpt_B_s42/ 2>/dev/null || echo NO-B-DIR' % RUN))
    print(sh(cli, 'ls -la %s/ckpt_A_s42/ 2>/dev/null' % RUN))

    sec('disk')
    print(sh(cli, 'df -h /root/autodl-tmp | tail -1'))

    sec('kernel messages (OOM / platform kill clues)')
    print(sh(cli, "dmesg -T 2>/dev/null | tail -15 || echo no-dmesg"))

    if not launch:
        print('\nDIAGNOSE ONLY (no changes made).  Review above, then rerun '
              'with --launch to resume the protocol.')
        cli.close()
        return

    # ---------------- relaunch ----------------
    stale = sh(cli, "ps aux | grep -E 'run_protocol|run_guarded|ops.py watchdog' "
                    "| grep -v grep || echo CLEAN")
    if 'CLEAN' not in stale:
        print('STALE PROCESSES -- refusing to double-launch')
        sys.exit(3)
    if sh(cli, 'test -f %s/summary.json && echo EXISTS' % RUN) == 'EXISTS':
        print('PROTOCOL ALREADY FINISHED -- not relaunching')
        sys.exit(3)

    sh(cli, 'cd %s && (PYTHON=%s nohup bash run_guarded.sh settings.json %s '
            '%s/backups > guard.log 2>&1 < /dev/null &) ; echo LAUNCHED'
        % (GLM, PY, RUN, GLM), timeout=30)
    time.sleep(20)
    procs = sh(cli, "ps aux | grep -E 'run_protocol|run_guarded|ops.py watchdog' "
                    "| grep -v grep || echo NOT_ALIVE")
    sec('relaunched processes')
    print(procs)
    print(sh(cli, 'tail -6 %s/guard.log' % GLM))
    print(sh(cli, 'tail -6 %s/protocol.log' % RUN))
    if 'run_protocol' not in procs:
        print('RELAY: run_protocol NOT alive after relaunch')
        sys.exit(3)
    print('RELAUNCH OK')
    cli.close()


if __name__ == '__main__':
    main()
