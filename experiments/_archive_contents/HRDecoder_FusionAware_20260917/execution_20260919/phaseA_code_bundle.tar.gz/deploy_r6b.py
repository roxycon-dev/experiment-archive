# -*- coding: utf-8 -*-
"""Deploy round 6b (the authorised MA-expert recipe revision).

Use it after the round-6 process has finished and shut the machine down:
boot the instance, then run this.  It probes until the machine answers,
initialises the r6b ledger (2h window), uploads the two new files, compiles
+ runs the full server test suite, runs the smoke, and launches the guarded
run (SCRIPT=r6b_run.py).
"""
import os
import sys
import time

import paramiko

HOST, PORT, PASSWORD = '[REDACTED_USE_ENVIRONMENT]', 19057, 'NKgdrekCg6bo'
PY = '/root/miniconda3/bin/python'
GLM = '/root/autodl-tmp/hrplus/glm'
RUN = GLM + '/runs/r6b'
OTHER_COST = 46.04          # 50 - 2h*1.98  -> a 2h window from the boot
LOCAL_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD = {
    'r6b_expert.py': GLM + '/r6b_expert.py',
    'r6b_run.py': GLM + '/r6b_run.py',
}


def sh(cli, cmd, timeout=900, check=False):
    _, o, e = cli.exec_command(cmd, timeout=timeout)
    out = o.read().decode('utf-8', 'replace')
    err = e.read().decode('utf-8', 'replace')
    rc = o.channel.recv_exit_status()
    if check and rc != 0:
        raise RuntimeError('rc=%d cmd=%s\n%s' % (rc, cmd, err[-1500:]))
    return out, err, rc


def main():
    cli = None
    for i in range(20):
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
        print('STATUS: STILL_DOWN'); sys.exit(2)

    print('== time ==\n%s' % sh(cli, 'date -Is')[0])
    out, _, _ = sh(cli, "ps aux | grep -E 'r6_run|r6b_run|run_guarded' "
                        "| grep -v grep || echo CLEAN")
    print('== processes ==\n%s' % out)
    if 'CLEAN' not in out:
        print('REFUSING: another run is still active'); cli.close(); sys.exit(3)
    print('== df ==\n%s' % sh(cli, 'df -h /root/autodl-tmp | tail -1')[0])
    print('== r6 (main) tail ==\n%s'
          % sh(cli, 'tail -6 %s/runs/r6/r6.log' % GLM)[0])

    sftp = cli.open_sftp()
    for local, remote in UPLOAD.items():
        sftp.put(os.path.join(LOCAL_DIR, local), remote)
        print('uploaded %s' % local)
    sftp.close()
    out, err, rc = sh(cli, 'cd %s && %s -m py_compile r6b_expert.py r6b_run.py '
                      'r6_expert.py r6_lib.py runtime.py && '
                      'bash -n run_guarded.sh && echo OK' % (GLM, PY))
    print('== server compile: %s' % out.strip())
    if rc != 0:
        print(err[-1500:]); sys.exit(3)
    out, err, rc = sh(cli, 'cd %s && %s -m unittest discover -s tests 2>&1'
                      % (GLM, PY), timeout=900)
    print('== server unittest: %s'
          % (out.strip().splitlines()[-1] if out.strip() else '?'))
    if rc != 0 or '\nOK' not in out[-4000:]:
        print(out[-2500:]); sys.exit(3)

    now = sh(cli, 'date -Is')[0].strip()
    sh(cli, 'mkdir -p %s' % RUN)
    out, _, rc = sh(cli, 'test -f %s/budget_ledger.json && cat '
                    '%s/budget_ledger.json' % (RUN, RUN))
    if rc == 0:
        print('== ledger present (kept):\n%s' % out.strip())
    else:
        print('== server now: %s' % now)
        out, _, _ = sh(cli, 'cd %s && %s ops.py init --run-dir %s '
                        '--started-at "%s" --other-cost %s'
                        % (GLM, PY, RUN, now, OTHER_COST))
        print('== ledger init: %s' % out.strip())

    out, err, rc = sh(cli, 'cd %s && %s -u r6b_run.py --settings settings.json '
                      '--run-dir %s --smoke 2>&1' % (GLM, PY, RUN),
                      timeout=1800)
    print('== smoke tail:\n%s' % out[-2000:])
    if rc != 0 or 'SMOKE OK' not in out:
        print('REFUSING: smoke failed (rc=%d)' % rc); cli.close(); sys.exit(3)

    sh(cli, 'cd %s && (SCRIPT=r6b_run.py PYTHON=%s nohup bash run_guarded.sh '
            'settings.json %s %s/backups > guard.log 2>&1 < /dev/null &) ; '
            'echo LAUNCHED' % (GLM, PY, RUN, GLM), timeout=30)
    time.sleep(30)
    out, _, _ = sh(cli, "ps aux | grep -E 'r6b_run|run_guarded|ops.py watchdog'"
                        " | grep -v grep || echo NOT_ALIVE")
    print('== processes after launch\n%s' % out)
    print('== r6b.log tail\n%s' % sh(cli, 'tail -6 %s/r6b.log' % RUN)[0])
    if 'r6b_run' not in out:
        print('REFUSING: r6b_run not alive'); sys.exit(3)
    print('ROUND6B LAUNCHED -- guard + watchdog armed (2h window)')
    cli.close()


if __name__ == '__main__':
    main()
