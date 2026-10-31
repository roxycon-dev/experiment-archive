# -*- coding: utf-8 -*-
"""Round-5 deployment: upload code, server-side compile + full unit suite,
initialize the r5 ledger (series cap 50 yuan; prior spend folded into
other_cost), run the GPU smoke, then launch run_guarded.sh with
SCRIPT=run_round5.py.  Any failure aborts BEFORE launch."""
import sys
import time

import paramiko

HOST, PORT, PASSWORD = '[REDACTED_USE_ENVIRONMENT]', 19057, 'NKgdrekCg6bo'
PY = '/root/miniconda3/bin/python'
GLM = '/root/autodl-tmp/hrplus/glm'
RUN = GLM + '/runs/r5'
BOOT_UTC = '2026-09-18T00:26:00+00:00'   # container boot (first reachable
                                          # probe 08:26:47 +08; conservative)
OTHER_COST = 37.4                        # 30.2 prior series + 7.2 r1 container
LOCAL = {
    'run_round5.py': GLM + '/run_round5.py',
    'run_protocol.py': GLM + '/run_protocol.py',
    'tests/test_round5.py': GLM + '/tests/test_round5.py',
    'run_guarded.sh': GLM + '/run_guarded.sh',
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

    # ---- 0. sanity: no stale protocol, GPU free, r1 assets present ----
    out, _, _ = sh(cli, "ps aux | grep -E 'run_protocol|run_round5|"
                        "run_guarded|ops.py watchdog' | grep -v grep "
                        "|| echo CLEAN", check=False)
    print('## stale processes\n%s' % out.strip())
    if 'CLEAN' not in out:
        print('REFUSING: stale processes'); sys.exit(3)
    out, _, _ = sh(cli, 'nvidia-smi --query-gpu=memory.used --format=csv,noheader')
    mem = int(out.strip().split()[0])
    print('## gpu mem used: %d MiB' % mem)
    if mem > 2000:
        print('REFUSING: GPU busy'); sys.exit(3)
    for p in ('/root/autodl-tmp/hrplus/glm/runs/r1/ckpt_B_s42/iter_3000.pth',
              '/root/autodl-tmp/hrplus/glm/runs/r1/ckpt_B_s42/iter_4000.pth',
              GLM + '/settings.json'):
        out, _, rc = sh(cli, 'test -f %s && echo OK' % p, check=False)
        if rc != 0:
            print('REFUSING: missing %s' % p); sys.exit(3)
    print('r1 assets present (BNEW ckpts + settings.json)')

    # ---- 1. upload ----
    sftp = cli.open_sftp()
    for local, remote in LOCAL.items():
        sftp.put(local, remote)
        print('uploaded %s' % local)
    sftp.close()

    # ---- 2. server compile + full unit suite ----
    out, err, rc = sh(cli, 'cd %s && %s -m py_compile run_round5.py '
                      'run_protocol.py tests/test_round5.py && '
                      'bash -n run_guarded.sh && echo COMPILE_OK' % (GLM, PY))
    print('## server compile: %s' % out.strip())
    out, err, rc = sh(cli, 'cd %s && %s -m unittest discover -s tests 2>&1'
                      % (GLM, PY), check=False, timeout=900)
    print('## server unittest: %s' % out.strip().splitlines()[-1])
    if rc != 0 or '\nOK' not in out[-4000:]:
        print(out[-3000:]); print('REFUSING: server tests failed'); sys.exit(3)

    # ---- 3. ledger (fresh r5; series cap 50 -> deadline ~14:48) ----
    out, _, _ = sh(cli, 'mkdir -p %s && cd %s && %s ops.py init --run-dir %s '
                    '--started-at "%s" --other-cost %s'
                    % (RUN, GLM, PY, RUN, BOOT_UTC, OTHER_COST))
    print('## ledger init: %s' % out.strip())

    # ---- 4. GPU smoke (champion+BNEW predict; one V1/V2 step) ----
    out, err, rc = sh(cli, 'cd %s && %s -u run_round5.py --settings '
                      'settings.json --run-dir %s --smoke 2>&1'
                      % (GLM, PY, RUN), check=False, timeout=1200)
    print('## smoke:\n%s' % out[-3000:])
    if rc != 0 or 'SMOKE OK' not in out:
        print('REFUSING: smoke failed (rc=%d)' % rc); sys.exit(3)

    # ---- 5. launch guarded (subshell-wrapped to free the channel) ----
    sh(cli, 'cd %s && (SCRIPT=run_round5.py PYTHON=%s nohup bash '
            'run_guarded.sh settings.json %s %s/backups > guard.log 2>&1 '
            '< /dev/null &) ; echo LAUNCHED' % (GLM, PY, RUN, GLM),
       timeout=30)
    time.sleep(25)
    out, _, _ = sh(cli, "ps aux | grep -E 'run_round5|run_guarded|"
                        "ops.py watchdog' | grep -v grep || echo NOT_ALIVE")
    print('## relaunched processes\n%s' % out)
    print(sh(cli, 'tail -6 %s/guard.log' % GLM)[0])
    print(sh(cli, 'tail -6 %s/protocol.log' % RUN)[0])
    if 'run_round5' not in out:
        print('REFUSING: run_round5 not alive after launch'); sys.exit(3)
    print('ROUND5 LAUNCHED -- guard + watchdog armed')
    cli.close()


if __name__ == '__main__':
    main()
