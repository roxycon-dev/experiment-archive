# -*- coding: utf-8 -*-
"""Round-6 deployment: upload code, server compile + full unit suite,
initialise the r6 ledger (6h from boot = the user's stated balance), run the
GPU smoke, then launch run_guarded.sh with SCRIPT=r6_run.py.  ANY failure
aborts before launch (nothing is started half-checked).

Ledger math: ops.deadline_of = start + (BUDGET50 - other_cost)/RATE1.98, so
other_cost = 50 - 6*1.98 = 38.12 gives a hard 6h window from started_at.
"""
import sys
import time

import paramiko

HOST, PORT, PASSWORD = '[REDACTED_USE_ENVIRONMENT]', 19057, 'NKgdrekCg6bo'
PY = '/root/miniconda3/bin/python'
GLM = '/root/autodl-tmp/hrplus/glm'
RUN = GLM + '/runs/r6'
OTHER_COST = 38.12               # 50 - 6h * 1.98  -> 6h deadline
LOCAL = {
    'r6_lib.py': GLM + '/r6_lib.py',
    'r6_features.py': GLM + '/r6_features.py',
    'r6_head.py': GLM + '/r6_head.py',
    'r6_expert.py': GLM + '/r6_expert.py',
    'r6_run.py': GLM + '/r6_run.py',
    'runtime.py': GLM + '/runtime.py',
    'run_guarded.sh': GLM + '/run_guarded.sh',
    'tests/test_r6.py': GLM + '/tests/test_r6.py',
}


def sh(cli, cmd, timeout=600, check=True):
    _, o, e = cli.exec_command(cmd, timeout=timeout)
    out = o.read().decode('utf-8', 'replace')
    err = e.read().decode('utf-8', 'replace')
    rc = o.channel.recv_exit_status()
    if check and rc != 0:
        raise RuntimeError('cmd failed rc=%d: %s\nOUT:%s\nERR:%s'
                           % (rc, cmd, out[-3000:], err[-3000:]))
    return out, err, rc


def main():
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli.connect(HOST, port=PORT, username='root', password=PASSWORD,
                timeout=30, banner_timeout=30, auth_timeout=30)
    out, _, _ = sh(cli, "ps aux | grep -E 'run_protocol|run_round5|r6_run|"
                        "run_guarded|ops.py watchdog' | grep -v grep "
                        "|| echo CLEAN", check=False)
    print('## stale processes: %s' % out.strip())
    if 'CLEAN' not in out:
        print('REFUSING: stale processes'); sys.exit(3)
    out, _, _ = sh(cli, 'nvidia-smi --query-gpu=memory.used '
                        '--format=csv,noheader')
    mem = int(out.strip().split()[0])
    print('## gpu mem used: %d MiB' % mem)
    if mem > 2000:
        print('REFUSING: GPU busy'); sys.exit(3)
    print('## df: %s' % sh(cli, 'df -h /root/autodl-tmp | tail -1')[0].strip())
    for p in (GLM + '/settings.json',
              GLM + '/runs/r5/round5_summary.json'):
        out, _, rc = sh(cli, 'test -f %s && echo OK' % p, check=False)
        if rc != 0:
            print('note: missing %s (continuing)' % p)

    sftp = cli.open_sftp()
    for local, remote in LOCAL.items():
        sftp.put(local, remote)
        print('uploaded %s' % local)
    sftp.close()

    out, _, rc = sh(cli, 'cd %s && %s -m py_compile r6_lib.py r6_features.py '
                    'r6_head.py r6_expert.py r6_run.py runtime.py '
                    'tests/test_r6.py && bash -n run_guarded.sh && '
                    'echo COMPILE_OK' % (GLM, PY))
    print('## server compile: %s' % out.strip())
    out, err, rc = sh(cli, 'cd %s && %s -m unittest discover -s tests 2>&1'
                      % (GLM, PY), check=False, timeout=900)
    tail = out.strip().splitlines()[-1] if out.strip() else '(empty)'
    print('## server unittest: %s' % tail)
    if rc != 0 or '\nOK' not in out[-4000:]:
        print(out[-3000:]); print('REFUSING: server tests failed'); sys.exit(3)

    now = sh(cli, 'date -Is')[0].strip()
    print('## server now: %s' % now)
    sh(cli, 'mkdir -p %s' % RUN)
    out, _, rc = sh(cli, 'test -f %s/budget_ledger.json && cat %s/budget_ledger.json'
                    % (RUN, RUN), check=False)
    if rc == 0:
        print('## ledger already present (preserved; started_at immutable):\n%s'
              % out.strip())
        probe = ('cd %s && %s -c "import ops; '
                 'print(ops.deadline_of(ops.load_ledger(\'%s\')).isoformat())"'
                 % (GLM, PY, RUN))
        print('## deadline: %s' % sh(cli, probe)[0].strip())
    else:
        out, _, _ = sh(cli, 'cd %s && %s ops.py init --run-dir %s '
                        '--started-at "%s" --other-cost %s'
                        % (GLM, PY, RUN, now, OTHER_COST))
        print('## ledger init: %s' % out.strip())

    out, err, rc = sh(cli, 'cd %s && %s -u r6_run.py --settings settings.json '
                      '--run-dir %s --smoke 2>&1' % (GLM, PY, RUN),
                      check=False, timeout=1800)
    print('## smoke tail:\n%s' % out[-2500:])
    if rc != 0 or 'SMOKE OK' not in out:
        print('REFUSING: smoke failed (rc=%d)' % rc); sys.exit(3)

    sh(cli, 'cd %s && (SCRIPT=r6_run.py PYTHON=%s nohup bash run_guarded.sh '
            'settings.json %s %s/backups > guard.log 2>&1 < /dev/null &) ; '
            'echo LAUNCHED' % (GLM, PY, RUN, GLM), timeout=30)
    time.sleep(30)
    out, _, _ = sh(cli, "ps aux | grep -E 'r6_run|run_guarded|ops.py watchdog' "
                        "| grep -v grep || echo NOT_ALIVE")
    print('## processes after launch\n%s' % out)
    print(sh(cli, 'tail -6 %s/guard.log' % GLM)[0])
    print(sh(cli, 'tail -6 %s/r6.log' % RUN)[0])
    if 'r6_run' not in out:
        print('REFUSING: r6_run not alive'); sys.exit(3)
    print('ROUND6 LAUNCHED -- guard + watchdog armed, 6h ledger window')
    cli.close()


if __name__ == '__main__':
    main()
