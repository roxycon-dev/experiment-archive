# -*- coding: utf-8 -*-
"""Final deployment stage: run the GPU smoke in background (nohup) and poll,
then on ALL GATES PASS init the ledger and launch run_guarded.sh."""
import json
import time

import paramiko

HOST, PORT, PASSWORD = '[REDACTED_USE_ENVIRONMENT]', 19057, 'NKgdrekCg6bo'
PY = '/root/miniconda3/bin/python'
GLM = '/root/autodl-tmp/hrplus/glm'
RUN = GLM + '/runs/r1'
BACKUP = GLM + '/backups'
LEDGER_STARTED_AT = '2026-09-17T15:48:11+00:00'
LEDGER_OTHER_COST = 30.2


def sh(cli, cmd, timeout=60, check=True):
    _, stdout, stderr = cli.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode('utf-8', 'replace')
    err = stderr.read().decode('utf-8', 'replace')
    rc = stdout.channel.recv_exit_status()
    if check and rc != 0:
        raise RuntimeError('cmd failed rc=%d: %s\n%s\n%s' % (rc, cmd, out[-2000:], err[-2000:]))
    return out, err, rc


def main():
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli.connect(HOST, port=PORT, username='root', password=PASSWORD,
                timeout=30, banner_timeout=30, auth_timeout=30)

    # fresh smoke run in background
    sh(cli, 'cd %s && rm -f smoke.log smoke.done && nohup %s smoke_gpu.py '
          '> smoke.log 2>&1 & echo PID $!' % (GLM, PY))
    print('smoke launched, polling...', flush=True)

    t0 = time.time()
    passed = False
    while time.time() - t0 < 1500:          # 25 min cap
        time.sleep(45)
        out, _, _ = sh(cli, 'ps aux | grep smoke_gpu | grep -v grep '
                       '| wc -l', check=False)
        alive = out.strip() not in ('0', '')
        log, _, _ = sh(cli, 'tail -c 3000 %s/smoke.log 2>/dev/null' % GLM,
                       check=False)
        print('[%5.0fs alive=%s] %s' % (time.time() - t0, alive,
                                        log.strip().replace('\n', ' | ')[-400:]),
              flush=True)
        if 'ALL GATES PASS' in log:
            passed = True
            break
        if not alive:
            # process gone without pass -> failure; dump full log
            full, _, _ = sh(cli, 'cat %s/smoke.log' % GLM, check=False)
            print('SMOKE DIED:\n%s' % full)
            break
    assert passed, 'smoke did not pass (see log above)'
    full, _, _ = sh(cli, 'cat %s/smoke.log' % GLM)
    print('\n===== SMOKE LOG =====\n%s\n=====================\n' % full)

    # ledger init (idempotent)
    out, _, _ = sh(cli, 'cd %s && %s ops.py init --run-dir %s '
                   '--started-at "%s" --other-cost %s' %
                   (GLM, PY, RUN, LEDGER_STARTED_AT, LEDGER_OTHER_COST))
    print('LEDGER: %s' % out.strip())

    # launch guarded run
    sh(cli, 'cd %s && PYTHON=%s nohup bash run_guarded.sh settings.json '
          '%s %s > guard.log 2>&1 & echo LAUNCHED $!' % (GLM, PY, RUN, BACKUP))
    time.sleep(25)
    out, _, _ = sh(cli, 'ps aux | grep -E "run_protocol|run_guarded|ops.py watchdog" '
                   '| grep -v grep || echo NOT_ALIVE')
    print('## processes\n%s' % out)
    assert 'run_protocol' in out or 'run_guarded' in out, 'guarded run not alive'
    out, _, _ = sh(cli, 'tail -30 %s/protocol.log 2>/dev/null; echo ---GUARD---; '
                   'tail -8 %s/guard.log 2>/dev/null' % (RUN, GLM))
    print('## early logs\n%s' % out)
    print('\nDEPLOYMENT COMPLETE -- guarded protocol running')
    cli.close()


if __name__ == '__main__':
    main()
