# -*- coding: utf-8 -*-
"""Autonomous resume controller (runs LOCALLY, chunked).

Each chunk: probe until the instance is reachable (bounded by MAX_WAIT),
then execute the decision tree:

  1. snapshot evidence (guard.log / watchdog.log preserved as *.ts copies
     BEFORE any relaunch can overwrite them)
  2. round5_summary.json exists        -> pull artifacts, shut down       -> exit 5
  3. traceback in protocol.log  -> pull evidence, HOLD (machine UP,
     no relaunch; agent decides on next turn)                     -> exit 3
  4. ledger deadline passed     -> pull small evidence, shutdown
     (heavy checkpoints stay safe on the AutoDL data disk)        -> exit 4
  5. otherwise                  -> upload restart patch, server
     py_compile + full unit suite, relaunch run_guarded.sh        -> exit 0

Exit codes: 0 relaunched / 2 still down / 3 hold-traceback /
4 salvaged+shutdown / 5 finished+shutdown / 6 sanity failure.
"""
import json
import os
import sys
import time

import paramiko

HOST, PORT, PASSWORD = '[REDACTED_USE_ENVIRONMENT]', 19057, 'NKgdrekCg6bo'
PY = '/root/miniconda3/bin/python'
GLM = '/root/autodl-tmp/hrplus/glm'
RUN = GLM + '/runs/r5'
LOCAL_DIR = os.path.dirname(os.path.abspath(__file__))
PULL_ROOT = os.path.join(LOCAL_DIR, 'pulled')
MAX_WAIT = 540                     # probe window per chunk (seconds)
sys.path.insert(0, LOCAL_DIR)
import ops  # noqa: E402  (local, stdlib-only: ledger math)


def connect():
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli.connect(HOST, port=PORT, username='root', password=PASSWORD,
                timeout=25, banner_timeout=25, auth_timeout=30)
    return cli


def sh(cli, cmd, timeout=120):
    _, o, e = cli.exec_command(cmd, timeout=timeout)
    out = o.read().decode('utf-8', 'replace')
    err = e.read().decode('utf-8', 'replace')
    rc = o.channel.recv_exit_status()
    return out, err, rc


def pull_files(cli, stamp, names):
    """Best-effort SFTP pull of (remote, localname) pairs that may not exist."""
    got = []
    sftp = cli.open_sftp()
    try:
        for remote, localname in names:
            try:
                local = os.path.join(PULL_ROOT, stamp, localname)
                sftp.get(remote, local)
                got.append(localname)
            except Exception:
                pass
    finally:
        sftp.close()
    if got:
        print('pulled -> %s: %s' % (stamp, ', '.join(got)), flush=True)
    return got


def issue_shutdown(cli):
    try:
        _, o, _ = cli.exec_command('shutdown -h now', timeout=8)
        try:
            o.channel.recv_exit_status()
        except Exception:
            pass
        print('shutdown issued', flush=True)
    except Exception as ex:
        print('shutdown channel closed (%r) -- expected while halting'
              % ex, flush=True)


def snapshot_evidence(cli):
    ts = time.strftime('%Y%m%dT%H%M%S')
    sh(cli, 'cd %s && cp guard.log guard.log.%s 2>/dev/null; '
            'cp %s/watchdog.log %s/watchdog.log.%s 2>/dev/null; echo OK'
        % (GLM, ts, RUN, RUN, ts))
    for title, cmd in (
            ('time / uptime', 'date -Is && uptime'),
            ('processes', "ps aux | grep -E 'run_round5|run_guarded|"
                          "ops.py watchdog' | grep -v grep || echo CLEAN"),
            ('protocol.log tail', 'tail -25 %s/protocol.log' % RUN),
            ('guard.log.<ts> (shutdown evidence)',
             'cat %s/guard.log.%s 2>/dev/null || echo NONE' % (GLM, ts)),
            ('backups', 'ls -la %s/backups/ 2>/dev/null || echo NONE' % GLM),
            ('ckpts', 'ls -la %s/ckpt_A_s42 %s/ckpt_B_s42 2>/dev/null'
             % (RUN, RUN)),
            ('ledger', 'cat %s/budget_ledger.json' % RUN),
            ('disk (full-disk crash check)',
             'df -h /root/autodl-tmp | tail -1; du -sh %s/cache/* 2>/dev/null; '
             'du -sh %s/runs/r1 %s/runs/r5 %s/backups 2>/dev/null'
             % (GLM, GLM, GLM, GLM)),
            ('dmesg tail', 'dmesg -T 2>/dev/null | tail -8 || echo none'),
    ):
        print('\n===== %s =====' % title, flush=True)
        print(sh(cli, cmd)[0], flush=True)


def pull_core(cli, stamp):
    os.makedirs(os.path.join(PULL_ROOT, stamp), exist_ok=True)
    names = [
        (RUN + '/protocol.log', 'protocol.log'),
        (GLM + '/guard.log', 'guard.log'),
        (RUN + '/budget_ledger.json', 'budget_ledger.json'),
        (RUN + '/protocol_state.json', 'protocol_state.json'),
        (RUN + '/round5_summary.json', 'summary.json'),
        (RUN + '/test_report.json', 'test_report.json'),
        (RUN + '/selection_lock.json', 'selection_lock.json'),
        (RUN + '/protocol_amendments.jsonl', 'protocol_amendments.jsonl'),
    ]
    # pull the freshest generation of the preserved guard logs too
    out, _, _ = sh(cli, 'ls -t %s/guard.log.* 2>/dev/null | head -1' % GLM)
    g = out.strip()
    if g and g != GLM + '/guard.log':
        names.append((g, os.path.basename(g)))
    return pull_files(cli, stamp, names)


def main():
    stamp = time.strftime('%Y%m%dT%H%M%S')
    cli = None
    t_end = time.time() + MAX_WAIT
    while time.time() < t_end:
        try:
            cli = connect()
            break
        except Exception as ex:
            print('probe down (%s): %r' % (time.strftime('%H:%M:%S'), ex),
                  flush=True)
            time.sleep(45)
    if cli is None:
        print('STATUS: STILL_DOWN', flush=True)
        sys.exit(2)

    try:
        snapshot_evidence(cli)

        # ---- branch 1: protocol already finished ----
        out, _, _ = sh(cli, 'test -f %s/round5_summary.json && echo EXISTS '
                        '|| echo NO' % RUN)
        if 'EXISTS' in out:
            pull_core(cli, stamp)
            print('STATUS: FINISHED -- artifacts pulled; shutting down the '
                  'paid instance', flush=True)
            issue_shutdown(cli)
            sys.exit(5)

        # ---- branch 2: crash evidence -> hold for review ----
        # a python traceback lands on stderr -> guard.log (protocol.log only
        # carries log() lines, so it just ends mid-progress on a crash)
        out, _, _ = sh(cli, 'tail -40 %s/protocol.log; echo ---GUARD---; '
                            'tail -60 %s/guard.log 2>/dev/null' % (RUN, GLM))
        if 'Traceback' in out:
            pull_core(cli, stamp)
            print('STATUS: HOLD_TRACEBACK -- evidence pulled; machine left '
                  'UP, NO relaunch (agent decides next)', flush=True)
            sys.exit(3)

        # ---- branch 3: ledger deadline already passed -> salvage ----
        ledger_local = os.path.join(PULL_ROOT, stamp, 'budget_ledger.json')
        out, _, _ = sh(cli, 'cat %s/budget_ledger.json' % RUN)
        srv_now = int(sh(cli, 'date +%s')[0].strip())
        with open(ledger_local, 'w', encoding='utf-8') as f:
            f.write(out)
        with open(ledger_local, 'r', encoding='utf-8') as f:
            led = json.load(f)
        deadline_epoch = ops.deadline_of(led).timestamp()
        print('server now=%d deadline=%d (%.2fh left)'
              % (srv_now, deadline_epoch,
                 (deadline_epoch - srv_now) / 3600.0), flush=True)
        if srv_now >= deadline_epoch:
            pull_core(cli, stamp)
            print('STATUS: DEADLINE_PASSED -- small evidence pulled to '
                  'pulled/%s; heavy checkpoints remain safe on the AutoDL '
                  'data disk (%s); shutting down' % (stamp, RUN), flush=True)
            issue_shutdown(cli)
            sys.exit(4)

        # ---- branch 4: sanity, deploy patch, relaunch ----
        out, _, _ = sh(cli, "ps aux | grep -E 'run_round5|run_guarded|"
                            "ops.py watchdog' | grep -v grep || echo CLEAN")
        if 'CLEAN' not in out:
            print('STATUS: SANITY_FAIL stale processes:\n%s' % out, flush=True)
            sys.exit(6)
        out, _, _ = sh(cli, 'nvidia-smi --query-gpu=memory.used '
                            '--format=csv,noheader')
        mem = int(out.strip().split()[0])
        if mem > 2000:
            print('STATUS: SANITY_FAIL gpu busy (%d MiB)' % mem, flush=True)
            sys.exit(6)

        sftp = cli.open_sftp()
        sftp.put(os.path.join(LOCAL_DIR, 'run_protocol.py'),
                 GLM + '/run_protocol.py')
        sftp.put(os.path.join(LOCAL_DIR, 'run_round5.py'),
                 GLM + '/run_round5.py')
        sftp.put(os.path.join(LOCAL_DIR, 'tests', 'test_round5.py'),
                 GLM + '/tests/test_round5.py')
        sftp.close()
        print('round5 code re-uploaded (run_round5.py + run_protocol.py)',
              flush=True)

        out, err, rc = sh(cli, 'cd %s && %s -m py_compile run_round5.py '
                          'run_protocol.py tests/test_round5.py '
                          'tests/test_protocol.py && echo COMPILE_OK'
                          % (GLM, PY), timeout=120)
        if rc != 0:
            print('STATUS: SANITY_FAIL server py_compile\n%s' % err[-2000:],
                  flush=True)
            sys.exit(6)
        print('server py_compile OK', flush=True)

        out, err, rc = sh(cli, 'cd %s && %s -m unittest discover -s tests '
                          '2>&1' % (GLM, PY), timeout=900)
        print(out[-2600:], flush=True)
        if rc != 0 or '\nOK' not in out[-4000:]:
            print('STATUS: SANITY_FAIL server unit tests (rc=%d)' % rc,
                  flush=True)
            sys.exit(6)
        print('server unit tests OK', flush=True)

        sh(cli, 'cd %s && (PYTHON=%s SCRIPT=run_round5.py nohup bash run_guarded.sh settings.json '
                '%s %s/backups > guard.log 2>&1 < /dev/null &) ; echo LAUNCHED'
            % (GLM, PY, RUN, GLM), timeout=30)
        time.sleep(25)
        out, _, _ = sh(cli, "ps aux | grep -E 'run_round5|run_guarded|"
                            "ops.py watchdog' | grep -v grep || echo NOT_ALIVE")
        print('\n===== relaunched processes =====', flush=True)
        print(out, flush=True)
        print(sh(cli, 'tail -8 %s/guard.log' % GLM)[0], flush=True)
        print(sh(cli, 'tail -8 %s/protocol.log' % RUN)[0], flush=True)
        if 'run_round5' not in out:
            print('STATUS: SANITY_FAIL run_round5 not alive after relaunch',
                  flush=True)
            sys.exit(6)
        print('STATUS: RELAUNCHED -- guard re-armed, watchdog re-armed, '
              'EXIT trap will back up + shut down at the end', flush=True)
        sys.exit(0)
    finally:
        try:
            cli.close()
        except Exception:
            pass


if __name__ == '__main__':
    main()
