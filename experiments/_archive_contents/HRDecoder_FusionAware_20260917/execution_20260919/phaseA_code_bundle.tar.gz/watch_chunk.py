# -*- coding: utf-8 -*-
"""Bounded monitoring chunk for the resumed protocol (runs LOCALLY).

Loops ~8 minutes: every POLL seconds snapshot protocol.log (and pull it as
insurance against a post-run shutdown), report new milestone lines, check
liveness.  Breaks early on:
  - summary.json exists  -> pull all artifacts, STATUS COMPLETE
  - machine unreachable  -> STATUS UNREACHABLE (normal after the guard's
                            EXIT trap shuts down at run end)
  - protocol dead w/o summary -> pull evidence, STATUS DEAD_NO_SUMMARY

Exit codes: 0 running/complete / 2 unreachable / 3 dead-without-summary.
"""
import os
import sys
import time

import paramiko

HOST, PORT, PASSWORD = '[REDACTED_USE_ENVIRONMENT]', 19057, 'NKgdrekCg6bo'
GLM = '/root/autodl-tmp/hrplus/glm'
RUN = GLM + '/runs/r5'
LOCAL_DIR = os.path.dirname(os.path.abspath(__file__))
PULL_ROOT = os.path.join(LOCAL_DIR, 'pulled')
POLL = 40
MAX_ITER = 13                      # ~8.7 minutes per chunk
STAMP = time.strftime('%Y%m%dT%H%M%S')

MILESTONES = ('E0 ', 'G0 pass', 'E0 test confirm', 'E1 winner', 'G1 pass',
              'E2 ', 'E3 ', 'G3 pass', 'E3 locked', 'ROUND5 COMPLETE',
              'DEVIATION', 'shutdown-guard', 'resumed at iter',
              'BNEW', 'champion val', 's42 iter', 's43 iter')


def connect():
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli.connect(HOST, port=PORT, username='root', password=PASSWORD,
                timeout=25, banner_timeout=25, auth_timeout=30)
    return cli


def sh(cli, cmd, timeout=90):
    _, o, e = cli.exec_command(cmd, timeout=timeout)
    out = o.read().decode('utf-8', 'replace')
    err = e.read().decode('utf-8', 'replace')
    rc = o.channel.recv_exit_status()
    return out, err, rc


def pull_files(cli, names):
    got = []
    sftp = cli.open_sftp()
    try:
        for remote, local in names:
            try:
                sftp.get(remote, local)
                got.append(os.path.basename(local))
            except Exception:
                pass
    finally:
        sftp.close()
    if got:
        print('pulled: %s' % ', '.join(got), flush=True)


def main():
    stamp = STAMP
    os.makedirs(os.path.join(PULL_ROOT, stamp), exist_ok=True)
    seen = set()
    cli = None
    for it in range(MAX_ITER):
        try:
            if cli is None:
                cli = connect()
            # live protocol.log snapshot + insurance pull
            out, _, _ = sh(cli, 'cat %s/protocol.log' % RUN, timeout=120)
            local_log = os.path.join(PULL_ROOT, stamp, 'protocol.log')
            with open(local_log, 'w', encoding='utf-8') as f:
                f.write(out)
            for line in out.splitlines():
                for m in MILESTONES:
                    if m in line and line not in seen:
                        seen.add(line)
                        print('MILESTONE %s' % line, flush=True)
            # summary / test report as soon as they appear
            out, _, _ = sh(cli, 'test -f %s/round5_summary.json && echo YES '
                           '|| echo NO' % RUN)
            if 'YES' in out:
                pull_files(cli, [
                    (RUN + '/round5_summary.json',
                     os.path.join(PULL_ROOT, stamp, 'round5_summary.json')),
                    (RUN + '/round5_state.json',
                     os.path.join(PULL_ROOT, stamp, 'round5_state.json')),
                    (RUN + '/protocol.log',
                     os.path.join(PULL_ROOT, stamp, 'protocol.log')),
                    (RUN + '/DEVIATIONS.jsonl',
                     os.path.join(PULL_ROOT, stamp, 'DEVIATIONS.jsonl')),
                ])
                print('STATUS: COMPLETE -- artifacts in pulled/%s' % stamp,
                      flush=True)
                sys.exit(0)
            # liveness
            out, _, _ = sh(cli, "ps aux | grep -E 'run_round5|run_guarded' "
                                "| grep -v grep || echo DEAD")
            if 'DEAD' in out:
                # guard may be in its EXIT-trap backup phase: run_guarded
                # alive is enough; only DEAD if BOTH are gone
                out2, _, _ = sh(cli, "ps aux | grep -E 'ops.py watchdog' "
                                     "| grep -v grep || echo DEAD2")
                if 'DEAD2' in out2:
                    pull_files(cli, [(GLM + '/guard.log',
                                      os.path.join(PULL_ROOT, stamp,
                                                   'guard.log'))])
                    tail, _, _ = sh(cli, 'tail -15 %s/protocol.log' % RUN)
                    print('protocol died without summary; tail:\n%s' % tail,
                          flush=True)
                    print('STATUS: DEAD_NO_SUMMARY', flush=True)
                    sys.exit(3)
            print('tick %d: running (%s)' %
                  (it, time.strftime('%H:%M:%S')), flush=True)
            if it % 4 == 0:            # periodic disk-watermark check
                d, _, _ = sh(cli, 'df -h /root/autodl-tmp | tail -1')
                print('   disk: %s' % d.strip(), flush=True)
            time.sleep(POLL)
        except paramiko.SSHException as ex:
            print('ssh dropped (%r); reconnecting' % ex, flush=True)
            try:
                cli.close()
            except Exception:
                pass
            cli = None
            time.sleep(20)
        except OSError as ex:
            print('unreachable (%r) -- likely post-run shutdown' % ex,
                  flush=True)
            sys.exit(2)
    print('STATUS: RUNNING (chunk elapsed)', flush=True)
    sys.exit(0)


if __name__ == '__main__':
    try:
        main()
    finally:
        print('latest evidence dir: pulled/%s' % STAMP, flush=True)
