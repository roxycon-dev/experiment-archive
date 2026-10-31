# -*- coding: utf-8 -*-
"""Final round-6/6b evidence collection, then shut the instance down.

Pulls every artefact worth keeping (summaries, states, logs, deviations,
per-class count JSONs, the trained head and both expert checkpoints -- all
small), prints the tail of the r6b log so the alpha-sweep outcome and the
failure mode are visible, then powers the machine off to stop billing.

No experiments, no code changes: collection only.
"""
import os
import sys
import time

import paramiko

HOST, PORT, PASSWORD = '[REDACTED_USE_ENVIRONMENT]', 19057, 'NKgdrekCg6bo'
GLM = '/root/autodl-tmp/hrplus/glm'
LOCAL_DIR = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(LOCAL_DIR, 'pulled', 'final_' + time.strftime('%Y%m%dT%H%M%S'))

FILES = [
    (GLM + '/runs/r6/r6_summary.json', 'r6_summary.json'),
    (GLM + '/runs/r6/r6_state.json', 'r6_state.json'),
    (GLM + '/runs/r6/r6.log', 'r6.log'),
    (GLM + '/runs/r6/DEVIATIONS.jsonl', 'r6_DEVIATIONS.jsonl'),
    (GLM + '/runs/r6/p0_report.json', 'p0_report.json'),
    (GLM + '/runs/r6/p0_champ_val_counts.json', 'p0_champ_val_counts.json'),
    (GLM + '/runs/r6/p0_champ_test_counts.json', 'p0_champ_test_counts.json'),
    (GLM + '/runs/r6/heads/head_HE.pth', 'head_HE.pth'),
    (GLM + '/runs/r6/heads/head_HE_val_counts.json', 'head_HE_val_counts.json'),
    (GLM + '/runs/r6/expert/ma_expert.pth', 'ma_expert_r6.pth'),
    (GLM + '/runs/r6/expert/expert_curve.json', 'expert_curve_r6.json'),
    (GLM + '/runs/r6/expert/alpha_sweep.json', 'alpha_sweep_r6.json'),
    (GLM + '/runs/r6b/r6b_summary.json', 'r6b_summary.json'),
    (GLM + '/runs/r6b/r6b_state.json', 'r6b_state.json'),
    (GLM + '/runs/r6b/r6b.log', 'r6b.log'),
    (GLM + '/runs/r6b/DEVIATIONS.jsonl', 'r6b_DEVIATIONS.jsonl'),
    (GLM + '/runs/r6b/expert/ma_expert_r6b.pth', 'ma_expert_r6b.pth'),
    (GLM + '/runs/r6b/expert/r6b_curve.json', 'r6b_curve.json'),
    (GLM + '/guard.log', 'guard.log'),
    (GLM + '/settings.json', 'settings.json'),
]


def main():
    do_shutdown = '--no-shutdown' not in sys.argv
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
            time.sleep(20)
    if cli is None:
        print('STATUS: STILL_DOWN'); sys.exit(2)

    def sh(cmd, timeout=300):
        _, o, e = cli.exec_command(cmd, timeout=timeout)
        return (o.read().decode('utf-8', 'replace') +
                e.read().decode('utf-8', 'replace')).strip()

    print('== time ==\n%s' % sh('date -Is'))
    print('== r6b.log tail (alpha sweep / failure evidence) ==')
    print(sh('tail -25 %s/runs/r6b/r6b.log' % GLM))
    print('== guard.log tail ==')
    print(sh('tail -12 %s/guard.log' % GLM))
    print('== r6b summary present? ==')
    print(sh('test -f %s/runs/r6b/r6b_summary.json && cat '
             '%s/runs/r6b/r6b_summary.json | head -40 || echo MISSING'
             % (GLM, GLM)))

    os.makedirs(OUT, exist_ok=True)
    sftp = cli.open_sftp()
    got, missed = 0, []
    for remote, name in FILES:
        try:
            sftp.get(remote, os.path.join(OUT, name))
            got += 1
        except Exception as ex:
            missed.append('%s (%s)' % (name, ex))
    sftp.close()
    print('\npulled %d/%d files -> %s' % (got, len(FILES), OUT))
    if missed:
        print('not present: %s' % missed)
    print('== df ==\n%s' % sh('df -h /root/autodl-tmp | tail -1'))

    if do_shutdown:
        print('== shutting down to stop billing ==')
        print(sh('shutdown -h now', timeout=30) or '(shutdown issued)')
    cli.close()
    print('DONE')


if __name__ == '__main__':
    main()
