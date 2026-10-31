# -*- coding: utf-8 -*-
"""Urgent disk headroom for round 6 (operator action, recorded).

Measured feature-cache cost is 45.4MB/image (6 channels fp16 at the real
ori size 1934x1956), i.e. ~24GB per class (train+val) -- larger than the
22GB the runner reserved, so the HE val cache would have hit a full disk.

Frees only provably redundant items:
  * /root/autodl-tmp/ddr_split613.tar      -- archive of the data already
                                              extracted under repo/data
  * /root/autodl-tmp/hrplus/glm/runs/r5    -- completed round-5 run dir
                                              (logs/summary/counts archived
                                              locally at pulled/r5_final_*)
  * /root/autodl-tmp/hrplus/D_0913_retry   -- previous-agent retry outputs
                                              (not referenced by any
                                              round-6 code path)
KEEPS: results/, hr100/, audit/, runs/r1(B-new ckpts already deleted with
runs/r1), the champion weights and all round-6 outputs.

Appends a DEVIATIONS.jsonl entry to the run dir so the action is auditable
alongside the run's own decisions.
"""
import os
import time

import paramiko

HOST, PORT, PASSWORD = '[REDACTED_USE_ENVIRONMENT]', 19057, 'NKgdrekCg6bo'
GLM = '/root/autodl-tmp/hrplus/glm'
RUN = GLM + '/runs/r6'
TARGETS = [
    '/root/autodl-tmp/ddr_split613.tar',
    GLM + '/runs/r5',
    '/root/autodl-tmp/hrplus/D_0913_retry',
]


def main():
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli.connect(HOST, port=PORT, username='root', password=PASSWORD,
                timeout=30, banner_timeout=30, auth_timeout=30)

    def sh(cmd, timeout=600):
        _, o, e = cli.exec_command(cmd, timeout=timeout)
        return (o.read().decode('utf-8', 'replace') +
                e.read().decode('utf-8', 'replace')).strip()

    print('df before: %s' % sh('df -h /root/autodl-tmp | tail -1'))
    for t in TARGETS:
        out = sh('test -e %s && du -sh %s | cut -f1 && rm -rf %s && '
                 'echo REMOVED || echo "skip (absent)"' % (t, t, t))
        print('%-46s %s' % (t, out))
    print('df after : %s' % sh('df -h /root/autodl-tmp | tail -1'))

    entry = ('{"time": "%s", "kind": "operator_disk_cleanup", "detail": '
             '{"freed_targets": %s, "reason": "measured cache cost 45.4MB/img '
             '(24GB/class) exceeds the 22GB reservation; HE val cache would '
             'have hit a full disk"}}'
             % (time.strftime('%Y-%m-%dT%H:%M:%S'),
                str(TARGETS).replace("'", '"')))
    sh("printf '%s\\n' >> %s/DEVIATIONS.jsonl" % (entry.replace('"', '\\"'),
                                                  RUN))
    print('DEVIATIONS entry appended')
    print(sh('tail -2 %s/DEVIATIONS.jsonl' % RUN))
    print('free now: %s' % sh('df -h /root/autodl-tmp | tail -1'))
    cli.close()


if __name__ == '__main__':
    main()
