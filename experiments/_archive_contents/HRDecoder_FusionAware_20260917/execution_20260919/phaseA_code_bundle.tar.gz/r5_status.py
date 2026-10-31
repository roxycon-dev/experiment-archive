# -*- coding: utf-8 -*-
"""Compact round-5 status probe: current time, V2 checkpoint freshness,
last protocol lines, round5_state summary."""
import paramiko

cli = paramiko.SSHClient()
cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
cli.connect('connect.nmb1.seetacloud.com', port=19057, username='root',
            password='[REDACTED_USE_ENVIRONMENT]', timeout=25, banner_timeout=25,
            auth_timeout=30)
cmds = [
    'date "+now %H:%M:%S"',
    'ls --time-style=+%H:%M:%S -la /root/autodl-tmp/hrplus/glm/runs/r5/'
    'ckpt_V2_s42/ 2>/dev/null | tail -4',
    'tail -5 /root/autodl-tmp/hrplus/glm/runs/r5/protocol.log',
    'df -h /root/autodl-tmp | tail -1',
    'ps -o etime= -p $(pgrep -f "run_round5.py --settings" | tail -1)',
]
for c in cmds:
    _, o, e = cli.exec_command(c, timeout=60)
    print('--- %s' % c)
    print((o.read().decode('utf-8', 'replace') +
           e.read().decode('utf-8', 'replace')).strip())
cli.close()
