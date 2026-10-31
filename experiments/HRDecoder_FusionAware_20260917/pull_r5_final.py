# -*- coding: utf-8 -*-
"""Pull the round-5 final artifacts BEFORE any disk cleanup removes the
backup tar that contains them.  Read-only on the server."""
import os
import time

import paramiko

HOST, PORT, PASSWORD = '[REDACTED_USE_ENVIRONMENT]', 19057, 'NKgdrekCg6bo'
GLM = '/root/autodl-tmp/hrplus/glm'
RUN = GLM + '/runs/r5'
LOCAL = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pulled',
                     'r5_final_' + time.strftime('%Y%m%dT%H%M%S'))

FILES = [
    (RUN + '/round5_summary.json', 'round5_summary.json'),
    (RUN + '/round5_state.json', 'round5_state.json'),
    (RUN + '/protocol.log', 'protocol.log'),
    (RUN + '/DEVIATIONS.jsonl', 'DEVIATIONS.jsonl'),
    (RUN + '/budget_ledger.json', 'budget_ledger.json'),
    (GLM + '/guard.log', 'guard.log'),
]


def main():
    os.makedirs(LOCAL, exist_ok=True)
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli.connect(HOST, port=PORT, username='root', password=PASSWORD,
                timeout=30, banner_timeout=30, auth_timeout=30)
    sftp = cli.open_sftp()
    for remote, name in FILES:
        try:
            sftp.get(remote, os.path.join(LOCAL, name))
            print('pulled %s' % name)
        except Exception as ex:
            print('MISSING %s (%r)' % (name, ex))
    sftp.close()
    _, o, _ = cli.exec_command(
        'ls -la %s; echo ---BACKUPS---; ls -la %s/backups; echo ---DF---; '
        'df -h /root/autodl-tmp | tail -1; echo ---R5DIR---; du -sh %s'
        % (RUN, GLM, RUN), timeout=120)
    listing = o.read().decode('utf-8', 'replace')
    with open(os.path.join(LOCAL, 'server_listing.txt'), 'w',
              encoding='utf-8') as f:
        f.write(listing)
    print(listing)
    cli.close()
    print('\nARTIFACTS PULLED -> %s' % LOCAL)


if __name__ == '__main__':
    main()
