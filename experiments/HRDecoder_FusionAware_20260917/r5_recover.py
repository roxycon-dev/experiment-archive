# -*- coding: utf-8 -*-
"""Round-5 crash recovery: (1) read V1@3000 val_history from the complete
iter_3000.pth, (2) free disk space by removing REDUNDANT artifacts only,
(3) report df.  Deletions are limited to: run-r5 partial tmp file, the
redundant r1 backup tar (live runs/r1 still exists), the truncated 10:00
tar, r1's A-arm checkpoints + pilots + B iter_1000/2000/last (B iter_3000/
4000 are KEPT: needed by round-5 E0/E3)."""
import paramiko

HOST, PORT, PASSWORD = '[REDACTED_USE_ENVIRONMENT]', 19057, 'NKgdrekCg6bo'
PY = '/root/miniconda3/bin/python'
GLM = '/root/autodl-tmp/hrplus/glm'
R5 = GLM + '/runs/r5'
R1 = GLM + '/runs/r1'

DELETE = [
    R5 + '/ckpt_V1_s42/last.pth.tmp',                 # 0.6G partial write
    GLM + '/backups/run_20260918T032508.tar',         # 8.6G r1 backup
    GLM + '/backups/run_20260918T100020.tar',         # 320K truncated
    R1 + '/ckpt_A_s42/iter_1000.pth',                 # A-arm, recorded
    R1 + '/ckpt_A_s42/iter_2000.pth',
    R1 + '/ckpt_A_s42/iter_3000.pth',
    R1 + '/ckpt_A_s42/iter_4000.pth',
    R1 + '/ckpt_A_s42/last.pth',
    R1 + '/pilot_A.pth',
    R1 + '/pilot_B.pth',
    R1 + '/ckpt_B_s42/iter_1000.pth',
    R1 + '/ckpt_B_s42/iter_2000.pth',
    R1 + '/ckpt_B_s42/last.pth',
]
KEEP_NOTE = ('KEPT: r1/ckpt_B_s42/iter_3000.pth + iter_4000.pth (BNEW '
             'ensemble members), all r5 ckpts, all logs/state/counts')


def main():
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli.connect(HOST, port=PORT, username='root', password=PASSWORD,
                timeout=30, banner_timeout=30, auth_timeout=30)

    def sh(cmd, timeout=180):
        _, o, e = cli.exec_command(cmd, timeout=timeout)
        return (o.read().decode('utf-8', 'replace') +
                e.read().decode('utf-8', 'replace')).strip()

    print('===== df BEFORE =====')
    print(sh('df -h /root/autodl-tmp | tail -1'))

    print('\n===== V1@3000 val_history (from complete iter_3000.pth) =====')
    out = sh('cd %s && %s -c "import torch; '
             'ck=torch.load(\'%s/ckpt_V1_s42/iter_3000.pth\', '
             'map_location=\'cpu\', weights_only=False); '
             'print(\'iter\', ck.get(\'iter\'), \'seed\', ck.get(\'seed\'), '
             '\'arm\', ck.get(\'arm\')); '
             'print(\'val_history:\', ck.get(\'val_history\'))"'
             % (GLM, PY, R5), timeout=300)
    print(out)

    print('\n===== deletions =====')
    for p in DELETE:
        out = sh("test -f '%s' && du -h '%s' | cut -f1 && rm -f '%s' "
                 "&& echo DELETED || echo 'skip (absent)'" % (p, p, p))
        print('%-70s %s' % (p, out))
    print(KEEP_NOTE)

    print('\n===== df AFTER =====')
    print(sh('df -h /root/autodl-tmp | tail -1'))
    print(sh('du -sh %s/backups %s/runs/r1 %s/runs/r5 2>/dev/null'
             % (GLM, GLM, GLM)))
    cli.close()


if __name__ == '__main__':
    main()
