# -*- coding: utf-8 -*-
"""Fetch GLM's h5_ensemble_corrector.py (ground-truth champion implementation)
and eval_ens.py to see exact member inference semantics."""
import paramiko

HOST, PORT, PASSWORD = '[REDACTED_USE_ENVIRONMENT]', 19057, 'NKgdrekCg6bo'
FILES = [
    '/root/autodl-tmp/hrplus/ops/h5_ensemble_corrector.py',
    '/root/autodl-tmp/hrplus/ops/eval_ens.py',
]


def main():
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli.connect(HOST, port=PORT, username='root', password=PASSWORD,
                timeout=25, banner_timeout=25, auth_timeout=25)
    sftp = cli.open_sftp()
    for f in FILES:
        local = 'glm_ref_' + f.rsplit('/', 1)[-1]
        try:
            sftp.get(f, local)
            print('downloaded %s -> %s' % (f, local))
        except Exception as e:
            print('FAILED %s: %r' % (f, e))
    sftp.close()
    cli.close()


if __name__ == '__main__':
    main()
