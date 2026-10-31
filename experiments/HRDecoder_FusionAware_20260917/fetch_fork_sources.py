# -*- coding: utf-8 -*-
"""Fetch the fork's segmentor/model sources to pin down forward/inference/
_student_prob semantics (needed to fix member_prob and validate d_inf)."""
import paramiko

HOST, PORT, PASSWORD = '[REDACTED_USE_ENVIRONMENT]', 19057, 'NKgdrekCg6bo'
FILES = [
    ('/root/autodl-tmp/hrplus/repo/mmseg/models/segmentors/lesion_encoder_decoder.py',
     'fork_lesion_encoder_decoder.py'),
    ('/root/autodl-tmp/hrplus/repo/hrplus_patch.py', 'fork_hrplus_patch.py'),
    ('/root/autodl-tmp/hrplus/repo/configs/lesion/exp/hrplus_patch_dice.py',
     'fork_cfg_hrplus_patch_dice.py'),
]


def main():
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli.connect(HOST, port=PORT, username='root', password=PASSWORD,
                timeout=25, banner_timeout=25, auth_timeout=25)
    sftp = cli.open_sftp()
    for remote, local in FILES:
        try:
            sftp.get(remote, local)
            print('downloaded %s -> %s' % (remote, local))
        except Exception as e:
            print('FAILED %s: %r' % (remote, e))
    sftp.close()
    cli.close()


if __name__ == '__main__':
    main()
