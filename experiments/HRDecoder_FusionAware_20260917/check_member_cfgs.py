# -*- coding: utf-8 -*-
"""Fetch the remaining member configs and print their model type +
test_cfg (compute_aupr / use_sigmoid determine simple_test output shape)."""
import paramiko

HOST, PORT, PASSWORD = '[REDACTED_USE_ENVIRONMENT]', 19057, 'NKgdrekCg6bo'
FILES = [
    '/root/autodl-tmp/hrplus/repo/configs/lesion/exp/hrplus_mares.py',
    '/root/autodl-tmp/hrplus/repo/configs/lesion/exp/hr100_b.py',
    '/root/autodl-tmp/hrplus/repo/configs/lesion/exp/e4_detail_split613.py',
]


def main():
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli.connect(HOST, port=PORT, username='root', password=PASSWORD,
                timeout=25, banner_timeout=25, auth_timeout=25)

    def sh(cmd, timeout=60):
        _, o, e = cli.exec_command(cmd, timeout=timeout)
        return o.read().decode('utf-8', 'replace').strip()

    for f in FILES:
        print('=' * 70)
        print('## %s' % f)
        # model type / use_sigmoid / my_metrics / source_checkpoint lines
        print(sh("grep -nE \"type=|use_sigmoid|my_metrics|source_checkpoint|"
                 "compute_aupr|_base_|lovasz\" %s | head -25" % f))
        # find the base configs it inherits from
        bases = sh("grep -oE \"_base_[^=]*= *\\[? *'[^']*'\" %s | grep -oE \"'[^']*'\" | tr -d \\\"" % f)
        print('## base chain: %s' % bases)

    # resolve test_cfg from the base dataset/model configs
    print('=' * 70)
    print('## test_cfg in hr_ddr_2048 base:')
    print(sh("grep -n -A8 'test_cfg' /root/autodl-tmp/hrplus/repo/configs/_base_/"
             "datasets/hr_ddr_2048.py | head -30"))
    print('## test_cfg in hrdecoder_fcn_hr18 base:')
    print(sh("grep -n -A8 'test_cfg' /root/autodl-tmp/hrplus/repo/configs/_base_/"
             "models/hrdecoder_fcn_hr18.py | head -30"))
    cli.close()


if __name__ == '__main__':
    main()
