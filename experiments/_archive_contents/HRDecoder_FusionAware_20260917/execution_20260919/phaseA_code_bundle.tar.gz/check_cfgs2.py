# -*- coding: utf-8 -*-
"""Check hr100_a.py and baseline_split613.py: segmentor type, use_sigmoid,
my_metrics, test_cfg (compute_aupr) -- these determine simple_test output."""
import paramiko

HOST, PORT, PASSWORD = '[REDACTED_USE_ENVIRONMENT]', 19057, 'NKgdrekCg6bo'
FILES = [
    '/root/autodl-tmp/hrplus/repo/configs/lesion/exp/hr100_a.py',
    '/root/autodl-tmp/hrplus/repo/configs/lesion/exp/baseline_split613.py',
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
        print(sh("grep -nE \"_base_|type=|use_sigmoid|my_metrics|"
                 "source_checkpoint|compute_aupr|test_cfg\" %s | head -30" % f))
    cli.close()


if __name__ == '__main__':
    main()
