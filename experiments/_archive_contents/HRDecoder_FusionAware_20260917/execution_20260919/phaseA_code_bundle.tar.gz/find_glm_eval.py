# -*- coding: utf-8 -*-
"""Find GLM's original champion/h5 eval scripts on the server to see exactly
how member inference was performed (ground truth for fixing member_prob)."""
import paramiko

HOST, PORT, PASSWORD = '[REDACTED_USE_ENVIRONMENT]', 19057, 'NKgdrekCg6bo'


def main():
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli.connect(HOST, port=PORT, username='root', password=PASSWORD,
                timeout=25, banner_timeout=25, auth_timeout=25)

    def sh(cmd, timeout=60):
        _, o, e = cli.exec_command(cmd, timeout=timeout)
        out = o.read().decode('utf-8', 'replace').strip()
        err = e.read().decode('utf-8', 'replace').strip()
        return out + ('\n[STDERR] ' + err if err else '')

    print('== time ==')
    print(sh('date -Is'))

    print('== glm dir listing ==')
    print(sh('ls -la /root/autodl-tmp/hrplus/glm/ | head -60'))

    print('== find h5/champion/ensemble eval scripts anywhere under hrplus ==')
    print(sh("find /root/autodl-tmp/hrplus -maxdepth 3 -name '*.py' "
             "| grep -Ei 'h5|champ|ensem|eval|test' | grep -v __pycache__ "
             "| grep -v '/glm/' | head -40"))

    print('== h5_test.json location ==')
    print(sh("find /root/autodl-tmp/hrplus -maxdepth 4 -name 'h5_test.json' "
             "2>/dev/null | head"))

    print('== GLM scripts mentioning members/ensemble (glm dir, own files) ==')
    print(sh("grep -lEi 'ensemble|champion|h5' /root/autodl-tmp/hrplus/glm/*.py "
             "2>/dev/null | head -20"))

    cli.close()


if __name__ == '__main__':
    main()
