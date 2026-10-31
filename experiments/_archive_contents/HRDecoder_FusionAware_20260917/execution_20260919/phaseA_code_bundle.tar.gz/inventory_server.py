# -*- coding: utf-8 -*-
"""Inventory the handover server (port 44795) in a few batched commands."""
import paramiko

HOST, PORT, PASSWORD = '[REDACTED_USE_ENVIRONMENT]', 44795, 'eg3XqGIXhEdX'

CMDS = [
    ('disk usage by top dir', 'du -sh /root/autodl-tmp/* /root/HRDecoder /root/*.tgz 2>/dev/null | sort -h'),
    ('hrplus tree', 'find /root/autodl-tmp/hrplus -maxdepth 2 -type d | sort; echo ---; du -sh /root/autodl-tmp/hrplus/* 2>/dev/null | sort -h'),
    ('repo tree (fork)', 'find /root/autodl-tmp/hrplus/repo -maxdepth 2 | sort | head -60'),
    ('fork configs', 'ls -la /root/autodl-tmp/hrplus/repo/configs/lesion/exp/ 2>/dev/null; ls /root/autodl-tmp/hrplus/repo/configs/lesion/ 2>/dev/null'),
    ('fork model files', 'ls -la /root/autodl-tmp/hrplus/repo/mmseg/models/segmentors/ 2>/dev/null | head -20; echo ---; ls /root/autodl-tmp/hrplus/repo/mmseg/models/ 2>/dev/null'),
    ('fork weights', 'ls -la /root/autodl-tmp/hrplus/repo/weights/ 2>/dev/null'),
    ('ARCH grep (fork)', "grep -rn 'class HRPlus\\|class MAHead\\|class HRPlusMAResidual\\|class HRPlusRefinement' /root/autodl-tmp/hrplus/repo/mmseg /root/autodl-tmp/hrplus/repo 2>/dev/null | head -20"),
    ('ARCH grep (orig)', "grep -rn 'class HRDecoder\\|class EncoderDecoder\\|class HRNeck\\|class UPerHead' /root/HRDecoder/mmseg/models 2>/dev/null | head -20"),
    ('ops dir', 'ls -la /root/autodl-tmp/hrplus/ops/ 2>/dev/null | head -60'),
    ('results dir', 'find /root/autodl-tmp/hrplus/results -maxdepth 2 | head -40'),
    ('cache/audit', 'ls -la /root/autodl-tmp/hrplus/cache 2>/dev/null | head; echo ---AUDIT---; find /root/autodl-tmp/audit -maxdepth 2 -type d | head -30'),
    ('hr100', 'find /root/autodl-tmp/hr100 -maxdepth 2 | head -40'),
    ('root scripts', 'for f in /root/_run.sh /root/_deploy_guard.sh /root/box1_next.sh /root/box1_next.out /root/box1_v1b.out; do echo "===== $f"; cat "$f" 2>/dev/null | head -40; done'),
    ('HRDecoder repo tree', 'find /root/HRDecoder -maxdepth 2 -not -path "*/.git/*" | head -50'),
    ('HRDecoder configs', 'find /root/HRDecoder/configs -type f | head -30'),
    ('gpu/venv', 'nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; ls /root/miniconda3/envs 2>/dev/null; python -V 2>&1; which python'),
]


def main():
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli.connect(HOST, port=PORT, username='root', password=PASSWORD,
                timeout=25, banner_timeout=25, auth_timeout=30)
    for title, cmd in CMDS:
        _, o, e = cli.exec_command(cmd, timeout=180)
        out = (o.read().decode('utf-8', 'replace') +
               e.read().decode('utf-8', 'replace')).strip()
        print('\n########## %s ##########' % title)
        print(out[:4000])
    cli.close()


if __name__ == '__main__':
    main()
