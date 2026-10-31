# -*- coding: utf-8 -*-
"""Final detail pass: architecture class locations, project docs, git state."""
import paramiko

HOST, PORT, PASSWORD = '[REDACTED_USE_ENVIRONMENT]', 44795, 'eg3XqGIXhEdX'

CMDS = [
    ('DetailBranch/DetailHead/MAHead definitions',
     "grep -rn 'class DetailBranch\\|class DetailHead\\|class MAHead\\|class HRDecoder(' "
     "/root/autodl-tmp/hrplus/repo /root/HRDecoder/mmseg 2>/dev/null | head -12"),
    ('register_module lines in the fork',
     "grep -rn 'register_module' /root/autodl-tmp/hrplus/repo/*.py /root/autodl-tmp/hrplus/repo/mmseg/models/segmentors/HRDecoder.py 2>/dev/null | head -25"),
    ('fork mmseg segmentors HRDecoder classes',
     "grep -n '^class \\|^    def \\|^def ' /root/autodl-tmp/hrplus/repo/mmseg/models/segmentors/HRDecoder.py | head -30"),
    ('fork mmseg submodule listings',
     "for d in backbones decode_heads necks losses utils core datasets ops apis; do echo \"== mmseg/$d\"; ls /root/autodl-tmp/hrplus/repo/mmseg/$d 2>/dev/null | head -8; done"),
    ('CLAUDE.md (fork conventions)',
     "head -60 /root/autodl-tmp/hrplus/repo/CLAUDE.md"),
    ('ops/PREFLIGHT.md',
     "cat /root/autodl-tmp/hrplus/ops/PREFLIGHT.md"),
    ('git state HRDecoder',
     "cd /root/HRDecoder && git log --oneline -5 2>&1 | head; echo ---; git status --short 2>&1 | head -12; echo ---; git branch -a 2>&1 | head"),
    ('audit operations checksums head',
     "head -25 /root/autodl-tmp/audit/audit_20260916_A/operations/artifact_checksums.sha256 2>/dev/null; wc -l /root/autodl-tmp/audit/audit_20260916_A/operations/artifact_checksums.sha256"),
    ('fork root scripts/docs',
     "ls -la /root/autodl-tmp/hrplus/repo/*.py /root/autodl-tmp/hrplus/repo/*.md /root/autodl-tmp/hrplus/repo/*.txt 2>/dev/null"),
    ('repo data + configs base',
     "ls /root/autodl-tmp/hrplus/repo/configs/_base_/ 2>/dev/null; ls /root/autodl-tmp/hrplus/repo/configs/_base_/models 2>/dev/null; ls /root/autodl-tmp/hrplus/repo/configs/_base_/datasets 2>/dev/null"),
    ('hr100 configs/scripts',
     "ls /root/autodl-tmp/hr100/ /root/autodl-tmp/hr100/code/configs/lesion/exp/ 2>/dev/null | head -30"),
    ('val cache tarball + arm tgz contents',
     "tar -tzf /root/arm_small.tgz 2>/dev/null | head -12; echo ---; tar -tzf /root/arm_heads.tgz 2>/dev/null | head -8; echo ---VALCACHE---; tar -tzf /root/valcache_box2.tgz 2>/dev/null | head -5"),
    ('env / python',
     "ls /root/miniconda3/envs 2>/dev/null; ls /root/miniconda3/bin/python*; /root/miniconda3/bin/python -c 'import torch,mmcv;print(torch.__version__, mmcv.__version__)' 2>&1 | tail -2"),
]


def main():
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli.connect(HOST, port=PORT, username='root', password=PASSWORD,
                timeout=25, banner_timeout=25, auth_timeout=30)
    for title, cmd in CMDS:
        _, o, e = cli.exec_command(cmd, timeout=240)
        out = (o.read().decode('utf-8', 'replace') +
               e.read().decode('utf-8', 'replace')).strip()
        print('\n########## %s ##########' % title)
        print(out[:4000])
    cli.close()


if __name__ == '__main__':
    main()
