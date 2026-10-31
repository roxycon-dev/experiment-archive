# -*- coding: utf-8 -*-
"""Depth pass: class/def inventories, config contents, script purposes,
weights provenance, audit/results/work_dirs contents."""
import paramiko

HOST, PORT, PASSWORD = '[REDACTED_USE_ENVIRONMENT]', 44795, 'eg3XqGIXhEdX'

CMDS = [
    ('fork core module classes',
     "for f in hrplus_core.py hr100_refinement.py hrplus_patch.py ma_residual.py hr100_ensemble.py; do "
     "echo \"===== $f ($(wc -l < /root/autodl-tmp/hrplus/repo/$f) lines)\"; "
     "grep -n '^class \\|^def \\|^    def ' /root/autodl-tmp/hrplus/repo/$f | head -25; done"),
    ('config contents (fork key 4)',
     "for f in hrplus_patch_dice.py hr100_b.py e4_detail_split613.py hrplus_mares.py; do "
     "echo \"===== $f\"; cat /root/autodl-tmp/hrplus/repo/configs/lesion/exp/$f; done"),
    ('ops docstrings',
     "cd /root/autodl-tmp/hrplus/ops && for f in *.py; do "
     "echo \"- $f: $(sed -n '2,4p' $f | tr '\\n' ' ' | cut -c1-140)\"; done"),
    ('weights inventory (all .pth with size/date)',
     "find /root/autodl-tmp /root/HRDecoder -name '*.pth' -size +1M -printf '%10s  %TY-%Tm-%Td %TH:%TM  %p\\n' 2>/dev/null | sort -k4 | head -45"),
    ('audit subdirs content',
     "for d in /root/autodl-tmp/audit/audit_20260916_A/*/; do echo \"== $d\"; ls \"$d\" | head -6; done"),
    ('results subdirs',
     "for d in /root/autodl-tmp/hrplus/results/*/; do echo \"== $d ($(du -sh $d | cut -f1))\"; ls \"$d\" | head -12; done"),
    ('HRDecoder work_dirs',
     "ls -la /root/HRDecoder/work_dirs/ | head -20; echo ---; for d in /root/HRDecoder/work_dirs/*/; do echo \"== $d\"; ls \"$d\" | head -5; done 2>/dev/null | head -40"),
    ('HRDecoder base weights + mmseg models',
     "ls -la /root/HRDecoder/*.pth /root/HRDecoder/mmseg/models/segmentors/*.py 2>/dev/null"),
    ('data layout',
     "ls /root/autodl-tmp/hrplus/repo/data/DDR_split_613/ 2>/dev/null; "
     "for s in train val test; do echo -n \"$s images: \"; ls /root/autodl-tmp/hrplus/repo/data/DDR_split_613/images/$s 2>/dev/null | wc -l; done; "
     "ls /root/autodl-tmp/hrplus/repo/data/DDR_split_613/*.txt 2>/dev/null | head"),
    ('D_0913_retry content', 'ls -la /root/autodl-tmp/hrplus/D_0913_retry/ | head -20; cat /root/autodl-tmp/hrplus/D_0913_retry_summary.json'),
    ('hr100 code vs fork repo diff hint', 'ls /root/autodl-tmp/hr100/code/ | head -20; echo ---; ls /root/autodl-tmp/hr100/code/mmseg/models/segmentors/ 2>/dev/null'),
    ('repo tests/tools/diagnostics', 'ls /root/autodl-tmp/hrplus/repo/tests /root/autodl-tmp/hrplus/repo/tools /root/autodl-tmp/hrplus/repo/diagnostics 2>/dev/null'),
    ('all root-level files', 'ls -la /root/*.out /root/*.sh /root/*.tgz /root/*.json 2>/dev/null | head -25'),
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
        print(out[:4500])
    cli.close()


if __name__ == '__main__':
    main()
