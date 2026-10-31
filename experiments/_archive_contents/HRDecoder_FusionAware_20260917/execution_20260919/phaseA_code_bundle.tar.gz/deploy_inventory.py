# -*- coding: utf-8 -*-
"""Step 1: server inventory.  Read-only -- connects, gathers environment
facts needed to finalize settings.json, prints a JSON report.  No changes."""
import json
import sys

import paramiko

HOST, PORT, PASSWORD = '[REDACTED_USE_ENVIRONMENT]', 19057, 'NKgdrekCg6bo'

CMDS = [
    ('host', 'hostname && date -Is && uptime'),
    ('gpu', 'nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader'),
    ('disk', 'df -h / /root/autodl-tmp 2>/dev/null'),
    ('python', 'which python python3 2>/dev/null; python -V 2>&1'),
    ('torch', 'python -c "import torch,mmcv,mmseg;print(torch.__version__, torch.cuda.is_available(), mmcv.__version__, mmseg.__version__)" 2>&1'),
    ('hrplus_tree', 'ls /root/autodl-tmp/hrplus/ 2>&1'),
    ('hrplus_repo', 'ls /root/autodl-tmp/hrplus/repo/ 2>&1'),
    ('repo_configs', 'ls /root/autodl-tmp/hrplus/repo/configs/lesion/exp/ 2>&1'),
    ('repo_weights', 'ls -la /root/autodl-tmp/hrplus/repo/weights/ 2>&1'),
    ('hrplus_model_files', 'ls /root/autodl-tmp/hrplus/repo/*.py 2>&1'),
    ('data_dirs', 'for d in /root/autodl-tmp/hrplus/data /root/autodl-tmp/hrplus/repo/data /root/autodl-tmp/hr100/data /root/autodl-tmp/data; do echo "== $d"; ls $d 2>/dev/null; done'),
    ('ddr_split', 'for d in /root/autodl-tmp/hrplus/data/DDR_split_613 /root/autodl-tmp/hrplus/repo/data/DDR_split_613 /root/autodl-tmp/hr100/data/DDR_split_613; do if [ -d "$d" ]; then echo "== $d"; for s in train val test; do echo "$s: $(ls $d/images/$s 2>/dev/null | wc -l) imgs, $(ls $d/labels/$s 2>/dev/null | wc -l) labels"; done; fi; done'),
    ('d_full', 'ls -la /root/autodl-tmp/hrplus/D_0913_retry/ 2>&1'),
    ('e_ckpt', 'ls -la /root/autodl-tmp/hrplus/results/E_0916/ 2>&1'),
    ('b_ckpt', 'ls -la /root/autodl-tmp/hr100/trial_b/ 2>&1'),
    ('e4_ckpt', 'ls -la /root/HRDecoder/work_dirs/exp_e4_detail_split613/ 2>&1'),
    ('shas', 'for f in /root/autodl-tmp/hrplus/repo/weights/b_weights.pth /root/autodl-tmp/hr100/trial_b/best_mIoU_iter_3000.pth /root/autodl-tmp/hrplus/results/E_0916/best_mIoU_iter_1500.pth /root/autodl-tmp/hrplus/D_0913_retry/best_mIoU_iter_2000.pth /root/HRDecoder/work_dirs/exp_e4_detail_split613/best_mIoU_iter_36000.pth; do if [ -f "$f" ]; then echo "$(sha256sum $f)"; else echo "MISSING $f"; fi; done'),
    ('cfg_shas', 'cd /root/autodl-tmp/hrplus/repo && for f in configs/lesion/exp/hrplus_patch_dice.py configs/lesion/exp/hrplus_mares.py configs/lesion/exp/hr100_b.py configs/lesion/exp/e4_detail_split613.py; do if [ -f "$f" ]; then echo "$(sha256sum $f)"; else echo "MISSING $f"; fi; done'),
    ('mares_cfg', 'cat /root/autodl-tmp/hrplus/repo/configs/lesion/exp/hrplus_mares.py 2>&1'),
    ('glm_dir', 'ls -la /root/autodl-tmp/hrplus/glm/ 2>/dev/null || echo NO_GLM_DIR'),
    ('cache_dir', 'ls /root/autodl-tmp/hrplus/cache/ 2>/dev/null || echo NO_CACHE'),
    ('running', 'ps aux | grep -E "python|run_" | grep -v grep || echo NOTHING_RUNNING'),
    ('autodl_shutdown', 'which shutdown poweroff 2>&1'),
]


def main():
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli.connect(HOST, port=PORT, username='root', password=PASSWORD,
                timeout=30, banner_timeout=30, auth_timeout=30)
    out = {}
    for tag, cmd in CMDS:
        try:
            _, stdout, stderr = cli.exec_command(cmd, timeout=120)
            so = stdout.read().decode('utf-8', 'replace').strip()
            se = stderr.read().decode('utf-8', 'replace').strip()
            out[tag] = {'stdout': so[-8000:], 'stderr': se[-2000:]}
        except Exception as e:
            out[tag] = {'error': repr(e)}
    cli.close()
    with open(r'D:\dataset\experiment\HRDecoder_FusionAware_20260917\inventory_report.json', 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    for tag, r in out.items():
        print('=' * 70)
        print('## %s' % tag)
        if 'error' in r:
            print('  ERROR %s' % r['error'])
        else:
            print(r['stdout'])
            if r.get('stderr'):
                print('  [stderr] %s' % r['stderr'][:500])


if __name__ == '__main__':
    main()
