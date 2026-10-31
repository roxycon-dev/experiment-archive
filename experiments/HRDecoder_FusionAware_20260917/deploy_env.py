# -*- coding: utf-8 -*-
"""Step 2: environment + reference values fetch.  Read-only."""
import json

import paramiko

HOST, PORT, PASSWORD = '[REDACTED_USE_ENVIRONMENT]', 19057, 'NKgdrekCg6bo'

CMDS = [
    ('container_uptime', 'ps -o etimes=,lstart= -p 1 2>/dev/null; stat -c %y /proc/1 2>/dev/null'),
    ('pyver', '/root/miniconda3/bin/python -V 2>&1'),
    ('torchver', '/root/miniconda3/bin/python -c "import torch,mmcv,mmseg,cv2,numpy;print(\'torch\',torch.__version__);print(\'cuda\',torch.cuda.is_available());print(\'mmcv\',mmcv.__version__);print(\'mmseg\',mmseg.__version__);print(\'cv2\',cv2.__version__);print(\'numpy\',numpy.__version__)" 2>&1'),
    ('backup_student_sha', 'sha256sum /root/autodl-tmp/hrplus/D_0913_retry/backup_student_2000.pth'),
    ('ens_val_ref', 'cat /root/autodl-tmp/hrplus/results/E_0916/eval_ENS_D3E2B1E41_flip.json 2>&1'),
    ('e_val_ref', 'cat /root/autodl-tmp/hrplus/results/E_0916/eval_VAL_E1500.json 2>&1'),
    ('d_val_ref', 'cat /root/autodl-tmp/hrplus/results/E_0916/eval_VAL_D_ref.json 2>&1'),
    ('b_val_ref', 'cat /root/autodl-tmp/hrplus/results/E_0916/eval_VAL_B_base.json 2>&1'),
    ('ma_residual', 'head -120 /root/autodl-tmp/hrplus/repo/ma_residual.py 2>&1'),
    ('model_file_shas', 'cd /root/autodl-tmp/hrplus/repo && sha256sum hrplus_patch.py hr100_refinement.py hrplus_core.py ma_residual.py hr100_ensemble.py mmseg/models/segmentors/HRDecoder.py mmseg/models/segmentors/lesion_encoder_decoder.py mmseg/datasets/lesion_dataset.py mmseg/models/losses/binary_loss.py 2>&1'),
    ('base_cfg_shas', 'cd /root/autodl-tmp/hrplus/repo && sha256sum configs/lesion/exp/hr100_a.py configs/_base_/models/hrdecoder_fcn_hr18.py configs/_base_/datasets/hr_ddr_2048.py 2>&1'),
    ('data_split_txt', 'ls /root/autodl-tmp/hrplus/repo/data/DDR_split_613/ 2>&1'),
    ('free_mem', 'free -g 2>/dev/null'),
]


def main():
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli.connect(HOST, port=PORT, username='root', password=PASSWORD,
                timeout=30, banner_timeout=30, auth_timeout=30)
    out = {}
    for tag, cmd in CMDS:
        try:
            _, stdout, stderr = cli.exec_command(cmd, timeout=180)
            out[tag] = {'stdout': stdout.read().decode('utf-8', 'replace').strip()[-12000:],
                        'stderr': stderr.read().decode('utf-8', 'replace').strip()[-1500:]}
        except Exception as e:
            out[tag] = {'error': repr(e)}
    cli.close()
    with open(r'D:\dataset\experiment\HRDecoder_FusionAware_20260917\env_report.json', 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    for tag, r in out.items():
        print('=' * 70)
        print('## %s' % tag)
        print(r.get('stdout', r.get('error', '')))


if __name__ == '__main__':
    main()
