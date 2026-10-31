#!/bin/bash
export PATH=/root/miniconda3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
PY=/root/miniconda3/bin/python
cd /root/autodl-tmp/hrplus/repo

echo '=== 1) B checkpoint 的键（验证 strict=True 是否成立）==='
$PY - <<'EOF'
import torch
ck = torch.load('/root/autodl-tmp/hr100/trial_b/best_mIoU_iter_3000.pth', map_location='cpu')
sd = ck.get('state_dict', ck)
ks = list(sd.keys())
print('total keys:', len(ks))
print('top-level prefixes:', sorted(set(k.split('.')[0] for k in ks)))
ref = [k for k in ks if 'refiner' in k]
print('refiner-like keys count =', len(ref), ref[:5])
print('ckpt meta keys:', [k for k in ck.keys() if k != 'state_dict'] if isinstance(ck, dict) else 'n/a')
EOF

echo
echo '=== 2) 单卡训练是否走 DDP ==='
grep -n 'MMDataParallel\|MMDistributedDataParallel\|launcher\|distributed' /root/autodl-tmp/hrplus/repo/tools/train.py | head -20

echo
echo '=== 3) D 方案当时的启动与结果 ==='
cat /root/autodl-tmp/hrplus/results/D_0913_retry/D_0913_retry_summary.json 2>/dev/null
echo '--- 目录 ---'
ls /root/autodl-tmp/hrplus/results/D_0913_retry/ 2>/dev/null | head

echo
echo '=== 4) 验证 D 的 val 日志中 loss 项（确认无 lovasz 项）==='
$PY - <<'EOF'
import json
p = '/root/autodl-tmp/hrplus/results/D_0913_retry/20260913_004131.log.json'
n = 0
for line in open(p):
    try:
        o = json.loads(line)
    except Exception:
        continue
    if o.get('mode') == 'train' and n < 2:
        print('train keys:', sorted(o.keys()))
        n += 1
    if o.get('mode') == 'val' and n < 4:
        print('val keys:', sorted(o.keys()))
        n += 1
EOF

echo
echo '=== 5) 当前 hrplus_patch_lovasz.py 解析后的实际配置 ==='
$PY - <<'EOF'
import sys
sys.path.insert(0, '/root/autodl-tmp/hrplus/repo')
from mmcv import Config
c = Config.fromfile('/root/autodl-tmp/hrplus/repo/configs/lesion/exp/hrplus_patch_lovasz.py')
print('model.type =', c.model.type)
print('lovasz_weight =', c.model.get('lovasz_weight'))
print('semantic_refinement =', c.model.get('semantic_refinement'))
print('hard_weight =', c.model.get('hard_weight'))
print('freeze_bn =', c.model.get('freeze_bn'))
print('source_checkpoint =', c.model.get('source_checkpoint'))
print('test_cfg =', c.model.get('test_cfg'))
print('max_iters =', c.runner.max_iters)
print('eval interval =', c.evaluation.get('interval'), 'save_best =', c.evaluation.get('save_best'))
print('lr =', c.optimizer.lr, 'seed =', c.seed)
print('samples_per_gpu =', c.data.samples_per_gpu, 'workers =', c.data.workers_per_gpu)
print('D iff: lovasz_weight', c.model.get('lovasz_weight'), 'vs D 0.0')
EOF
