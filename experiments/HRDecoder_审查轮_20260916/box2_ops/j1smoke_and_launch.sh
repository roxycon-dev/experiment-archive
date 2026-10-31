#!/bin/bash
# 在【2 号机】上运行：J1 20 步 smoke -> 证据判定门 -> 通过则后台启动 J0/J1 正式串行训练
set -u
PY=/root/miniconda3/bin/python
R=/root/box1_sync/repo
W=/root/box1_sync/workdir
cd $R && export PYTHONPATH=$R

echo "[$(date '+%T')] 1) J1 smoke 20 步"
rm -rf $W/smoke_j1
timeout 900 $PY tools/train.py configs/lesion/exp/j1_split613.py \
  --work-dir $W/smoke_j1 --no-validate --seed 42 \
  --options runner.max_iters=20 evaluation.interval=100000 checkpoint_config.interval=20 log_config.interval=5 \
  > $W/smoke_j1.out 2>&1
echo "   rc=$?"
grep -aE 'FreezeScope|Traceback|AssertionError|CUDA out of memory' $W/smoke_j1.out | tail -12

echo "[$(date '+%T')] 2) J1 证据判定门"
$PY - <<'PYEOF'
import json, os, subprocess, sys
p = '/root/box1_sync/workdir/smoke_j1/J0J1_evidence_J1.json'
if not os.path.exists(p):
    print('!! 没有证据文件 -> 判定门不可用，不启动正式训练'); sys.exit(1)
d = json.load(open(p)); g = d['grad_first_step']
checks = {
    'trainable_tensors==440': d['trainable_tensors'] == 440,
    'backbone_tensors==432': d['backbone_tensors'] == 432,
    'last_stage_grad>0': g['last_stage_grad_sum'] > 0,
    'other_backbone_grad==0': g['other_backbone_grad_sum'] == 0.0,
    'decoder_grad>0': g['decoder_grad_sum'] > 0,
    'bn_all_eval': bool(d['bn_all_eval']),
    'bn_buffers_unchanged': bool(d.get('bn_buffers_unchanged')),
    'reference_frozen': bool(d['reference_frozen']),
    'student!=reference': bool(d['init_probe']['differ']),
    'last_stage==stage4': d['j1_last_stage'] == 'stage4',
}
for k, v in checks.items():
    print('  [%s] %s' % ('ok' if v else 'FAIL', k))
print('  last_stage_grad_sum = %.4f  /  other_backbone_grad_sum = %.6f'
      % (g['last_stage_grad_sum'], g['other_backbone_grad_sum']))
if all(checks.values()):
    print('  => 判定门通过，启动正式 J0/J1 串行训练')
    subprocess.run('setsid nohup bash /root/box1_sync/ops/run_j0j1.sh '
                   '> /root/box1_sync/workdir/runner_j0j1.log 2>&1 < /dev/null &',
                   shell=True, cwd='/root/box1_sync')
    print('  已后台启动')
else:
    print('  => 判定门未通过，不启动'); sys.exit(1)
PYEOF

echo "[$(date '+%T')] 3) 确认后台任务"
sleep 20
ps -eo pid,etime,args | grep -E '[t]rain\.py|[r]un_j0j1' | head -4
tail -4 $W/runner_j0j1.log 2>/dev/null
