#!/bin/bash
export PATH=/root/miniconda3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
PY=/root/miniconda3/bin/python
cd /root/autodl-tmp/hrplus/repo
export PYTHONPATH=/root/autodl-tmp/hrplus/repo
export OMP_NUM_THREADS=4

echo '=== 检查 D / E4 权重是否存在 ==='
ls -la /root/autodl-tmp/hrplus/D_0913_retry/*.pth 2>/dev/null
ls -la /root/HRDecoder/work_dirs/exp_e4_detail_split613/best_mIoU_iter_36000.pth 2>/dev/null || echo 'E4 权重不在实例上'
ls -la /root/autodl-tmp/hr100/trial_b/best_mIoU_iter_3000.pth

echo
echo '=== 验证评估器：单模型 D@2000 在 val 上应复现 41.615 ==='
$PY /root/autodl-tmp/hrplus/ops/eval_ens.py \
  --base-config configs/lesion/exp/hrplus_patch_dice.py \
  --models "configs/lesion/exp/hrplus_patch_dice.py=/root/autodl-tmp/hrplus/D_0913_retry/best_mIoU_iter_2000.pth" \
  --split val \
  --out /root/autodl-tmp/hrplus/results/L_0915/validate_D_val.json 2>&1 | grep -aE 'RESULT|models:|loaded:|pass|\[f0|Error|error|Traceback' | tail -20
echo '--- 结果文件 ---'
cat /root/autodl-tmp/hrplus/results/L_0915/validate_D_val.json 2>/dev/null
