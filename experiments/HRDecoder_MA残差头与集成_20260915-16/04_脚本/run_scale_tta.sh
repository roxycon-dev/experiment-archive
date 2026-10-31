#!/bin/bash
# 多尺度 TTA：先验证脚本、再测收益
export PATH=/root/miniconda3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
PY=/root/miniconda3/bin/python
OPS=/root/autodl-tmp/hrplus/ops
REPO=/root/autodl-tmp/hrplus/repo
OUT=/root/autodl-tmp/hrplus/results/E_0916
cd $REPO
export PYTHONPATH=$REPO
export OMP_NUM_THREADS=4

D_CFG=configs/lesion/exp/hrplus_patch_dice.py
E_CFG=configs/lesion/exp/hrplus_mares.py
D_CK=/root/autodl-tmp/hrplus/D_0913_retry/best_mIoU_iter_2000.pth
E_CK=/root/autodl-tmp/hrplus/results/E_0916/best_mIoU_iter_1500.pth

run () {
  name=$1; shift
  echo "### $name  $(date '+%F %T')"
  $PY $OPS/eval_ens.py --base-config $D_CFG --out $OUT/eval_${name}.json "$@" \
    2>&1 | grep -aE 'RESULT|LOAD|scale |Traceback|Error' | tail -6
  echo
}

# 1) 验证：改脚本后 D 单模型仍应给出 42.576
run SCALE_VALID_D --models "$D_CFG=$D_CK" --scales 2048 --split test
# 2) 多尺度 TTA 收益（D 单模型，2048+2560，无翻转）
run SCALE_D_2048_2560 --models "$D_CFG=$D_CK" --scales 2048,2560 --split test
# 3) 最优集成 D+E：翻转 + 多尺度
run ENS_DE_flip_2scale --models "$D_CFG=$D_CK;$E_CFG=$E_CK" --flip --scales 2048,2560 --split test

echo "SCALE TTA DONE $(date '+%F %T')"
