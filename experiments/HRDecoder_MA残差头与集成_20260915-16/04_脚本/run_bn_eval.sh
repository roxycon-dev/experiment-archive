#!/bin/bash
# 评估 BN 重估效果（val 76 张，快）
export PATH=/root/miniconda3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
PY=/root/miniconda3/bin/python
REPO=/root/autodl-tmp/hrplus/repo
OPS=/root/autodl-tmp/hrplus/ops
WD=/root/autodl-tmp/hrplus/results/E_0916
cd $REPO
export PYTHONPATH=$REPO
export OMP_NUM_THREADS=4

CFG=configs/lesion/exp/hrplus_patch_dice.py
D_CK=/root/autodl-tmp/hrplus/D_0913_retry/best_mIoU_iter_2000.pth

run () {
  echo "### $1  $(date '+%F %T')"
  $PY $OPS/eval_ens.py --base-config $CFG \
     --models "$CFG=$2" --split ${3:-val} \
     --out $WD/eval_$1.json 2>&1 | grep -aE 'RESULT|LOAD|Traceback|Error' | tail -5
  echo
}

run VAL_D_ref        $D_CK                    val
run VAL_D_renorm_stu $WD/d_renorm_student.pth val
run VAL_D_renorm_all $WD/d_renorm_all.pth     val

echo "BN-EVAL DONE $(date '+%F %T')"
