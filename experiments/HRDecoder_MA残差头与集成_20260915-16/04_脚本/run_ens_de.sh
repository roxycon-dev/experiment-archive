#!/bin/bash
# 集成评估：把新模型 E（B基准+MA残差头）纳入
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
B_CFG=configs/lesion/exp/hr100_b.py
E4_CFG=configs/lesion/exp/e4_detail_split613.py

D_CK=/root/autodl-tmp/hrplus/D_0913_retry/best_mIoU_iter_2000.pth
E_CK=/root/autodl-tmp/hrplus/results/E_0916/best_mIoU_iter_1500.pth
B_CK=/root/autodl-tmp/hr100/trial_b/best_mIoU_iter_3000.pth
E4_CK=/root/HRDecoder/work_dirs/exp_e4_detail_split613/best_mIoU_iter_36000.pth

run () {
  name=$1; shift
  echo "############################################################"
  echo "### EVAL $name  $(date '+%F %T')"
  echo "############################################################"
  $PY $OPS/eval_ens.py --base-config $D_CFG --out $OUT/eval_${name}.json "$@" \
    2>&1 | grep -aE 'RESULT|models:|LOAD|loaded|pass|\[f[01]|!!|Error|Traceback' | tail -25
  echo "--- -> $OUT/eval_${name}.json"
  cat $OUT/eval_${name}.json 2>/dev/null
  echo
}

run ENS_DE_flip      --models "$D_CFG=$D_CK;$E_CFG=$E_CK" --flip --split test
run ENS_DEB_E4_flip  --models "$D_CFG=$D_CK;$E_CFG=$E_CK;$B_CFG=$B_CK;$E4_CFG=$E4_CK" --flip --split test
run ENS_D2E1B1E41_flip --models "$D_CFG=$D_CK:2.0;$E_CFG=$E_CK:1.0;$B_CFG=$B_CK:1.0;$E4_CFG=$E4_CK:1.0" --flip --split test
run ENS_D1E2B1E41_flip --models "$D_CFG=$D_CK:1.0;$E_CFG=$E_CK:2.0;$B_CFG=$B_CK:1.0;$E4_CFG=$E4_CK:1.0" --flip --split test
run ENS_D3E2B1E41_flip --models "$D_CFG=$D_CK:3.0;$E_CFG=$E_CK:2.0;$B_CFG=$B_CK:1.0;$E4_CFG=$E4_CK:1.0" --flip --split test

echo "ENSEMBLE DONE $(date '+%F %T')"
