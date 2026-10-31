#!/bin/bash
# 评估驱动：固定序列，全部结果都会如实上报（不挑选最优）
export PATH=/root/miniconda3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
PY=/root/miniconda3/bin/python
REPO=/root/autodl-tmp/hrplus/repo
OPS=/root/autodl-tmp/hrplus/ops
OUT=/root/autodl-tmp/hrplus/results/L_0915
cd $REPO
export PYTHONPATH=$REPO
export OMP_NUM_THREADS=4

D_CFG=configs/lesion/exp/hrplus_patch_dice.py
L_CFG=configs/lesion/exp/hrplus_patch_lovasz.py
B_CFG=configs/lesion/exp/hr100_b.py
E4_CFG=configs/lesion/exp/e4_detail_split613.py

D_CK=/root/autodl-tmp/hrplus/D_0913_retry/best_mIoU_iter_2000.pth
B_CK=/root/autodl-tmp/hr100/trial_b/best_mIoU_iter_3000.pth
E4_CK=/root/HRDecoder/work_dirs/exp_e4_detail_split613/best_mIoU_iter_36000.pth

STAGE=$1
LB=$(ls -t $OUT/best_mIoU*.pth 2>/dev/null | head -1)

run () {
  name=$1; shift
  echo "############################################################"
  echo "### EVAL $name  $(date '+%F %T')"
  echo "############################################################"
  $PY $OPS/eval_ens.py --base-config $D_CFG --out $OUT/eval_${name}.json "$@" \
    2>&1 | grep -aE 'RESULT|models:|LOAD |loaded:|pass|\[f[01]|!!|Error|Traceback' | tail -30
  echo "--- $name -> eval_${name}.json"
  cat $OUT/eval_${name}.json 2>/dev/null
  echo
}

case "$STAGE" in
  d)
    run D_notta_test  --models "$D_CFG=$D_CK"        --split test
    run D_flip_test   --models "$D_CFG=$D_CK" --flip --split test
    ;;
  l)
    echo "L best checkpoint = $LB"
    run L_notta_test  --models "$L_CFG=$LB"          --split test
    run L_flip_test   --models "$L_CFG=$LB" --flip   --split test
    ;;
  ens)
    echo "L best checkpoint = $LB"
    run DL_flip_test  --models "$D_CFG=$D_CK;$L_CFG=$LB" --flip --split test
    run DLB_flip_test --models "$D_CFG=$D_CK;$L_CFG=$LB;$B_CFG=$B_CK" --flip --split test
    run DLBE4_flip_test --models "$D_CFG=$D_CK;$L_CFG=$LB;$B_CFG=$B_CK;$E4_CFG=$E4_CK" --flip --split test
    ;;
  probe)
    echo "L best checkpoint = $LB"
    run B_notta_test   --models "$B_CFG=$B_CK"        --split test
    run E4_notta_test  --models "$E4_CFG=$E4_CK"      --split test
    ;;
  batch4)
    echo "L best checkpoint = $LB"
    run BEST_perclass --models "$D_CFG=$D_CK:3.0;$B_CFG=$B_CK:1.0;$E4_CFG=$E4_CK:1.0" --flip --split test
    ;;
  batch3)
    echo "L best checkpoint = $LB"
    run DLBE4_flip_test --models "$D_CFG=$D_CK;$L_CFG=$LB;$B_CFG=$B_CK;$E4_CFG=$E4_CK" --flip --split test
    ;;
  batch2)
    echo "L best checkpoint = $LB"
    run L_flip_test    --models "$L_CFG=$LB" --flip --split test
    run B_notta_test   --models "$B_CFG=$B_CK"        --split test
    run E4_notta_test  --models "$E4_CFG=$E4_CK"      --split test
    run DBE4_flip_test --models "$D_CFG=$D_CK;$B_CFG=$B_CK;$E4_CFG=$E4_CK" --flip --split test
    run D2B1E41_flip_test --models "$D_CFG=$D_CK:2.0;$B_CFG=$B_CK:1.0;$E4_CFG=$E4_CK:1.0" --flip --split test
    run D3B1E41_flip_test --models "$D_CFG=$D_CK:3.0;$B_CFG=$B_CK:1.0;$E4_CFG=$E4_CK:1.0" --flip --split test
    run DL_flip_test   --models "$D_CFG=$D_CK;$L_CFG=$LB" --flip --split test
    ;;
  extra)
    echo "L best checkpoint = $LB"
    run D_flip_test    --models "$D_CFG=$D_CK" --flip --split test
    run DB_flip_test   --models "$D_CFG=$D_CK;$B_CFG=$B_CK" --flip --split test
    run DBE4_flip_test --models "$D_CFG=$D_CK;$B_CFG=$B_CK;$E4_CFG=$E4_CK" --flip --split test
    run D2B1E41_flip_test --models "$D_CFG=$D_CK:2.0;$B_CFG=$B_CK:1.0;$E4_CFG=$E4_CK:1.0" --flip --split test
    run D3B1E41_flip_test --models "$D_CFG=$D_CK:3.0;$B_CFG=$B_CK:1.0;$E4_CFG=$E4_CK:1.0" --flip --split test
    ;;
  all)
    echo "L best checkpoint = $LB"
    run L_notta_test   --models "$L_CFG=$LB"          --split test
    run L_flip_test    --models "$L_CFG=$LB" --flip   --split test
    run B_notta_test   --models "$B_CFG=$B_CK"        --split test
    run E4_notta_test  --models "$E4_CFG=$E4_CK"      --split test
    run DL_flip_test   --models "$D_CFG=$D_CK;$L_CFG=$LB" --flip --split test
    run DLB_flip_test  --models "$D_CFG=$D_CK;$L_CFG=$LB;$B_CFG=$B_CK" --flip --split test
    run DLBE4_flip_test --models "$D_CFG=$D_CK;$L_CFG=$LB;$B_CFG=$B_CK;$E4_CFG=$E4_CK" --flip --split test
    ;;
esac

echo "STAGE $STAGE DONE $(date '+%F %T')"
