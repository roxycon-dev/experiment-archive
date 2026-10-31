#!/bin/bash
# P0: 对 D@2000 做 BN 运行统计重估（两种范围），串行执行
export PATH=/root/miniconda3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
PY=/root/miniconda3/bin/python
REPO=/root/autodl-tmp/hrplus/repo
WD=/root/autodl-tmp/hrplus/results/E_0916
cd $REPO
export PYTHONPATH=$REPO
export OMP_NUM_THREADS=4

CFG=configs/lesion/exp/hrplus_patch_dice.py
CK=/root/autodl-tmp/hrplus/D_0913_retry/best_mIoU_iter_2000.pth

for WHICH in student all; do
  echo "################ BN renorm which=$WHICH  $(date '+%F %T')"
  $PY /root/autodl-tmp/hrplus/ops/renorm_bn.py \
      --config $CFG --ckpt $CK \
      --out-ckpt $WD/d_renorm_${WHICH}.pth \
      --report $WD/renorm_${WHICH}.json \
      --which $WHICH 2>&1 | grep -aE '待重估|训练集|batchs|指纹|DONE|Traceback|Error|error' | tail -12
  echo
done
echo "P0 DONE $(date '+%F %T')"
