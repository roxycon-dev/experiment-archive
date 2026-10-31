#!/bin/bash
# 测两个残差头模型的 test（与 D 的 42.575 同协议）
export PATH=/root/miniconda3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
PY=/root/miniconda3/bin/python
OPS=/root/autodl-tmp/hrplus/ops
REPO=/root/autodl-tmp/hrplus/repo
E1=/root/autodl-tmp/hrplus/results/E_0916
E2=/root/autodl-tmp/hrplus/results/E2_0916
cd $REPO
export PYTHONPATH=$REPO
export OMP_NUM_THREADS=4

echo "######## TEST E(B基准+头)@1500  $(date '+%F %T')"
$PY $OPS/eval_ens.py --base-config configs/lesion/exp/hrplus_mares.py \
  --models "configs/lesion/exp/hrplus_mares.py=$E1/best_mIoU_iter_1500.pth" \
  --split test --out $E1/eval_TEST_E_head.json 2>&1 | grep -aE 'RESULT|LOAD|Traceback|Error' | tail -3
echo

LB=$(ls -t $E2/best_mIoU_iter_*.pth 2>/dev/null | head -1)
echo "######## TEST E2(D基准+头)@$LB  $(date '+%F %T')"
$PY $OPS/eval_ens.py --base-config configs/lesion/exp/hrplus_maond.py \
  --models "configs/lesion/exp/hrplus_maond.py=$LB" \
  --split test --out $E2/eval_TEST_E2_head.json 2>&1 | grep -aE 'RESULT|LOAD|Traceback|Error' | tail -3
echo
echo "TEST DONE $(date '+%F %T')"
