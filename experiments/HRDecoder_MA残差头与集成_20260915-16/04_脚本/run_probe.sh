#!/bin/bash
# 串行跑两个 LR 探针
export PATH=/root/miniconda3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
REPO=/root/autodl-tmp/hrplus/repo
WD=/root/autodl-tmp/hrplus/results/E_0916
cd /root
for P in p0005 p002; do
  echo "############ PROBE $P  $(date '+%F %T')"
  bash launch_E.sh $REPO/configs/lesion/exp/hrplus_mares_${P}.py $WD/probe_${P} 2>&1 | tail -3
  echo "--- $P val 曲线 ---"
  grep -aE 'Iter\(val\) \[76\]' $WD/probe_${P}/e_train.log | tail -10
  echo
done
echo "PROBES DONE $(date '+%F %T')"
