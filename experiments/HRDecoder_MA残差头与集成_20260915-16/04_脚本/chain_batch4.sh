#!/bin/bash
# 等 batch3 (DLBE4) 跑完，再跑 batch4（最佳配置 + 逐类明细）
export PATH=/root/miniconda3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
OUT=/root/autodl-tmp/hrplus/results/L_0915
cd /root
echo "[chain4] waiting $(date '+%F %T')"
while [ "$(ps -ef | grep -c '[h]rplus/ops/eval')" -gt 0 ]; do sleep 30; done
echo "[chain4] start batch4 $(date '+%F %T')"
bash run_evals.sh batch4 >> $OUT/evals_batch4.log 2>&1
echo "[chain4] done $(date '+%F %T')"
