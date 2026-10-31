#!/bin/bash
# 等 batch2 跑完，再跑 batch3（4 模型均匀集成）
export PATH=/root/miniconda3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
OUT=/root/autodl-tmp/hrplus/results/L_0915
cd /root

echo "[chain] waiting for batch2 to finish $(date '+%F %T')"
while [ "$(ps -ef | grep -c '[h]rplus/ops/eval')" -gt 0 ]; do
  sleep 30
done
echo "[chain] batch2 done, starting batch3 $(date '+%F %T')"
bash run_evals.sh batch3 >> $OUT/evals_batch3.log 2>&1
echo "[chain] batch3 done $(date '+%F %T')"
