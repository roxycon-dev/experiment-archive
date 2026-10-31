#!/bin/bash
# 串行 runner：T0 -> T1（内存受限，不能并行）
export PATH=/root/miniconda3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
PY=/root/miniconda3/bin/python
OPS=/root/autodl-tmp/hrplus/ops
REPO=/root/autodl-tmp/hrplus/repo
RUN=/root/autodl-tmp/audit/audit_20260916_A
LOGD=/root/autodl-tmp/audit
cd $REPO; export PYTHONPATH=$REPO
for V in T0 T1; do
  echo "[$(date '+%T')] ===== 串行启动 $V ====="
  $PY $OPS/train_T0T1.py --variant $V --out-dir $RUN --seed 1 \
      --iters 8000 --val-every 1000 > $LOGD/$V.log 2>&1
  RC=$?
  echo "[$(date '+%T')] $V 结束 rc=$RC"
  grep -aE '全 ignore 跳过步|无 MA 阳性的 batch|采样来源计数' $LOGD/$V.log | tail -3
  ls -la $RUN/head_$V.pth 2>/dev/null
done
echo "[$(date '+%T')] T0/T1 串行全部结束"
