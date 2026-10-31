#!/bin/bash
# 严格串行：等 T1 结束后 -> U0（不再并发任何任务）
export PATH=/root/miniconda3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
PY=/root/miniconda3/bin/python
OPS=/root/autodl-tmp/hrplus/ops
REPO=/root/autodl-tmp/hrplus/repo
RUN=/root/autodl-tmp/audit/audit_20260916_A
LOGD=/root/autodl-tmp/audit
cd $REPO; export PYTHONPATH=$REPO

echo "[$(date '+%T')] 等待 T1 结束（严格串行，期间不启动任何其他任务）"
while pgrep -f "train_T0T1.py --variant T1" > /dev/null 2>&1; do
  sleep 30
done
echo "[$(date '+%T')] T1 已结束"
tail -3 $LOGD/T1.log
ls -la $RUN/head_T1.pth 2>/dev/null

echo "[$(date '+%T')] ===== 串行启动 U0 ====="
$PY $OPS/train_T0T1.py --variant U0 --out-dir $RUN --seed 1 \
    --iters 8000 --val-every 1000 > $LOGD/U0.log 2>&1
echo "[$(date '+%T')] U0 结束 rc=$?"
grep -aE '全 ignore 跳过步|无 MA 阳性的 batch|采样来源计数' $LOGD/U0.log | tail -3
ls -la $RUN/head_U0.pth 2>/dev/null
echo "[$(date '+%T')] T1/U0 串行全部结束"
