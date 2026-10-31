#!/bin/bash
# T0 -> T1 连续受控对照（共享同一几何/时程/缓存，唯一差异=采样）
export PATH=/root/miniconda3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
PY=/root/miniconda3/bin/python
OPS=/root/autodl-tmp/hrplus/ops
REPO=/root/autodl-tmp/hrplus/repo
RUN=/root/autodl-tmp/audit/audit_20260916_A
LOGD=/root/autodl-tmp/audit
cd $REPO; export PYTHONPATH=$REPO

run_one () {
  V=$1
  echo "[$(date '+%T')] ===== 启动 $V (8000 步, seed 1) ====="
  $PY $OPS/train_T0T1.py --variant $V --out-dir $RUN --seed 1 \
      --iters 8000 --val-every 1000 > $LOGD/$V.log 2>&1
  RC=$?
  echo "[$(date '+%T')] $V 结束 rc=$RC"
  echo "----- $V 曲线 -----"
  grep -aE 'iter=|val' $LOGD/$V.log | tail -20
  echo "----- $V 产物 -----"
  ls -la $RUN/head_$V.pth $RUN/${V}_val.json 2>/dev/null
}

run_one T0
run_one T1
echo "[$(date '+%T')] T0/T1 全部结束；下一步 U0（需先写 U0 多尺度网络，不会自动开跑）"
