#!/bin/bash
# 部署独立硬截止关机保护（修正 python 路径）
export PATH=/root/miniconda3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
PY=/root/miniconda3/bin/python
cd /root/autodl-tmp/hrplus/ops

DEADLINE=$($PY -c "
from datetime import datetime, timezone, timedelta
CST=timezone(timedelta(hours=8))
now=datetime.now(CST)
d=now.replace(hour=5,minute=0,second=0,microsecond=0)
if d<=now: d=d+timedelta(days=1)
print(int(d.timestamp()))
")
$PY -c "
from datetime import datetime,timezone,timedelta
print('DEADLINE_CST=',datetime.fromtimestamp($DEADLINE,timezone(timedelta(hours=8))).strftime('%Y-%m-%d %H:%M:%S'),'unix=',$DEADLINE)
"

pkill -f ops_deadline_guard.py 2>/dev/null || true
sleep 1

nohup $PY ops_deadline_guard.py "$DEADLINE" watch > deadline_guard.out 2>&1 &
sleep 2
echo "--- guard 进程 ---"
ps -ef | grep ops_deadline_guard.py | grep -v grep
echo "--- 日志 ---"
cat deadline_guard.log 2>/dev/null
echo "--- state 目录准备 ---"
mkdir -p /root/autodl-tmp/hrplus/results/L_0915
ls -la /root/autodl-tmp/hrplus/results/
