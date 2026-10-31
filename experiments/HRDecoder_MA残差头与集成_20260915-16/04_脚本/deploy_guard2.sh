#!/bin/bash
# 部署独立硬截止关机保护（今天 16:00 CST）
export PATH=/root/miniconda3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
PY=/root/miniconda3/bin/python
cd /root/autodl-tmp/hrplus/ops

DEADLINE=$($PY -c "
from datetime import datetime, timezone, timedelta
CST=timezone(timedelta(hours=8))
now=datetime.now(CST)
d=now.replace(hour=16,minute=0,second=0,microsecond=0)
if d<=now: d=d+timedelta(days=1)
print(int(d.timestamp()))
")
$PY -c "
from datetime import datetime,timezone,timedelta
print('DEADLINE_CST=',datetime.fromtimestamp($DEADLINE,timezone(timedelta(hours=8))).strftime('%Y-%m-%d %H:%M:%S'),'unix=',$DEADLINE)
"

for p in $(ps -ef | grep 'deadline_guard' | grep -v grep | awk '{print $2}'); do kill -TERM $p 2>/dev/null; done
sleep 2
rm -f deadline_guard.log
nohup $PY ops_deadline_guard.py "$DEADLINE" watch > deadline_guard.out 2>&1 &
sleep 3
echo "--- guard 进程 ---"
ps -ef | grep 'deadline_guard' | grep -v grep
cat deadline_guard.log 2>/dev/null
echo "--- GPU ---"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader
