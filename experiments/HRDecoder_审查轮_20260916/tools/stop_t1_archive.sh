#!/bin/bash
# 停掉当前（作废的）T1 与等待中的队列，并把旧产物归档留证。
# 本轮 D-006 修复：val 点未实现 -> 带 val 重跑全臂。
set -u
RUN=/root/autodl-tmp/audit/audit_20260916_A
LOGD=/root/autodl-tmp/audit
PAT="train_T0""T1.py"
QPAT="runner_after_T""1.sh"

echo "[$(date '+%F %T')] --- 1. 归档旧产物（补丁前，无 val 记录）---"
for f in head_T0.pth head_T1.pth val_curve_T0_s1.json val_curve_T1_s1.json; do
  if [ -f "$RUN/$f" ]; then
    cp -p "$RUN/$f" "$RUN/${f%.*}_prepatch_20260917.${f##*.}"
    echo "  archived $f"
  fi
done
for f in T0.log T1.log; do
  if [ -f "$LOGD/$f" ]; then
    cp -p "$LOGD/$f" "${LOGD}/${f%.log}_prepatch.log"
    echo "  archived $f -> ${f%.log}_prepatch.log"
  fi
done

echo "[$(date '+%F %T')] --- 2. 停止等待中的串行队列 ---"
for p in $(pgrep -f "$QPAT"); do echo "  kill queue pid=$p"; kill "$p" 2>/dev/null; done
sleep 2
pgrep -f "$QPAT" > /dev/null && echo "  !! 队列仍在" || echo "  队列已停"

echo "[$(date '+%F %T')] --- 3. 停止 T1（作废：val 点未实现）---"
for p in $(pgrep -f "$PAT"); do echo "  kill pid=$p"; kill "$p" 2>/dev/null; done
sleep 6
if pgrep -f "$PAT" > /dev/null; then
  echo "  仍在，强杀："; for p in $(pgrep -f "$PAT"); do kill -9 "$p" 2>/dev/null; done; sleep 2
fi

echo "[$(date '+%F %T')] --- 4. 残留确认 ---"
if pgrep -af "$PAT" > /dev/null; then pgrep -af "$PAT"; else echo "  无训练进程残留（CPU 已让出）"; fi
echo "  T1 作废前最后一行："; tail -1 "$LOGD/T1.log"
echo "  系统负载："; uptime
