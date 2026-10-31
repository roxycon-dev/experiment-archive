#!/bin/bash
# 实时状态监控：一条命令看全 R0/T0/T1/U0 的进程、曲线、guard、GPU、产物
# 用法: bash monitor.sh
export PATH=/root/miniconda3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
RUN=/root/autodl-tmp/audit/audit_20260916_A
LOGD=/root/autodl-tmp/audit

echo "════════ $(date '+%F %T') ════════"

echo "── 进程 ──"
printf "  guard    : "; ps -eo pid,etime,args | awk '/hard_deadline_guard/ && !/awk/ {printf "%s (up %s)\n",$1,$2}' | head -1 || true
for pat in r0_val_counts train_T0 train_T1 train_U0 h7_all_heads; do
  n=$(ps -eo args | grep -c "[/]ops/$pat\|${pat}\.py" 2>/dev/null || true)
  p=$(ps -eo pid,etime,args | awk -v P="$pat" '$0 ~ P && !/awk/ {print $1" (up "$2")"}' | head -1)
  [ -n "$p" ] && printf "  %-9s: %s\n" "$pat" "$p"
done

echo "── GPU ──"
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader | sed 's/^/  /'

echo "── guard 尾部 ──"
tail -2 $RUN/guard.log 2>/dev/null | sed 's/^/  /'

echo "── R0 进度 ──"
if [ -f $LOGD/r0_counts.log ]; then
  tail -c 400 $LOGD/r0_counts.log | tr '\r' '\n' | grep -E '[0-9]+/76|BASE_0|H4|delta|状态' | tail -4 | sed 's/^/  /'
else echo "  (无日志)"; fi

for ITEM in T0 T1 U0; do
  f=$LOGD/${ITEM}.log
  [ -f "$f" ] || continue
  echo "── $ITEM 曲线 ──"
  grep -aE '^(step|iter)|val' "$f" 2>/dev/null | tail -12 | sed 's/^/  /'
done

echo "── 本轮产物 ──"
ls -lt $RUN/diagnostics/ 2>/dev/null | head -6 | awk 'NR>1{printf "  %s %s %s\n",$6,$7,$9}'
ls -lt $RUN/experiments/ 2>/dev/null | head -5 | awk 'NR>1{printf "  %s %s %s\n",$6,$7,$9}'
