#!/bin/bash
# 重启 hard_deadline_guard：仅替换 guard，绝不触碰 T0/T1/U0 训练进程。
# 关键：本脚本以 `bash /root/restart_guard.sh` 方式调用，
# 自身命令行不含 guard 脚本路径，故 pgrep -f 不会自伤。
set -u
SCRIPT=/root/autodl-tmp/hrplus/ops/hard_deadline_guard.py
RUN=/root/autodl-tmp/audit/audit_20260916_A
NEWDL="${1:-2026-09-19T20:00:00+08:00}"
PAT='hard_deadline_guard\.py'

echo "[$(date '+%F %T')] 当前 guard 进程："
pgrep -af "$PAT" || echo "  (无)"

echo "[$(date '+%F %T')] 停止旧 guard："
for p in $(pgrep -f "$PAT"); do
  echo "  kill $p"
  kill "$p" 2>/dev/null
done
sleep 2
if pgrep -f "$PAT" > /dev/null; then echo "  !! 仍有残留，强杀"; for p in $(pgrep -f "$PAT"); do kill -9 "$p" 2>/dev/null; done; fi
echo "  清理完成"

echo "[$(date '+%F %T')] 校验脚本本体："
ls -la "$SCRIPT"
printf "  sha256(前16) = "; sha256sum "$SCRIPT" | cut -c1-16

echo "[$(date '+%F %T')] 启动新 guard，deadline=$NEWDL"
cd /root/autodl-tmp/hrplus/ops
setsid nohup /root/miniconda3/bin/python "$SCRIPT" run \
  --deadline "$NEWDL" \
  --instance-id autodl-container-cb6a4f850d-6aa7b741 \
  --run-dir "$RUN" \
  --poll 20 --grace 300 >> "$RUN/guard.out" 2>&1 < /dev/null &
sleep 5

echo "[$(date '+%F %T')] 存活确认："
for p in $(pgrep -f "$PAT"); do printf "  pid=%s up=%s\n" "$p" "$(ps -o etime= -p "$p" | tr -d ' ')"; done
echo -n "  实际 deadline 参数 = "
G=$(pgrep -f "$PAT" | head -1)
tr '\0' ' ' < /proc/$G/cmdline | sed 's/.*--deadline \([^ ]*\).*/\1/'

echo "[$(date '+%F %T')] guard.log 尾部："
tail -8 "$RUN/guard.log"

echo "[$(date '+%F %T')] 训练进程未受影响确认："
ps -eo pid,pcpu,etime,args | awk '/train_T0T1/ && !/awk/ {printf "  pid=%s cpu=%s%% up=%s variant=%s\n", $1,$2,$3,$NF}'
grep -aE 'iter=' /root/autodl-tmp/audit/T1.log | tail -1
