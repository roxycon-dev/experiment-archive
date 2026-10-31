#!/bin/bash
export PATH=/root/miniconda3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
OUT=/root/autodl-tmp/hrplus/results/L_0915

# 按 PID 精确清理游离的评估进程（避免 pkill 模式误伤本次 SSH 会话）
for p in $(ps -ef | grep 'hrplus/ops/eval' | grep -v grep | awk '{print $2}'); do
  echo "kill eval pid=$p"
  kill -TERM "$p" 2>/dev/null || true
done
sleep 5
for p in $(ps -ef | grep 'hrplus/ops/eval' | grep -v grep | awk '{print $2}'); do
  echo "kill -9 eval pid=$p"
  kill -9 "$p" 2>/dev/null || true
done
sleep 3

# 清理旧的运行脚本
for p in $(ps -ef | grep 'run_evals' | grep -v grep | awk '{print $2}'); do
  echo "kill run_evals pid=$p"
  kill -TERM "$p" 2>/dev/null || true
done
sleep 2

echo '--- 清理后 ---'
ps -ef | grep -E 'hrplus/ops/eval|run_evals' | grep -v grep || echo '(干净)'
echo '--- GPU ---'
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
echo '--- guard 仍在 ---'
ps -ef | grep ops_deadline_guard | grep -v grep || echo '!! guard 丢失'

# 归档旧日志，重新启动完整矩阵
mv $OUT/evals_d.log $OUT/evals_d_first.log 2>/dev/null || true
cd /root
setsid nohup bash run_evals.sh all > $OUT/evals_all.log 2>&1 < /dev/null &
echo "launched, waiting 20s..."
sleep 20
echo '--- evals_all.log ---'
tail -12 $OUT/evals_all.log
echo '--- GPU ---'
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
