#!/bin/bash
# 温和停止 L 训练：先杀 launcher（防止它随后关机），再终止 train.py
export PATH=/root/miniconda3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
echo "=== 停止前 ==="
ps -ef | grep -E 'launch_L.sh|tools/train.py' | grep -v grep | awk '{print $2, $8, $9, $10}'

# 1) 先杀 launcher，避免它执行 shutdown
pkill -f 'launch_L.sh' 2>/dev/null || true
sleep 1
pkill -f 'bash /root/launch_L.sh' 2>/dev/null || true
sleep 1

# 2) 再终止训练主进程（SIGTERM，让它有机会收尾），随后清掉子进程
MAIN=$(ps -ef | grep 'tools/train.py' | grep -v grep | awk '$3==1 || $4==1 {print $2}' | head -1)
echo "main train pid = $MAIN"
if [ -n "$MAIN" ]; then
  kill -TERM "$MAIN" 2>/dev/null || true
  sleep 12
fi
REMAIN=$(ps -ef | grep 'tools/train.py' | grep -v grep | awk '{print $2}')
if [ -n "$REMAIN" ]; then
  echo "强制清理残留: $REMAIN"
  kill -TERM $REMAIN 2>/dev/null || true
  sleep 8
fi
pkill -9 -f 'tools/train.py' 2>/dev/null || true
sleep 3

echo
echo "=== 停止后 ==="
ps -ef | grep -E 'launch_L.sh|tools/train.py' | grep -v grep || echo "(无训练进程)"
echo
echo "=== 关机保护仍在？ ==="
ps -ef | grep ops_deadline_guard | grep -v grep || echo "!! guard 不在，需重启"
echo
echo "=== 保留的检查点 ==="
ls -la /root/autodl-tmp/hrplus/results/L_0915/*.pth
echo
echo "=== GPU ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader
echo
echo "=== state.json 是否被误写 complete ==="
cat /root/autodl-tmp/hrplus/results/L_0915/state.json 2>/dev/null || echo "(无 state.json，安全)"
