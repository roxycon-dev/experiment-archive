#!/bin/bash
# 停止当前 E 训练（保留已有的 best 检查点），为 LR 探针腾出 GPU
export PATH=/root/miniconda3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
echo "=== 停止前 ==="
ps -ef | grep '[t]ools/train.py' | awk '{print $2}'
for p in $(ps -ef | grep '[t]ools/train.py' | awk '{print $2}'); do kill -TERM $p 2>/dev/null; done
sleep 10
for p in $(ps -ef | grep '[t]ools/train.py' | awk '{print $2}'); do kill -9 $p 2>/dev/null; done
sleep 5
echo "=== 停止后 ==="
ps -ef | grep '[t]ools/train.py' | awk '{print $2}' || echo "(无)"
echo "=== guard 仍在 ==="
ps -ef | grep '[d]eadline_guard' | awk '{print $2, $8, $9, $10}'
echo "=== 保留的检查点 ==="
ls -la /root/autodl-tmp/hrplus/results/E_0916/*.pth
echo "=== GPU ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
