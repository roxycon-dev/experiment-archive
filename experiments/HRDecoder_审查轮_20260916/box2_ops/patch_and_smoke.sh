#!/bin/bash
# 在【2 号机】上运行（用 rssh 以 HR_HOST=box2 调用）：打 custom_hooks 补丁 + 重跑 J0 smoke
set -u
PY=/root/miniconda3/bin/python
R=/root/box1_sync/repo
W=/root/box1_sync/workdir/smoke_j0

echo "[$(date '+%T')] 1) 打补丁：mmseg/apis/train.py 恢复 custom_hooks 支持"
$PY /root/box1_sync/ops/patch_custom_hooks.py | tail -14
$PY -m py_compile $R/mmseg/apis/train.py && echo "  语法 OK"

echo "[$(date '+%T')] 2) 重跑 J0 smoke 20 步"
rm -rf $W
cd $R && export PYTHONPATH=$R
timeout 900 $PY tools/train.py configs/lesion/exp/j0_split613.py \
  --work-dir $W --no-validate --seed 42 \
  --options runner.max_iters=20 evaluation.interval=100000 checkpoint_config.interval=20 log_config.interval=5 2>&1 \
  | grep -aE 'FreezeScope|Iter \[|Saving|Traceback|Error|assert|AssertionError' | tail -26

echo "[$(date '+%T')] 3) 证据文件"
cat $W/J0J1_evidence_J0.json 2>/dev/null || echo "  !! 证据文件仍未生成"
echo "[$(date '+%T')] 4) GPU / 产物"
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
ls -la $W 2>/dev/null | head -8
