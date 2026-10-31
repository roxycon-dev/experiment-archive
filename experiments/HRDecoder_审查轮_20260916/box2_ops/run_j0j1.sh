#!/bin/bash
# 在 2 号机上跑：J0 -> J1 串行正式训练（严格单任务；评测点 4000/8000）
set -u
PY=/root/miniconda3/bin/python
R=/root/box1_sync/repo
W=/root/box1_sync/workdir
mkdir -p $W
cd $R && export PYTHONPATH=$R

for ARM in J0 J1; do
  CFG=configs/lesion/exp/$(echo $ARM | tr 'A-Z' 'a-z')_split613.py
  echo "[$(date '+%T')] ===== 启动 $ARM（8000 步，评测 4000/8000）====="
  $PY tools/train.py $CFG --work-dir $W/$ARM --seed 42 \
      --options runner.max_iters=8000 evaluation.interval=4000 \
                checkpoint_config.interval=2000 checkpoint_config.max_keep_ckpts=4 \
      > $W/train_$ARM.log 2>&1
  RC=$?
  echo "[$(date '+%T')] $ARM 结束 rc=$RC"
  grep -aE 'FreezeScope|Saving checkpoint|mIoU|Traceback|AssertionError|CUDA out of memory' $W/train_$ARM.log | tail -10
  echo "  证据: $W/$ARM/J0J1_evidence_$ARM.json"
done
echo "[$(date '+%T')] ===== J0/J1 全链结束 ====="
