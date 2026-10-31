#!/bin/bash
# E 实验启动器：传入 config 与 work-dir，训练完不自动关机（后面还要串实验）
export PATH=/root/miniconda3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
PY=/root/miniconda3/bin/python
REPO=/root/autodl-tmp/hrplus/repo
CFG=$1
WD=$2
export PYTHONPATH=$REPO
export OMP_NUM_THREADS=4
cd $REPO
mkdir -p "$WD"
echo "[$(date '+%F %T')] E_START cfg=$CFG wd=$WD" >> "$WD/e_stage.log"
$PY tools/train.py "$CFG" --work-dir "$WD" >> "$WD/e_train.log" 2>&1
RC=$?
echo "[$(date '+%F %T')] E_DONE rc=$RC" >> "$WD/e_stage.log"
echo "$RC" > "$WD/train_returncode"
