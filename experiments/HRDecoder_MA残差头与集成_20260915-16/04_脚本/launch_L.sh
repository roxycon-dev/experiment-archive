#!/bin/bash
# L 方案训练启动器
# 训练 -> 写 returncode -> 留出收尾时间 -> 关机
export PATH=/root/miniconda3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
PY=/root/miniconda3/bin/python
REPO=/root/autodl-tmp/hrplus/repo
WD=/root/autodl-tmp/hrplus/results/L_0915
CFG=$REPO/configs/lesion/exp/hrplus_patch_lovasz.py
export PYTHONPATH=$REPO
export OMP_NUM_THREADS=4

cd $REPO
mkdir -p $WD

echo "[$(date '+%F %T')] L_STAGE_START" >> $WD/l_stage.log

$PY tools/train.py "$CFG" --work-dir "$WD" >> $WD/l_train.log 2>&1
RC=$?
echo "[$(date '+%F %T')] train finished rc=$RC" >> $WD/l_stage.log
echo "$RC" > $WD/train_returncode
# 让 eval/checkpoint 落盘
sync
sleep 90
echo "[$(date '+%F %T')] L_STAGE_SHUTDOWN" >> $WD/l_stage.log
/root/miniconda3/bin/python - <<'EOF'
import json, os, time
st = {"phase": "complete", "exitcode": int(open('/root/autodl-tmp/hrplus/results/L_0915/train_returncode').read().strip() or -1), "ended_unix": time.time()}
open('/root/autodl-tmp/hrplus/results/L_0915/state.json', 'w').write(json.dumps(st))
print("state written", st)
EOF
bash /usr/bin/shutdown
