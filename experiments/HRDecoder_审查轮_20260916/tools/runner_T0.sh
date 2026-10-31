#!/bin/bash
# 顺序 runner：train 缓存 -> val 缓存 -> T0 smoke(20步) -> 正式 T0
# 依赖满足即自动接续；每步核对退出码与产物。
export PATH=/root/miniconda3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
PY=/root/miniconda3/bin/python
REPO=/root/autodl-tmp/hrplus/repo
OPS=/root/autodl-tmp/hrplus/ops
RUN=/root/autodl-tmp/audit/audit_20260916_A
LOGD=/root/autodl-tmp/audit
MODELS='configs/lesion/exp/hrplus_patch_dice.py=/root/autodl-tmp/hrplus/D_0913_retry/best_mIoU_iter_2000.pth:3.0;configs/lesion/exp/hrplus_mares.py=/root/autodl-tmp/hrplus/results/E_0916/best_mIoU_iter_1500.pth:2.0;configs/lesion/exp/hr100_b.py=/root/autodl-tmp/hr100/trial_b/best_mIoU_iter_3000.pth:1.0;configs/lesion/exp/e4_detail_split613.py=/root/HRDecoder/work_dirs/exp_e4_detail_split613/best_mIoU_iter_36000.pth:1.0'
cd $REPO; export PYTHONPATH=$REPO

echo "[$(date '+%T')] 等 train 缓存结束"
for i in $(seq 1 240); do
  n=$(ls /dev/shm/hrdec_cache/base0 2>/dev/null | wc -l)
  p=$(ps -eo args | grep -c "[/]ops/build_cache.py")
  if [ "$p" -eq 0 ] && [ "$n" -ge 454 ]; then break; fi
  sleep 20
done
echo "[$(date '+%T')] train 缓存 = $(ls /dev/shm/hrdec_cache/base0 | wc -l)/454"

echo "[$(date '+%T')] STEP val 缓存"
mkdir -p /dev/shm/hrdec_cache_val
$PY $OPS/build_cache.py --split val --cache-dir /dev/shm/hrdec_cache_val --out-dir $RUN --flip --models "$MODELS" > $LOGD/cache_val.log 2>&1
echo "  rc=$? val=$(ls /dev/shm/hrdec_cache_val/base0 2>/dev/null | wc -l)/76"

echo "[$(date '+%T')] STEP T0 smoke(20 步)"
$PY $OPS/train_T0T1.py --variant T0 --out-dir $RUN --seed 1 --smoke 20 > $LOGD/T0_smoke.log 2>&1
RC=$?
echo "  rc=$RC"; tail -6 $LOGD/T0_smoke.log
if [ "$RC" -ne 0 ]; then echo "[$(date '+%T')] smoke 失败，停止（FAILED_IMPLEMENTATION）"; exit 1; fi

echo "[$(date '+%T')] STEP 正式 T0 (8000 步)"
$PY $OPS/train_T0T1.py --variant T0 --out-dir $RUN --seed 1 --iters 8000 --val-every 1000 > $LOGD/T0.log 2>&1
echo "  rc=$?"
tail -20 $LOGD/T0.log
echo "[$(date '+%T')] T0 结束"
