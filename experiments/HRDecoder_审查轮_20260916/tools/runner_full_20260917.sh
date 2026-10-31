#!/bin/bash
# 2026-09-17 全串行实验链（严格单任务，绝不并发）
#   smokeA(新T0,20步) -> T0 -> smokeB(从 head_T0 续训20步,验 E1 路径) -> T1 -> U0
#   -> E1 判定门(图像级配对 bootstrap) -> [条件] E1 配对续训 2x4000
#   -> V1 两臂 x seed{2,3} 共 4 次
# 契约 val 点已随补丁生效：0/1000/2000/4000/6000/8000
export PATH=/root/miniconda3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
PY=/root/miniconda3/bin/python
OPS=/root/autodl-tmp/hrplus/ops
REPO=/root/autodl-tmp/hrplus/repo
RUN=/root/autodl-tmp/audit/audit_20260916_A
LOGD=/root/autodl-tmp/audit
ITERS=8000
PAT="train_T0""T1.py"
cd $REPO; export PYTHONPATH=$REPO
mkdir -p $RUN/E1 $RUN/_smoke $RUN/_smoke2 $RUN/V1/s2_T0 $RUN/V1/s2_T1 $RUN/V1/s3_T0 $RUN/V1/s3_T1 $RUN/experiments

# 严格串行：启动前拒绝任何已有训练进程
if pgrep -f "$PAT" > /dev/null 2>&1; then
  echo "[$(date '+%F %T')] !! 已有训练进程在跑，拒绝启动（严格串行原则）"; exit 9
fi
echo "[$(date '+%F %T')] ===== 全链启动（严格串行）====="
echo "  补丁脚本 sha256: $(sha256sum $OPS/train_T0T1.py | cut -c1-16)"

FAILED=""
ran () {  # $1=tag $2=必须成功 flag  $3...=args
  local tag=$1; local must=$2; shift 2
  echo "[$(date '+%F %T')] ===== 启动 $tag ====="
  $PY $OPS/train_T0T1.py "$@" > $LOGD/$tag.log 2>&1
  local rc=$?
  echo "[$(date '+%F %T')] $tag 结束 rc=$rc"
  grep -aE '^\[val\]|^\[init\]|^\[U0\]|训练完成|val 曲线|全 ignore 跳过步|无 MA 阳性的 batch|采样来源计数|Traceback|Error|Assertion' $LOGD/$tag.log | tail -14
  if [ $rc -ne 0 ]; then
    FAILED="$FAILED $tag"
    echo "  !! $tag 失败 rc=$rc" | tee -a $LOGD/FAILED_ARMS.txt
    [ "$must" = "must" ] && { echo "[$(date '+%F %T')] 关键臂失败 -> 中止链条"; summarize; exit 1; }
  fi
  return 0
}
summarize () {
  echo "[$(date '+%F %T')] 失败臂:${FAILED:- 无}"
  echo "[$(date '+%F %T')] 产物:"; ls -la $RUN/*.pth $RUN/*.json $RUN/E1/*.pth 2>/dev/null | tail -12
}

# ---------- 1. smoke A：全新 T0，20 步（验 val@0 与补丁）----------
ran SMOKE_A must --variant T0 --out-dir $RUN/_smoke --seed 1 --iters 8000 --val-every 1000 --smoke 20
grep -qa '^\[val\] iter=0' $LOGD/SMOKE_A.log || { echo "[$(date '+%F %T')] !! smokeA 未产出 val@0，补丁未生效 -> 中止"; exit 1; }
echo "[$(date '+%F %T')] smokeA 通过：val@0 已产出"

# ---------- 2. T0 正式 ----------
ran T0v2 must --variant T0 --out-dir $RUN --seed 1 --iters $ITERS --val-every 1000

# ---------- 3. smoke B：从 head_T0 续训 20 步（验 E1 的 --init/--fixed-lr/rng_state）----------
ran SMOKE_B must --variant T0 --out-dir $RUN/_smoke2 --seed 1 --iters 4000 --val-every 1000 \
    --init $RUN/head_T0.pth --fixed-lr 1e-5 --smoke 20
grep -qa '采样 RNG 状态已恢复' $LOGD/SMOKE_B.log \
  && echo "[$(date '+%F %T')] smokeB 通过：续训 + rng_state 恢复路径可用" \
  || echo "[$(date '+%F %T')] !! smokeB 未确认 rng_state 恢复（E1 将退化为重放采样），继续但已登记"

# ---------- 4. T1 正式 ----------
ran T1v2 must --variant T1 --out-dir $RUN --seed 1 --iters $ITERS --val-every 1000

# ---------- 5. U0 正式 ----------
ran U0v2 must --variant U0 --out-dir $RUN --seed 1 --iters $ITERS --val-every 1000

# ---------- 6. E1 判定门 ----------
echo "[$(date '+%F %T')] ===== E1 判定门 ====="
GATE=$OPS/e1_gate.py
GATE_RC=2
if [ -f $GATE ]; then
  $PY $GATE --t0 $RUN/val_counts_T0_s1_it8000.json --t1 $RUN/val_counts_T1_s1_it8000.json \
      --out $RUN/experiments/E1_gate.json > $LOGD/E1_gate.log 2>&1
  GATE_RC=$?
  echo "  gate rc=$GATE_RC"; grep -aE 'DECISION|CI|delta|TRIGGER|SKIP|Δ' $LOGD/E1_gate.log | tail -8
else
  echo "  !! 判定门脚本缺失：$GATE"; echo "GATE_UNDECIDED" >> $LOGD/E1_gate.log
fi

# ---------- 7. 条件 E1 ----------
if [ $GATE_RC -eq 0 ] && grep -qa 'TRIGGER' $LOGD/E1_gate.log; then
  echo "[$(date '+%F %T')] E1 触发 -> 配对续训 2x4000 步（固定 lr=1e-5，真续训）"
  ran E1_T0 must --variant T0 --out-dir $RUN/E1 --seed 1 --iters 4000 --val-every 1000 \
      --init $RUN/head_T0.pth --fixed-lr 1e-5
  ran E1_T1 must --variant T1 --out-dir $RUN/E1 --seed 1 --iters 4000 --val-every 1000 \
      --init $RUN/head_T1.pth --fixed-lr 1e-5
else
  echo "[$(date '+%F %T')] E1 未触发（或判定门不可用）-> 跳过 E1"
  echo "E1_SKIPPED gate_rc=$GATE_RC $(date '+%F %T')" >> $LOGD/E1_gate.log
fi

# ---------- 8. V1：两臂 x seed 2,3 ----------
for SEED in 2 3; do
  for V in T0 T1; do
    ran V1_${V}_s${SEED} soft --variant $V --out-dir $RUN/V1/s${SEED}_${V} --seed $SEED \
        --iters $ITERS --val-every 1000
  done
done

summarize
echo "[$(date '+%F %T')] ===== 全链结束 ====="
