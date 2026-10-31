#!/bin/bash
# Resume the FORMAL refiner_v1 run from the first unfinished stage.
#
#   bash refiner_v1/tools/resume_formal_v1.sh
#
# Safe by construction (nothing is deleted, nothing is rebuilt):
#   * refuses if a supervisor is already alive for this run root
#   * archives the previous status.json / supervisor.out (the FAILED evidence)
#   * removes a stale STOP sentinel, which would stop the new run instantly
#   * passes --resume to run_formal_v1.py, which skips every stage whose
#     artefact exists: cache_train/cache_val (index_*.json complete=true) and
#     stage1_train (done.json + node_ep200.pth) are NOT re-run, so the run
#     continues at eval_stage1_ep200 and then stage2.
set -euo pipefail

PY=/root/miniconda3/bin/python
ROOT=/root/autodl-tmp/HRDecoder_FullCompetition_20260920
RUN=$ROOT/refiner_v1/runs/formal_20260921
SCRIPT=$ROOT/refiner_v1/run_formal_v1.py
TS=$(date +%Y%m%d_%H%M%S)

mkdir -p "$RUN/logs"
cd "$ROOT"

if [ -f "$RUN/supervisor.pid" ] && kill -0 "$(cat "$RUN/supervisor.pid")" 2>/dev/null; then
  echo "REFUSING: supervisor already running with pid $(cat "$RUN/supervisor.pid")"
  exit 1
fi

for f in status.json supervisor.out; do
  if [ -f "$RUN/$f" ]; then
    cp "$RUN/$f" "$RUN/${f%.*}.before_resume_$TS.${f##*.}"
    echo "archived $RUN/$f -> $RUN/${f%.*}.before_resume_$TS.${f##*.}"
  fi
done

if [ -f "$RUN/STOP" ]; then
  echo "removing stale STOP sentinel: $RUN/STOP"
  rm -f "$RUN/STOP"
fi

echo "--- stage artefacts that must make the run skip ---"
ls -la "$ROOT/refiner_v1/patch_cache/index_train.json" \
       "$ROOT/refiner_v1/patch_cache/index_val.json" \
       "$RUN/stage1/done.json" "$RUN/stage1/nodes/node_ep200.pth" 2>&1 || true

setsid nohup "$PY" "$SCRIPT" --run-root "$RUN" --resume \
  >> "$RUN/supervisor.out" 2>&1 < /dev/null &
sleep 10

echo "--- supervisor.out (tail) ---"
tail -n 25 "$RUN/supervisor.out" || true
echo "--- pid ---"
cat "$RUN/supervisor.pid" 2>/dev/null || echo "(pid file not written yet)"
echo "--- status head ---"
head -c 900 "$RUN/status.json" 2>/dev/null || echo "(status.json not written yet)"
echo
