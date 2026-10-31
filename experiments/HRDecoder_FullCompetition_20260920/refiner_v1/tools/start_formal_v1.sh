#!/bin/bash
# Launch the FORMAL refiner_v1 run as a detached, self-monitoring supervisor.
#
#   bash refiner_v1/tools/start_formal_v1.sh
#
# The supervisor owns status.json / supervisor.pid / logs/*.log and runs the
# stages strictly one GPU job at a time:
#   preflight_tests -> cache_train -> cache_val -> stage1_train ->
#   eval_stage1_ep200 -> stage2_train -> eval_stage2_ep{005..030} -> selection
#
# It survives the launching SSH/agent session (setsid + nohup + </dev/null).
set -euo pipefail

PY=/root/miniconda3/bin/python
ROOT=/root/autodl-tmp/HRDecoder_FullCompetition_20260920
RUN=$ROOT/refiner_v1/runs/formal_20260921
SCRIPT=$ROOT/refiner_v1/run_formal_v1.py

mkdir -p "$RUN/logs"
cd "$ROOT"

if [ -f "$RUN/supervisor.pid" ] && kill -0 "$(cat "$RUN/supervisor.pid")" 2>/dev/null; then
  echo "REFUSING: supervisor already running with pid $(cat "$RUN/supervisor.pid")"
  echo "status: $RUN/status.json"
  exit 1
fi

# stale STOP sentinel from a previous attempt would stop the new run instantly
if [ -f "$RUN/STOP" ]; then
  echo "removing stale STOP sentinel: $RUN/STOP"
  rm -f "$RUN/STOP"
fi

setsid nohup "$PY" "$SCRIPT" --run-root "$RUN" \
  >> "$RUN/supervisor.out" 2>&1 < /dev/null &
sleep 8

echo "--- supervisor.out ---"
tail -n 20 "$RUN/supervisor.out" || true
echo "--- pid ---"
cat "$RUN/supervisor.pid" 2>/dev/null || echo "(pid file not written yet)"
echo "--- status ---"
head -c 800 "$RUN/status.json" 2>/dev/null || echo "(status.json not written yet)"
echo
