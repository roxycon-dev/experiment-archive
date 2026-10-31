#!/bin/bash
# Graceful stop for the FORMAL refiner_v1 run.  NOTHING IS DELETED.
#
#   bash refiner_v1/tools/stop_formal_v1.sh
#
# The supervisor notices the STOP sentinel within one poll interval (10 s),
# sends SIGTERM to the current child's process group, marks state=STOPPED and
# leaves every artefact (cache, checkpoints, CSVs, logs) in place.
set -uo pipefail

RUN=/root/autodl-tmp/HRDecoder_FullCompetition_20260920/refiner_v1/runs/formal_20260921
mkdir -p "$RUN"
touch "$RUN/STOP"
echo "STOP sentinel written: $RUN/STOP"
PID=$(cat "$RUN/supervisor.pid" 2>/dev/null || echo "")
if [ -n "$PID" ]; then
  echo "supervisor pid: $PID (still alive until it reaches the next poll)"
  sleep 12
  if kill -0 "$PID" 2>/dev/null; then
    echo "still alive; waiting one more poll interval"
    sleep 12
  fi
  if kill -0 "$PID" 2>/dev/null; then
    echo "still alive -> sending SIGTERM (same graceful path)"
    kill -TERM "$PID"
    sleep 5
  fi
fi
echo "--- status ---"
head -c 900 "$RUN/status.json" 2>/dev/null || echo "(no status.json)"
echo
echo "remaining GPU compute apps (should be empty):"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader || true
