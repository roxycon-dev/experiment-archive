#!/usr/bin/env bash
# Only run on the authorized paid Linux instance.
set -Eeuo pipefail
[[ "$(uname -s)" == Linux ]] || { echo 'Linux server only' >&2; exit 2; }
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
PYTHON="${PYTHON:-python}"
SCRIPT="${SCRIPT:-run_protocol.py}"
RUN_DIR="${2:-}"
BACKUP_DIR="${3:-}"
DEADLINE_EPOCH=0
RUN_START=$(date +%s)
cleanup() {
  local rc=$? remaining cap elapsed
  trap - EXIT INT TERM
  set +e
  elapsed=$(( $(date +%s) - ${RUN_START:-0} ))
  # FAST-FAIL GUARD: a protocol that dies within 120s with an unexpected
  # status is almost always a code bug, not a budget end.  Do NOT shut the
  # instance down in that case so the failure can be diagnosed on the spot
  # (the independent watchdog still enforces the ledger deadline, so the
  # instance can never run unbounded).
  case "$rc" in
    0|3|130|143) ;;
    *) if (( elapsed < 120 )); then
         echo "FASTFAIL rc=$rc after ${elapsed}s -- skipping shutdown; watchdog still armed" \
           | tee -a "$RUN_DIR/FASTFAIL.txt" >&2
         exit "$rc"
       fi ;;
  esac
  remaining=$((DEADLINE_EPOCH - $(date +%s) - 10))
  if [[ -d "$RUN_DIR" && -n "$BACKUP_DIR" && "$remaining" -gt 0 ]]; then
    cap=300; (( cap > remaining )) && cap=$remaining
    # NOTE: regenerable feature caches are excluded -- a 24GB cache inside a
    # full-disk backup is exactly what truncated the 14:00 backup attempt.
    timeout --kill-after=5s "${cap}s" tar --exclude='featcache_r6' -cf "$BACKUP_DIR/run_$(date +%Y%m%dT%H%M%S).tar" -C "$(dirname "$RUN_DIR")" "$(basename "$RUN_DIR")"
  fi
  # Never cancel the independent watchdog; AutoDL shutdown is immediate.
  for attempt in 1 2 3; do
    shutdown -h now && exit "$rc"
    sleep 2
  done
  echo 'STOP FAILED: stop through the AutoDL console immediately' >&2
  exit 99
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
[[ $# == 3 ]] || { echo 'Usage: run_guarded.sh settings.json run_dir durable_backup_dir' >&2; exit 2; }
SETTINGS_FILE="$(realpath "$1")"
mkdir -p "$RUN_DIR" "$BACKUP_DIR"
RUN_DIR="$(realpath "$RUN_DIR")"
BACKUP_DIR="$(realpath "$BACKUP_DIR")"
case "$BACKUP_DIR/" in "$RUN_DIR/"*) echo 'Backup must be outside run directory' >&2; exit 2;; esac
DEADLINE_EPOCH="$("$PYTHON" - "$RUN_DIR" <<'PY'
import sys, ops
print(int(ops.deadline_of(ops.load_ledger(sys.argv[1])).timestamp()))
PY
)"
nohup "$PYTHON" -u "$SCRIPT_DIR/ops.py" watchdog --run-dir "$RUN_DIR" >"$RUN_DIR/watchdog.log" 2>&1 &
WATCHDOG_PID=$!
sleep 1
kill -0 "$WATCHDOG_PID"
REMAINING=$((DEADLINE_EPOCH - $(date +%s) - 15))
(( REMAINING > 0 )) || exit 3
timeout --signal=TERM --kill-after=10s "${REMAINING}s" "$PYTHON" -u "$SCRIPT_DIR/$SCRIPT" --settings "$SETTINGS_FILE" --run-dir "$RUN_DIR"

