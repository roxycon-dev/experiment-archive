#!/usr/bin/env bash
# =============================================================================
# finalize_after_training.sh -- post-training guard for the cs4000_v2 run.
#
# Usage:
#   bash finalize_after_training.sh <TRAIN_PID> <BASE_DIR> <SETTINGS_JSON> <BACKUP_DIR>
#
# What it does, in order:
#   1. Waits for <TRAIN_PID> to exit (polls /proc; guards against PID reuse).
#   2. Verifies training artifacts under <BASE_DIR>/cs4000_v2 are complete:
#        C_summary.json, S_summary.json,
#        C_step_2000.pth, C_step_4000.pth, S_step_2000.pth, S_step_4000.pth,
#        C_val_2000_counts.json, C_val_4000_counts.json,
#        S_val_2000_counts.json, S_val_4000_counts.json
#      Any missing / empty / corrupt file => exit non-zero immediately.
#   3. Evaluates the FOUR candidates on 'val' by calling eval_candidates.py
#      ONE AT A TIME (strictly serial, to bound GPU memory).  The champion
#      probability cache directory is FIXED across all four runs so runs
#      2..4 reuse the cache built by run 1 (identity-checked by the eval
#      script itself).
#      VAL ONLY: the command always passes '--split val' and NEVER passes
#      '--allow-test' / '--locked-candidate'.  This script can never run the
#      test split.
#   4. Only after all four eval runs succeed, packs the raw eval outputs and
#      the training logs into <BACKUP_DIR>/finalize_<ts>.tar.gz, verifies the
#      archive (gzip -t, tar -t, entry count, sha256 computed twice and
#      compared, .sha256 sidecar written) and ONLY THEN atomically creates
#      <BASE_DIR>/READY_TO_SHUTDOWN (tmp file + rename in the same dir).
#
# Hard guarantees:
#   * This script NEVER runs test, NEVER reboots, NEVER powers off, and
#     NEVER calls shutdown.  READY_TO_SHUTDOWN is a flag file only; the
#     actual shutdown is a separate human decision (or an external watchdog
#     that consumes the flag).
#   * Any failure => non-zero exit and NO READY_TO_SHUTDOWN flag.
#   * eval_candidates.py is protocol-frozen: this script only CALLS it and
#     never modifies it.
#
# Exit codes:
#   0  success (READY_TO_SHUTDOWN created)
#   2  usage / settings error
#   3  environment error, lock conflict, or termination signal
#   4  training artifact completeness failure
#   5  eval run failure
#   6  packaging / verification / final-gate failure
# =============================================================================
set -uo pipefail

POLL_INTERVAL="${FINALIZE_POLL_SEC:-30}"
case "$POLL_INTERVAL" in ''|*[!0-9]*) POLL_INTERVAL=30 ;; esac
RUN_DIR_NAME="cs4000_v2"
OUT_ROOT_NAME="evalcand_val"
CHAMP_CACHE_NAME="champ_prob_cache_val"
STEPS_EXPECTED=4000

log() { printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"; }
die() { local code="$1"; shift; log "FATAL: $*"; exit "$code"; }

# ------------------------------------------------------------------ arguments
if [ "$#" -ne 4 ]; then
  log "usage: bash $0 <TRAIN_PID> <BASE_DIR> <SETTINGS_JSON> <BACKUP_DIR>"
  exit 2
fi
TRAIN_PID="$1"; BASE_DIR="$2"; SETTINGS="$3"; BACKUP_DIR="$4"

case "$TRAIN_PID" in ''|*[!0-9]*)
  log "FATAL: TRAIN_PID must be a positive integer, got '$TRAIN_PID'"; exit 2 ;;
esac
[ "$TRAIN_PID" -gt 1 ] || { log "FATAL: refusing PID 0/1"; exit 2; }
[ -d "$BASE_DIR" ] || { log "FATAL: BASE_DIR is not a directory: $BASE_DIR"; exit 2; }
[ -f "$SETTINGS" ] || { log "FATAL: SETTINGS is not a file: $SETTINGS"; exit 2; }

abspath() { (cd -- "$(dirname -- "$1")" && printf '%s/%s\n' "$(pwd -P)" "$(basename -- "$1")"); }
BASE_DIR="$(abspath "$BASE_DIR")"
SETTINGS="$(abspath "$SETTINGS")"
mkdir -p "$BACKUP_DIR" || { log "FATAL: cannot create BACKUP_DIR: $BACKUP_DIR"; exit 2; }
BACKUP_DIR="$(abspath "$BACKUP_DIR")"

RUN_DIR="$BASE_DIR/$RUN_DIR_NAME"
OUT_ROOT="$BASE_DIR/$OUT_ROOT_NAME"
CHAMP_CACHE="$BASE_DIR/$CHAMP_CACHE_NAME"
FLAG="$BASE_DIR/READY_TO_SHUTDOWN"
FINALIZE_LOG="$BASE_DIR/finalize_after_training_$(date +%Y%m%d_%H%M%S).log"
LOCK_DIR="$BASE_DIR/.finalize_after_training.lock"
STAGE_DIR="$BACKUP_DIR/.finalize_stage.$$"

# ------------------------------------------------ self logging (tee to file)
exec > >(tee -a "$FINALIZE_LOG") 2>&1
log "finalize_after_training starting; pid=$$"
log "TRAIN_PID=$TRAIN_PID"
log "BASE_DIR=$BASE_DIR"
log "SETTINGS=$SETTINGS"
log "BACKUP_DIR=$BACKUP_DIR"
log "finalize log: $FINALIZE_LOG"

cleanup() {
  # Defensive: only ever remove the exact staging dir this instance created.
  case "$STAGE_DIR" in
    "$BACKUP_DIR"/.finalize_stage.*) rm -rf -- "$STAGE_DIR" 2>/dev/null || true ;;
  esac
  [ -n "${LOCK_DIR:-}" ] && rmdir -- "$LOCK_DIR" 2>/dev/null || true
  return 0
}
on_signal() { log "FATAL: termination signal received; aborting (no flag created)"; exit 3; }
trap on_signal TERM INT HUP
trap cleanup EXIT

# ------------------------------------------------------------------ lock
mkdir "$LOCK_DIR" 2>/dev/null \
  || die 3 "another finalize_after_training instance appears to be running (lock: $LOCK_DIR)"

# ------------------------------------------------------------------ environment
[ "$(uname -s)" = "Linux" ] || die 3 "this script requires Linux (/proc, GNU tar, sha256sum, gzip)"
PYTHON="${PYTHON:-python3}"
command -v "$PYTHON" >/dev/null 2>&1 || die 3 "python3 not found (override with PYTHON=...)"
command -v sha256sum >/dev/null 2>&1 || die 3 "sha256sum not found"
command -v gzip >/dev/null 2>&1 || die 3 "gzip not found"
command -v tar >/dev/null 2>&1 || die 3 "tar not found"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
[ -f "$SCRIPT_DIR/eval_candidates.py" ] \
  || die 3 "eval_candidates.py not found next to this script: $SCRIPT_DIR"

json_ok() {
  "$PYTHON" -c 'import json,sys; json.load(open(sys.argv[1], encoding="utf-8"))' \
    "$1" >/dev/null 2>&1
}
settings_keys_ok() {
  "$PYTHON" - "$1" <<'PYEOF' >/dev/null 2>&1
import json, sys
s = json.load(open(sys.argv[1], encoding='utf-8'))
req = ('repo', 'config', 'data_root', 'student_checkpoint',
       'reference_checkpoint', 'champion')
sys.exit(1 if [k for k in req if not s.get(k)] else 0)
PYEOF
}
json_ok "$SETTINGS" || die 2 "settings JSON does not parse: $SETTINGS"
settings_keys_ok "$SETTINGS" \
  || die 2 "settings JSON missing required keys (repo/config/data_root/student_checkpoint/reference_checkpoint/champion): $SETTINGS"

# ------------------------------------------------------------- wait for PID
REQUIRED_FILES=(
  C_summary.json S_summary.json
  C_step_2000.pth C_step_4000.pth S_step_2000.pth S_step_4000.pth
  C_val_2000_counts.json C_val_4000_counts.json
  S_val_2000_counts.json S_val_4000_counts.json
)
CAND_NAMES=(C_step_2000 C_step_4000 S_step_2000 S_step_4000)
CAND_ARMS=(C C S S)

pid_alive() {
  kill -0 "$TRAIN_PID" 2>/dev/null || return 1
  # A zombie still answers kill -0; treat it as exited.
  local state
  state="$(sed -e 's/^.*) //' "/proc/$TRAIN_PID/stat" 2>/dev/null | awk '{print $1}')"
  [ "$state" = "Z" ] && return 1
  return 0
}

# Snapshot the real stdout/stderr targets of the training process now, so the
# training log can be archived later regardless of its file name.
declare -a TRAIN_LOGS=()
add_log() {
  local f="$1" r
  { [ -n "$f" ] && [ -f "$f" ] && [ -s "$f" ]; } || return 0
  for r in ${TRAIN_LOGS[@]+"${TRAIN_LOGS[@]}"}; do
    [ "$r" = "$f" ] && return 0
  done
  TRAIN_LOGS+=("$f")
  return 0
}
for fd in 1 2; do
  tgt="$(readlink "/proc/$TRAIN_PID/fd/$fd" 2>/dev/null || true)"
  tgt="${tgt/ (deleted)/}"
  case "$tgt" in
    ''|/dev/*|pipe:*|socket:*|/proc/*|anon_inode:*) ;;
    *) add_log "$tgt" ;;
  esac
done

if pid_alive; then
  INIT_CMD="$(tr '\0' ' ' <"/proc/$TRAIN_PID/cmdline" 2>/dev/null || true)"
  log "waiting for training PID $TRAIN_PID to exit (poll every ${POLL_INTERVAL}s)"
  [ -n "$INIT_CMD" ] && log "training cmdline: $INIT_CMD"
  polls=0
  while pid_alive; do
    cur_cmd="$(tr '\0' ' ' <"/proc/$TRAIN_PID/cmdline" 2>/dev/null || true)"
    if [ -n "$INIT_CMD" ] && [ -n "$cur_cmd" ] && [ "$cur_cmd" != "$INIT_CMD" ]; then
      log "PID $TRAIN_PID was reused by another process; treating the original training process as ended"
      break
    fi
    polls=$((polls + 1))
    if [ $((polls % 20)) -eq 0 ]; then
      log "still waiting for training PID $TRAIN_PID ($polls polls)"
    fi
    sleep "$POLL_INTERVAL"
  done
  log "training PID $TRAIN_PID has exited"
else
  log "WARNING: training PID $TRAIN_PID is not running (already exited, or wrong PID);"
  log "proceeding to the artifact check, which fails closed if training did not produce its files"
fi

# ------------------------------------------------------- artifact completeness
log "checking training artifacts under $RUN_DIR"
[ -d "$RUN_DIR" ] || die 4 "run dir missing: $RUN_DIR"
missing=0
for f in "${REQUIRED_FILES[@]}"; do
  p="$RUN_DIR/$f"
  if [ ! -s "$p" ]; then
    log "MISSING/EMPTY: $p"
    missing=$((missing + 1))
  fi
done
[ "$missing" -eq 0 ] || die 4 "training artifact check failed: $missing file(s) missing/empty"
for f in C_summary.json S_summary.json; do
  json_ok "$RUN_DIR/$f" || die 4 "$RUN_DIR/$f is not valid JSON"
done
summary_steps_ok() {
  "$PYTHON" - "$1" "$2" "$STEPS_EXPECTED" <<'PYEOF' >/dev/null 2>&1
import json, sys
s = json.load(open(sys.argv[1], encoding='utf-8'))
ok = (s.get('arm') == sys.argv[2] and s.get('steps') == int(sys.argv[3])
      and isinstance(s.get('history'), list) and len(s['history']) >= 1)
sys.exit(0 if ok else 1)
PYEOF
}
summary_steps_ok "$RUN_DIR/C_summary.json" C \
  || die 4 "C_summary.json invalid (arm/steps/history mismatch)"
summary_steps_ok "$RUN_DIR/S_summary.json" S \
  || die 4 "S_summary.json invalid (arm/steps/history mismatch)"
log "training artifacts complete: 10/10 files present, summaries valid"

# ------------------------------------------- val evaluation of the 4 candidates
# VAL ONLY.  The eval command below is hard-coded to '--split val' and never
# receives '--allow-test' / '--locked-candidate' / '--write-lock'.
log "starting val evaluation of 4 candidates (strictly serial, one GPU process at a time)"
mkdir -p "$CHAMP_CACHE" || die 3 "cannot create champion cache dir: $CHAMP_CACHE"
log "fixed champion cache dir: $CHAMP_CACHE (run 1 builds it, runs 2-4 reuse it; "
log "NOTE: it is bound to this exact settings JSON -- delete it if settings ever change)"

for i in "${!CAND_NAMES[@]}"; do
  name="${CAND_NAMES[$i]}"
  arm="${CAND_ARMS[$i]}"
  ckpt="$RUN_DIR/$name.pth"
  out_dir="$OUT_ROOT/$name"
  [ -s "$ckpt" ] || die 4 "candidate checkpoint missing/empty: $ckpt"
  mkdir -p "$out_dir" || die 5 "cannot create eval out-dir: $out_dir"
  log "=== eval $((i + 1))/4: candidate=$name arm=$arm out=$out_dir"
  "$PYTHON" "$SCRIPT_DIR/eval_candidates.py" \
      --settings "$SETTINGS" \
      --out-dir "$out_dir" \
      --prefix evalcand \
      --split val \
      --candidate "$name=$ckpt" \
      --candidate-arm "$name=$arm" \
      --champion-cache-dir "$CHAMP_CACHE" \
      2>&1 | tee "$out_dir/run.log"
  rc="${PIPESTATUS[0]}"
  if [ "$rc" -ne 0 ]; then
    die 5 "eval_candidates.py failed for $name (exit $rc); see $out_dir/run.log"
  fi
  for f in evalcand_summary.json evalcand_champion_counts.json \
           "evalcand_cand_${name}_counts.json" "evalcand_mix_${name}_counts.json"; do
    [ -s "$out_dir/$f" ] || die 5 "eval output missing after success: $out_dir/$f"
  done
  json_ok "$out_dir/evalcand_summary.json" \
    || die 5 "eval summary is not valid JSON: $out_dir/evalcand_summary.json"
  log "=== eval $((i + 1))/4 OK: $name"
done
log "all 4 val evaluations succeeded"

# ------------------------------------------------- packaging (tar.gz + sha256)
log "packaging raw outputs and training logs into $BACKUP_DIR"
TS="$(date +%Y%m%d_%H%M%S)"
TAR_TMP="$BACKUP_DIR/finalize_${TS}.tar.gz.tmp"
TAR_FINAL="$BACKUP_DIR/finalize_${TS}.tar.gz"
SHA_TMP="$TAR_TMP.sha256"
SHA_FINAL="$TAR_FINAL.sha256"

mkdir -p "$STAGE_DIR/finalize_logs" "$STAGE_DIR/training_logs" \
  || die 6 "cannot create staging dir: $STAGE_DIR"
cp -f -- "$FINALIZE_LOG" "$STAGE_DIR/finalize_logs/finalize_after_training.log" \
  || die 6 "cannot snapshot finalize log"

rel_under_base() {
  case "$1" in
    "$BASE_DIR"/*) printf '%s\n' "${1#"$BASE_DIR"/}" ; return 0 ;;
    *) return 1 ;;
  esac
}

# Raw training outputs: the small JSON reports only.  The .pth checkpoints and
# the champion probability cache are deliberately NOT archived (large; they
# remain on disk under BASE_DIR).
declare -a ARCH_REL=()
n_json=0
for f in "$RUN_DIR"/*.json; do
  [ -s "$f" ] || continue
  ARCH_REL+=("$RUN_DIR_NAME/$(basename -- "$f")")
  n_json=$((n_json + 1))
done
[ "$n_json" -ge 6 ] || die 6 "unexpectedly few JSON reports under $RUN_DIR"
[ -d "$OUT_ROOT" ] && ARCH_REL+=("$OUT_ROOT_NAME")

# Training logs: targets captured from /proc plus conventional locations.
for f in "$BASE_DIR"/*.log "$BASE_DIR"/nohup.out "$RUN_DIR"/*.log; do
  [ -f "$f" ] || continue
  [ "$f" = "$FINALIZE_LOG" ] && continue      # archived via stage snapshot only
  add_log "$f"
done
declare -a STAGE_RELS=("finalize_logs" "training_logs")
ext_i=0
for f in ${TRAIN_LOGS[@]+"${TRAIN_LOGS[@]}"}; do
  if rel="$(rel_under_base "$f")"; then
    ARCH_REL+=("$rel")
  else
    ext_i=$((ext_i + 1))
    dest="training_logs/log_$(printf '%02d' "$ext_i")_$(basename -- "$f")"
    cp -f -- "$f" "$STAGE_DIR/$dest" || die 6 "cannot copy training log: $f"
  fi
done
if [ "$ext_i" -eq 0 ] && [ "${#TRAIN_LOGS[@]}" -eq 0 ]; then
  log "WARNING: no training log file discovered; archive will still contain all JSON reports and eval logs"
fi

log "creating archive (this may take a moment)"
tar -C "$BASE_DIR" -czf "$TAR_TMP" "${ARCH_REL[@]}" \
    -C "$STAGE_DIR" "${STAGE_RELS[@]}" \
  || die 6 "tar creation failed"
[ -s "$TAR_TMP" ] || die 6 "archive is empty"
gzip -t "$TAR_TMP" || die 6 "gzip integrity check failed"
tar -tzf "$TAR_TMP" >/dev/null 2>&1 || die 6 "tar listing failed"
n_entries="$(tar -tzf "$TAR_TMP" | wc -l)"
[ "${n_entries:-0}" -ge 12 ] || die 6 "archive has too few entries ($n_entries)"
DIGEST="$(sha256sum "$TAR_TMP" | awk '{print $1}')"
[ -n "$DIGEST" ] || die 6 "sha256 computation failed"
DIGEST2="$(sha256sum "$TAR_TMP" | awk '{print $1}')"
[ "$DIGEST" = "$DIGEST2" ] || die 6 "sha256 unstable across reads"
log "archive verified: $n_entries entries, sha256=$DIGEST"

printf '%s  %s\n' "$DIGEST" "$(basename -- "$TAR_FINAL")" > "$SHA_TMP" \
  || die 6 "cannot write sha256 sidecar"
mv -f "$TAR_TMP" "$TAR_FINAL" || die 6 "atomic rename of archive failed"
mv -f "$SHA_TMP" "$SHA_FINAL" || die 6 "atomic rename of sha256 sidecar failed"
log "archive written: $TAR_FINAL"
log "sha256 sidecar:  $SHA_FINAL"

# ------------------------------------------------------ final gate + flag gate
for f in "${REQUIRED_FILES[@]}"; do
  [ -s "$RUN_DIR/$f" ] || die 4 "final gate: $f disappeared"
done
for name in "${CAND_NAMES[@]}"; do
  [ -s "$OUT_ROOT/$name/evalcand_summary.json" ] \
    || die 5 "final gate: eval summary missing for $name"
done
[ -s "$TAR_FINAL" ] || die 6 "final gate: archive missing"
[ -s "$SHA_FINAL" ] || die 6 "final gate: sha256 sidecar missing"

# Atomic flag creation: tmp file + rename within the same directory.
FLAG_TMP="$BASE_DIR/.READY_TO_SHUTDOWN.$$"
{
  echo "READY_TO_SHUTDOWN"
  echo "created_utc: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  echo "archive: $TAR_FINAL"
  echo "sha256: $DIGEST"
  echo "eval_out_root: $OUT_ROOT"
  echo "candidates_evaluated: ${CAND_NAMES[*]}"
} > "$FLAG_TMP" || die 6 "cannot write flag temp file"
sync "$FLAG_TMP" 2>/dev/null || true
mv -f "$FLAG_TMP" "$FLAG" || die 6 "atomic flag rename failed"

log "READY_TO_SHUTDOWN created atomically: $FLAG"
log "NOTE: this script does NOT shut the machine down; shutdown is a separate human decision."
log "ALL DONE (exit 0)"
exit 0
