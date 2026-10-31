set -eu
ROOT=/root/autodl-tmp/HRDecoder_Q_20261003
ENV=/root/miniconda3/envs/hrdecoder_q_20261003
cd "$ROOT"
if [ -f reports/formal.pid ] && kill -0 "$(cat reports/formal.pid)" 2>/dev/null; then
  echo FORMAL_ALREADY_RUNNING
  exit 0
fi
if [ -f reports/formal_launcher.pid ] && kill -0 "$(cat reports/formal_launcher.pid)" 2>/dev/null; then
  echo FORMAL_LAUNCH_ALREADY_QUEUED
  exit 0
fi
cat > reports/formal_launcher.sh <<'LAUNCH'
set -eu
ROOT=/root/autodl-tmp/HRDecoder_Q_20261003
ENV=/root/miniconda3/envs/hrdecoder_q_20261003
cd "$ROOT"
PREFLIGHT_PID=$(cat reports/preflight.pid)
for ATTEMPT in $(seq 1 240); do
  if grep -q 'FORMAL_PREPARATION_READY' reports/preflight.log; then
    break
  fi
  if [ -f reports/gpu_preflight_failure.json ]; then
    echo TECHNICAL_PAUSE_NO_FORMAL_LAUNCH
    exit 1
  fi
  if ! kill -0 "$PREFLIGHT_PID" 2>/dev/null; then
    echo PREFLIGHT_EXITED_WITHOUT_PASS_NO_FORMAL_LAUNCH
    exit 1
  fi
  sleep 15
done
grep -q 'FORMAL_PREPARATION_READY' reports/preflight.log
"$ENV/bin/python" - <<'PY'
from pathlib import Path
from qlib.common import read_json, utc_now, write_json
root = Path('.')
gate = read_json(root/'reports/formal_readiness.json')
assert gate['status'] == 'READY'
write_json(root/'reports/preparation_status.json', {'stage': 'FORMAL_Q_DDR_SEED42_STARTING',
 'formal_training_started': True, 'user_start_authorization': '20261003_start_task_now',
 'monitor_minutes': 45, 'Q_MIX': 'NOT_READY_MISSING_OFFICIAL_ORIGINAL_IMAGES',
 'shutdown_authorized': True, 'updated_at': utc_now()})
PY
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 nohup "$ENV/bin/python" -u train_run.py --arm Q_DDR --seed 42 --endpoint 16000 > reports/formal_Q_DDR_seed42.log 2>&1 < /dev/null &
echo "$!" > reports/formal.pid
echo FORMAL_Q_DDR_SEED42_LAUNCHED
LAUNCH
nohup /bin/bash reports/formal_launcher.sh > reports/formal_launcher.log 2>&1 < /dev/null &
echo "$!" > reports/formal_launcher.pid
echo FORMAL_START_QUEUED_ON_SUCCESSFUL_REQUIRED_GPU_ACCEPTANCE
