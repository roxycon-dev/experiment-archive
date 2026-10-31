set -eu
ROOT=/root/autodl-tmp/HRDecoder_Q_20261003
ENV=/root/miniconda3/envs/hrdecoder_q_20261003
cd "$ROOT"
grep -q ISOLATED_ENVIRONMENT_READY "$ROOT/reports/bootstrap.log"
if [ -f "$ROOT/reports/preflight.pid" ] && kill -0 "$(cat "$ROOT/reports/preflight.pid")" 2>/dev/null; then
  echo PREFLIGHT_ALREADY_RUNNING
  exit 0
fi
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 nohup "$ENV/bin/python" -u preflight_gpu.py > "$ROOT/reports/preflight.log" 2>&1 < /dev/null &
echo "$!" > "$ROOT/reports/preflight.pid"
echo NECESSARY_GPU_PREFLIGHT_STARTED_NO_FORMAL_TRAINING
