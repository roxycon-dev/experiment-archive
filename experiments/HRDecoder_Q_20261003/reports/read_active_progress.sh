set -eu
ROOT=/root/autodl-tmp/HRDecoder_Q_20261003
cd "$ROOT"
tail -n 8 reports/preflight.log
if [ -f reports/gpu_preflight_failure.json ]; then cat reports/gpu_preflight_failure.json; fi
if [ -f reports/formal_launcher.log ]; then tail -n 5 reports/formal_launcher.log; fi
if [ -f reports/formal_Q_DDR_seed42.log ]; then tail -n 5 reports/formal_Q_DDR_seed42.log; fi
if [ -f runs/Q_DDR_seed42/status.json ]; then cat runs/Q_DDR_seed42/status.json; fi
if [ -f reports/preflight_full_val/progress.json ]; then cat reports/preflight_full_val/progress.json; fi
if [ -f reports/gpu_preflight.json ]; then
  /root/miniconda3/envs/hrdecoder_q_20261003/bin/python - <<'PY'
from qlib.common import read_json
r=read_json('reports/gpu_preflight.json')
p=r['performance']
print({k:v for k,v in p.items() if k!='timing_rows'})
print('cost_rows',r['cost_rows'])
PY
fi
