set -eu
ROOT=/root/autodl-tmp/HRDecoder_Q_20261003
if [ -f "$ROOT/reports/bootstrap.log" ]; then tail -n 18 "$ROOT/reports/bootstrap.log"; fi
if [ -f "$ROOT/reports/preflight.log" ]; then tail -n 18 "$ROOT/reports/preflight.log"; fi
if [ -f "$ROOT/reports/gpu_preflight_failure.json" ]; then cat "$ROOT/reports/gpu_preflight_failure.json"; fi
if [ -f "$ROOT/reports/formal_readiness.json" ]; then cat "$ROOT/reports/formal_readiness.json"; fi
