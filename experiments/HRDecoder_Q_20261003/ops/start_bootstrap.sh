set -eu
ROOT=/root/autodl-tmp/HRDecoder_Q_20261003
mkdir -p "$ROOT/ops" "$ROOT/reports" "$ROOT/assets"
nohup /bin/bash "$ROOT/ops/bootstrap_server.sh" > "$ROOT/reports/bootstrap.log" 2>&1 < /dev/null &
echo "$!" > "$ROOT/reports/bootstrap.pid"
echo BOOTSTRAP_STARTED
