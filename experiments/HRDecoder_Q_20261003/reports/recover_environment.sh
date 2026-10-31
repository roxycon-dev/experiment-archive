set -eu
ROOT=/root/autodl-tmp/HRDecoder_Q_20261003
ENV=/root/miniconda3/envs/hrdecoder_q_20261003
# Stop only this unfinished bootstrap's pip child, never any experiment.
BOOTSTRAP_PID=$(cat "$ROOT/reports/bootstrap.pid")
pkill -TERM -P "$BOOTSTRAP_PID" 2>/dev/null || true
nohup /bin/bash -c 'set -eu; ENV=/root/miniconda3/envs/hrdecoder_q_20261003; ROOT=/root/autodl-tmp/HRDecoder_Q_20261003; "$ENV/bin/python" -m pip install --no-cache-dir --index-url https://pypi.org/simple numpy==2.2.6 pillow==11.2.1 opencv-python-headless==4.11.0.86 termcolor==3.1.0; "$ENV/bin/python" -m pip freeze > "$ROOT/reports/environment_freeze.txt"; printf "ISOLATED_ENVIRONMENT_READY\n"' >> "$ROOT/reports/bootstrap.log" 2>&1 < /dev/null &
echo "$!" > "$ROOT/reports/bootstrap.pid"
echo DEPENDENCY_DOWNLOAD_RESUMED_FROM_OFFICIAL_PYPI_WITH_SAME_PINS
