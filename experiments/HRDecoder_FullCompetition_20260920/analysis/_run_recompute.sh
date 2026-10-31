set -u
R=/root/autodl-tmp/HRDecoder_FullCompetition_20260920
PY=/root/miniconda3/bin/python
mkdir -p /root/logs_run
echo "=== GPU before ==="; nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
$PY $R/analysis/independent_recompute.py 2>&1 | tee /root/logs_run/independent_recompute.log
echo "[recompute_exit=${PIPESTATUS[0]}]"
echo "=== GPU after ==="; nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
ls -la /root/logs_run/independent_recompute.json
