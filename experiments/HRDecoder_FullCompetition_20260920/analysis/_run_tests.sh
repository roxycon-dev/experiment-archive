set -u
R=/root/autodl-tmp/HRDecoder_FullCompetition_20260920
PY=/root/miniconda3/bin/python
export PYTHONPATH=$R:$R/analysis
mkdir -p /root/logs_run
echo "=== GPU before ==="; nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
echo "=== RUN REGRESSION TESTS (CPU) ==="
cd $R && $PY tests/test_packed_mask_contract.py 2>&1 | tee /root/logs_run/test_contract.log | tail -60
echo "[tests_exit=${PIPESTATUS[0]}]"
echo "=== GPU after ==="; nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
