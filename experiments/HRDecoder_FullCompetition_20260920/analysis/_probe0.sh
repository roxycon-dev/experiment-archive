set -u
R=/root/autodl-tmp/HRDecoder_FullCompetition_20260920
echo "=== tests dir on server ==="; ls -la $R/tests
echo "=== analysis dir on server ==="; ls -la $R/analysis
echo "=== find contract py ==="; find $R -name "expert_mask_contract.py" -o -name "test_packed_mask_contract.py" 2>/dev/null
echo "=== python/pytest ==="; which python; python -c "import pytest;print('pytest',pytest.__version__)" 2>&1 | tail -2
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
