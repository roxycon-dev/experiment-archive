BASE=/root/autodl-tmp/nativepyramid_sp_v3_20261016
date -u
cat "$BASE/job_status.json" 2>/dev/null || true
printf '\n--- Current preflight log ---\n'
tail -n 45 "$BASE/preflight.log" 2>/dev/null || true
printf '\n--- Current training log ---\n'
tail -n 8 "$BASE/training.log" 2>/dev/null || true
printf '\n--- GPU ---\n'
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
