set -eu
BASE=/root/autodl-tmp/nativepyramid_sp_v3_20261016
cat "$BASE/job_status.json"
tail -n 8 "$BASE/preflight_attempt03.log"
if test -f "$BASE/training_attempt03.log"; then tail -n 5 "$BASE/training_attempt03.log"; fi
nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw,temperature.gpu --format=csv
