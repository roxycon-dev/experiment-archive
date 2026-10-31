set -eu
BASE=/root/autodl-tmp/nativepyramid_sp_v3_20261016
cat "$BASE/run_seed42/progress.json"
ps -eo pid,ppid,etimes,time,stat,nlwp,wchan:24,comm,args | grep -E '1752|pt_data_worker|controller.py|benchmark_input.py' | grep -v grep
for TASK in /proc/1752/task/*; do printf '%s ' "$TASK"; cat "$TASK/wchan"; printf '\n'; done
tail -n 15 "$BASE/training_attempt02.log"
