set -eu
BASE=/root/autodl-tmp/nativepyramid_sp_v3_20261016
test ! -e "$BASE/controller_v2_started.json"
test ! -e "$BASE/preflight_attempt02"
test ! -e "$BASE/run_seed42"
cd "$BASE"
"$BASE/venv/bin/python" -m py_compile controller.py
sha256sum controller.py > receipts/controller_v2_sha256.txt
nohup "$BASE/venv/bin/python" -u controller.py > controller_attempt02.log 2>&1 < /dev/null &
echo "controller_v2_pid=$!"
sleep 2
cat controller_v2_started.json
cat job_status.json
