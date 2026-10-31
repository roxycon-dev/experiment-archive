set -eu
BASE=/root/autodl-tmp/nativepyramid_sp_v3_20261016
test -f "$BASE/controller.py"
test -f "$BASE/receipts/assets_ready.json"
test -f "$BASE/receipts/provider_shutdown_preserve_files.sh"
test ! -e "$BASE/bootstrap_started.json"
test ! -e "$BASE/controller_started.json"
cd "$BASE"
/root/miniconda3/envs/hrdecoder_q_20261003/bin/python -m py_compile controller.py bootstrap.py
sha256sum controller.py bootstrap.py > receipts/execution_wrapper_sha256.txt
nohup /root/miniconda3/envs/hrdecoder_q_20261003/bin/python -u bootstrap.py > controller.log 2>&1 < /dev/null &
echo "queued_bootstrap_pid=$!"
sleep 1
cat bootstrap_started.json
