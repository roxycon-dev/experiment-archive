set -eu
ROOT=/root/autodl-tmp/HRDecoder_Q_20261003
ENV=/root/miniconda3/envs/hrdecoder_q_20261003
cd "$ROOT"
if [ -f reports/formal.pid ] && kill -0 "$(cat reports/formal.pid)" 2>/dev/null; then
  echo FORMAL_ALREADY_RUNNING
  exit 0
fi
if [ -f reports/review_launcher.pid ] && kill -0 "$(cat reports/review_launcher.pid)" 2>/dev/null; then
  echo REVIEW_LAUNCH_ALREADY_RUNNING
  exit 0
fi
"$ENV/bin/python" - <<'PY'
from pathlib import Path
from qlib.common import assert_source,read_json
assert_source('.')
c=read_json('reports/campaign_state.json')
assert c['review_seed_authorized'] and c['discovery_complete'] and not c['architecture_stopped']
assert c['arms']==['Q_DDR'] and c['common_endpoint']==16000
assert read_json('runs/Q_DDR_seed42/status.json')['stage']=='COMPLETE'
assert not Path('runs/Q_DDR_seed2024/last.pth').exists()
PY
cat > reports/review_launcher.sh <<'LAUNCH'
set -eu
ROOT=/root/autodl-tmp/HRDecoder_Q_20261003
ENV=/root/miniconda3/envs/hrdecoder_q_20261003
cd "$ROOT"
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 "$ENV/bin/python" -u make_stream.py --runtime configs/runtime.server.json --arm Q_DDR --seed 2024 > reports/review_stream_seed2024.log 2>&1
"$ENV/bin/python" - <<'PY'
from qlib.common import read_json,write_json,utc_now
c=read_json('reports/campaign_state.json')
assert c['review_seed_authorized'] and c['common_endpoint']==16000
write_json('reports/preparation_status.json',{'stage':'FORMAL_Q_DDR_SEED2024_STARTING',
 'formal_training_started':True,'arm':'Q_DDR','seed':2024,'endpoint':16000,
 'monitor_minutes':45,'Q_MIX':'NOT_READY_MISSING_OFFICIAL_ORIGINAL_IMAGES',
 'shutdown_authorized':True,'updated_at':utc_now()})
PY
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 nohup "$ENV/bin/python" -u train_run.py --arm Q_DDR --seed 2024 --endpoint 16000 > reports/formal_Q_DDR_seed2024.log 2>&1 < /dev/null &
echo "$!" > reports/formal.pid
echo FORMAL_Q_DDR_SEED2024_LAUNCHED_AT_SHARED_16K_ENDPOINT
LAUNCH
nohup /bin/bash reports/review_launcher.sh > reports/review_launcher.log 2>&1 < /dev/null &
echo "$!" > reports/review_launcher.pid
echo REVIEW_SEED2024_REGISTERED_LAUNCH_QUEUED
