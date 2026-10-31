set -eu
BASE=/root/autodl-tmp/nativepyramid_sp_v3_20261016
"$BASE/venv/bin/python" - <<'PY'
from pathlib import Path
import hashlib,json,zipfile,subprocess
b=Path('/root/autodl-tmp/nativepyramid_sp_v3_20261016')
s=json.loads((b/'job_status.json').read_text())
assert s['status']=='FORMAL_TRAINING' and s['source_sha256']=='d3032fc8b20f30a058adbcf3e6bc5a3485b83adafe9b7cdce0682a2d55d7f627'
free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
assert free>16000, 'Insufficient remaining memory for isolated diagnostic without risking active run'
p=b/'packages/code_performance04.zip'
assert hashlib.sha256(p.read_bytes()).hexdigest()=='423733d671e091de19912ab89c880c2e5ec2786a384f80e414f1db75ab66847d'
d=b/'code_performance04_candidate';d.mkdir(exist_ok=False)
assert not (b/'preflight_performance04').exists()
with zipfile.ZipFile(p) as z:
    assert all((d/x.filename).resolve().is_relative_to(d.resolve()) for x in z.infolist())
    z.extractall(d)
print('ISOLATED_CANDIDATE_ONLY; ACTIVE_SOURCE_AND_PROCESS_UNCHANGED')
PY
cd "$BASE/code_performance04_candidate"
"$BASE/venv/bin/python" prepare_delivery.py --verify preparation_20261016/delivery_manifest.json
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 nohup "$BASE/venv/bin/python" -u verify.py --data-root "$BASE/data" --identity "$BASE/code_performance04_candidate/preparation_20261016/data_identity.json" --pretrained "$BASE/assets/hrnetv2_w48-d2186c55.pth" --output "$BASE/preflight_performance04" --workers 8 > "$BASE/preflight_performance04.log" 2>&1 < /dev/null &
echo "isolated_performance_preflight_pid=$!"
