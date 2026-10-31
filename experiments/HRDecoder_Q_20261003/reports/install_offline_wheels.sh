set -eu
ROOT=/root/autodl-tmp/HRDecoder_Q_20261003
ENV=/root/miniconda3/envs/hrdecoder_q_20261003
if grep -q ISOLATED_ENVIRONMENT_READY "$ROOT/reports/bootstrap.log"; then
  echo ENVIRONMENT_ALREADY_READY
  exit 0
fi
"$ENV/bin/python" - <<'PY'
import hashlib, json, zipfile
from pathlib import Path
root = Path('/root/autodl-tmp/HRDecoder_Q_20261003/assets')
with zipfile.ZipFile(root/'linux_wheels.zip') as z:
    identity = json.loads(z.read('sha256.json'))
    out = root/'linux_wheels'
    out.mkdir(exist_ok=True)
    for name, expected in identity.items():
        assert Path(name).name == name and name.endswith('.whl')
        blob = z.read(name)
        assert hashlib.sha256(blob).hexdigest() == expected
        (out/name).write_bytes(blob)
print('All 4 wheel hashes verified')
PY
BOOTSTRAP_PID=$(cat "$ROOT/reports/bootstrap.pid")
pkill -TERM -P "$BOOTSTRAP_PID" 2>/dev/null || true
"$ENV/bin/python" -m pip install --no-index --find-links="$ROOT/assets/linux_wheels" numpy==2.2.6 pillow==11.2.1 opencv-python-headless==4.11.0.86 termcolor==3.1.0 >> "$ROOT/reports/bootstrap.log" 2>&1
"$ENV/bin/python" -m pip freeze > "$ROOT/reports/environment_freeze.txt"
printf 'ISOLATED_ENVIRONMENT_READY\n' >> "$ROOT/reports/bootstrap.log"
echo OFFLINE_PINNED_ENVIRONMENT_READY
