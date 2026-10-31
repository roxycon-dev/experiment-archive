set -eu
BASE=/root/autodl-tmp/nativepyramid_sp_v3_20261016
/root/miniconda3/envs/hrdecoder_q_20261003/bin/python - <<'PY'
from pathlib import Path
import zipfile,hashlib
base=Path('/root/autodl-tmp/nativepyramid_sp_v3_20261016')
package=base/'packages/code.zip'
assert hashlib.sha256(package.read_bytes()).hexdigest()=='85108889079c030e763c345f9aa2929ef74392d8e191c516926073b0b54e8eaa'
dest=base/'code'
with zipfile.ZipFile(package) as z:
    assert all((dest/x.filename).resolve().is_relative_to(dest.resolve()) for x in z.infolist())
    z.extractall(dest)
for name in ('packages/data.tar','assets/hrnetv2_w48-d2186c55.pth'):
    p=base/name
    print(name,p.stat().st_size if p.exists() else 'not present')
PY
cd "$BASE/code"
/root/miniconda3/envs/hrdecoder_q_20261003/bin/python prepare_delivery.py --verify preparation_20261016/delivery_manifest.json
printf '\n--- Shutdown wrapper ---\n'
ls -l /usr/bin/shutdown
head -n 1 /usr/bin/shutdown
