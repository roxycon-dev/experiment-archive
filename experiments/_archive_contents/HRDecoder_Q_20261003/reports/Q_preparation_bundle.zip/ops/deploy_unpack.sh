set -eu
cd /root/autodl-tmp/HRDecoder_Q_20261003
/root/miniconda3/bin/python - <<'PY'
from pathlib import Path,PurePosixPath
import hashlib,json,zipfile
root=Path.cwd()
archive=root/'reports/Q_preparation_bundle.zip'
expected=json.loads((root/'reports/package_identity.json').read_text())['sha256']
if hashlib.sha256(archive.read_bytes()).hexdigest()!=expected:raise ValueError('Source bundle SHA differs')
with zipfile.ZipFile(archive) as z:
 for name in z.namelist():
  p=PurePosixPath(name)
  if p.is_absolute() or '..' in p.parts:raise ValueError('Unsafe archive target')
 z.extractall(root)
print('SOURCE_BUNDLE_SHA_VERIFIED_AND_EXTRACTED')
PY
