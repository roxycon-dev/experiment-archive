from pathlib import Path
import hashlib
import json
import zipfile

root = Path(__file__).resolve().parents[1]
files = sorted((root/'assets/linux_wheels').glob('*.whl'))
assert len(files) == 4
manifest = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
with zipfile.ZipFile(root/'assets/linux_wheels.zip', 'w', zipfile.ZIP_STORED) as z:
    for p in files:
        z.write(p, p.name)
    z.writestr('sha256.json', json.dumps(manifest))
print('Pinned Linux wheels bundled; 4 files')
