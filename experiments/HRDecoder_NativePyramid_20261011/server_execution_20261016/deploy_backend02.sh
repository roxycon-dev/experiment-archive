set -eu
BASE=/root/autodl-tmp/nativepyramid_sp_v3_20261016
"$BASE/venv/bin/python" - <<'PY'
from pathlib import Path
import hashlib,zipfile,shutil,json,ast
b=Path('/root/autodl-tmp/nativepyramid_sp_v3_20261016')
assert not (b/'controller_v2_started.json').exists(), 'A v2 controller was already started'
assert not (b/'preflight_attempt02').exists() and not (b/'run_seed42').exists()
snapshot=b/'code_attempt01_snapshot'
if not snapshot.exists():
    shutil.copytree(b/'code',snapshot)
old=b/'receipts/controller_v1.py'
if not old.exists():shutil.copy2(b/'controller.py',old)
p=b/'packages/code_backend02.zip'
assert hashlib.sha256(p.read_bytes()).hexdigest()=='a3293437c21a87fe7680183ec55f6cb11322215fb5d0a425c69ea5f73e11131d'
d=b/'code'
with zipfile.ZipFile(p) as z:
    assert all((d/x.filename).resolve().is_relative_to(d.resolve()) for x in z.infolist())
    z.extractall(d)
new=b/'controller_v2.py'
shutil.copy2(new,b/'controller.py')
for node in ast.parse(new.read_text()).body:
    if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='SOURCE_SHA' for t in node.targets):
        assert ast.literal_eval(node.value)=='ddb4cafb36503788f5dac8b24f1320b1bbe28aa8e014187f16d8aa847a3c6c3c'
print('V2_DEPLOYED_NO_PROCESS_STARTED')
PY
cd "$BASE/code"
"$BASE/venv/bin/python" prepare_delivery.py --verify preparation_20261016/delivery_manifest.json
