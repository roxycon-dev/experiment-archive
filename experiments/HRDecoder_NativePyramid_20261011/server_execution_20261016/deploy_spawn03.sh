set -eu
BASE=/root/autodl-tmp/nativepyramid_sp_v3_20261016
"$BASE/venv/bin/python" - <<'PY'
from pathlib import Path
import hashlib,zipfile,shutil,json,ast
b=Path('/root/autodl-tmp/nativepyramid_sp_v3_20261016')
assert not (b/'controller_v3_started.json').exists()
assert not (b/'preflight_attempt03').exists() and not (b/'run_seed42_launch02').exists()
status=json.loads((b/'job_status.json').read_text())
assert status['status']=='INSTANCE_PRESERVED' and status['formal_training_started'] is False
assert not Path('/proc/1752').exists() and not Path('/proc/1503').exists()
old=b/'control_attempt02_records'
old.mkdir(exist_ok=False)
for name in ('job_status.json','job_result.json','controller.py','controller_v2_started.json'):
    shutil.copy2(b/name,old/name)
shutil.copytree(b/'code',b/'code_attempt02_snapshot')
p=b/'packages/code_spawn03.zip'
assert hashlib.sha256(p.read_bytes()).hexdigest()=='df6a2ab736c7b1c89fd5746a111784589679cdb0a576e065e2cb692cdcf56865'
d=b/'code'
with zipfile.ZipFile(p) as z:
    assert all((d/x.filename).resolve().is_relative_to(d.resolve()) for x in z.infolist())
    z.extractall(d)
shutil.copy2(b/'controller_v3.py',b/'controller.py')
for node in ast.parse((b/'controller.py').read_text()).body:
    if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='SOURCE_SHA' for t in node.targets):
        assert ast.literal_eval(node.value)=='d3032fc8b20f30a058adbcf3e6bc5a3485b83adafe9b7cdce0682a2d55d7f627'
print('V3_DEPLOYED_OLD_FAILURE_PRESERVED')
PY
cd "$BASE/code"
"$BASE/venv/bin/python" prepare_delivery.py --verify preparation_20261016/delivery_manifest.json
cd "$BASE"
"$BASE/venv/bin/python" -m py_compile controller.py
sha256sum controller.py > receipts/controller_v3_sha256.txt
nohup "$BASE/venv/bin/python" -u controller.py > controller_attempt03.log 2>&1 < /dev/null &
echo "controller_v3_pid=$!"
sleep 2
cat controller_v3_started.json
cat job_status.json
