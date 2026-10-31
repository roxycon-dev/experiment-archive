"""Current v3 CPU check entry point; separate processes isolate tensor defaults.

No real images, checkpoints, GPU jobs or optimizer steps are used.
Historical v1/v2 check programs remain in archive/v1 and archive/v2.
"""
import json
from pathlib import Path
import subprocess
import sys

root=Path(__file__).resolve().parent
for name in ('verify_v3_objectives.py','verify_v3_modules.py'):
    subprocess.run([sys.executable,str(root/name)],cwd=root,check=True)
spec=json.loads((root/'02_架构规格.json').read_text(encoding='utf-8'))
module=json.loads((root/'evidence/v3_module_checks.json').read_text(encoding='utf-8'))
objective=json.loads((root/'evidence/v3_objective_checks.json').read_text(encoding='utf-8'))
assert spec['version']==objective['version']==3
assert module['status']=='PASS' and objective['failed']==0
assert spec['new_parameters_total']==module['parameters']['sum_new_parameters']
assert spec['new_forward_MAC_2048_batch1']['total']==module['mac_2048_analytic']['all_new_forward_mac']
assert spec['training']['global_batches']==spec['training']['optimizer_updates']==40000
print('V3 specification / objective checks / module checks consistent; no accuracy claim.')
