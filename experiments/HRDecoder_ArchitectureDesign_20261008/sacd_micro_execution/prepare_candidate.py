"""Create a sealed one-change candidate locally; never edit the running SACD core."""
import json
from pathlib import Path
import py_compile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OLD = ROOT / 'sacd_execution'
NEW = ROOT / 'sacd_isolated_execution'
NEW.mkdir(exist_ok=True)
assert not (NEW / 'SEALED_LOCAL_SOURCE.json').exists(), 'Review an existing candidate rather than replacing it'
names = ('baseline.py', 'model.py', 'private_modules.py', 'stable_objectives.py',
         'fast_objectives.py', 'objectives.py', 'data.py', 'loader.py',
         'verify_sacd.py', 'verify_checkpoint.py')
for name in names:
    (NEW / name).write_bytes((OLD / name).read_bytes())
source = (NEW / 'model.py').read_text(encoding='utf-8')
needle = 'actions = self.private(image, features, z0)'
assert source.count(needle) == 1
source = source.replace(needle, 'private_features = tuple(feature.detach() for feature in features)\n        actions = self.private(image, private_features, z0)')
(NEW / 'model.py').write_text(source, encoding='utf-8')
spec = json.loads((OLD / 'architecture_spec.json').read_text())
spec.update(design_version=7, name='HRDecoder-SACD-Isolated',
            private_feature_gradient='all four shared HRNet features detached at L/G input; direct base path unchanged',
            budget='one fresh8k screen; continue same run to40k only if registered screen passes; then deadline extended until completion',
            causal_effect_proven=False)
(NEW / 'architecture_spec.json').write_text(json.dumps(spec, indent=2) + '\n')
(NEW / 'screen_protocol.json').write_bytes((HERE / 'micro_protocol.json').read_bytes())
source = (OLD / 'train.py').read_text(encoding='utf-8')
source = source.replace("ROOT/'training_runs/sacd'", "ROOT/'training_runs/sacd_isolated'")
source = source.replace("ROOT/'reports/sacd'", "ROOT/'reports/sacd_isolated'")
source = source.replace("'design_version':6,'name':'HRDecoder-SACD'", "'design_version':7,'name':'HRDecoder-SACD-Isolated'")
source = source.replace("'gradient_route':'task all; action only L/G/head'", "'gradient_route':'task all with private feature inputs detached; direct base retained; action only L/G/head'")
source = source.replace("'spec_SHA':file_sha(HERE/'architecture_spec.json'),", "'spec_SHA':file_sha(HERE/'architecture_spec.json'),'screen_protocol_SHA':file_sha(HERE/'screen_protocol.json'),")
needle = '    assert completed==40000\n'
assert source.count(needle) == 1
source = source.replace(needle, '    if completed == 8000 and (WORK/\'SCREEN_REJECTED.json\').exists():\n        return\n' + needle)
needle = "                'best_val_mIoU':best if np.isfinite(best) else None,'last_validation':validation,'time_unix':time.time()})\n"
assert source.count(needle) == 1
gate = '''
            if completed == 8000:
                from screen_gate import screen_gate
                decision = screen_gate(WORK, HERE/'screen_protocol.json', digest)
                write_json(WORK/('SCREEN_PASS.json' if decision['status']=='PASS' else 'SCREEN_REJECTED.json'), decision)
                print(json.dumps({'fresh_screen': decision}), flush=True)
                if decision['status'] != 'PASS':
                    break
                write_json(REPORTS/'DEADLINE_EXTENSION_ACTIVATED.json', {
                    'status':'ACTIVE','user_authorization':'Through screening, server may remain available until completion',
                    'screen_identity_SHA':digest,'screen_pass_unix':decision['time_unix'],
                    'original_deadline_unix':1791568800,'full40k_authorized':True,
                    'same_fresh_run_continues':True,'test_only_after_locked40k':True})
'''
source = source.replace(needle, needle + gate)
(NEW / 'train.py').write_text(source, encoding='utf-8')
source = (OLD / 'test_eval.py').read_text(encoding='utf-8')
source = source.replace("'training_runs/sacd/last.pt'", "'training_runs/sacd_isolated/last.pt'")
# The original wall-clock cutoff is revoked only by the recorded screen pass;
# the new controller supervises its own training/test safety timeout.
start = source.index("    deadline=datetime(2026,10,10,2,0")
source = source[:start] + "    extension=json.loads((REPORTS/'DEADLINE_EXTENSION_ACTIVATED.json').read_text())\n    assert extension['full40k_authorized'] and extension['status']=='ACTIVE'\n    main()\n"
(NEW / 'test_eval.py').write_text(source, encoding='utf-8')
proof = (OLD / 'diagnosis_20261009/verify_feature_isolation.py').read_text(encoding='utf-8')
proof = proof.replace('CORE=HERE.parent', 'CORE=HERE')
# This mathematical two-arm proof must use the shared private module. The
# full candidate's Student forward is independently smoke-tested before launch.
(NEW / 'verify_feature_isolation.py').write_text(proof, encoding='utf-8')
(NEW / 'feature_isolation.patch').write_bytes((OLD / 'diagnosis_20261009/feature_isolation.patch').read_bytes())
for path in NEW.glob('*.py'):
    py_compile.compile(str(path), doraise=True)
print(json.dumps({'status':'CANDIDATE_CREATED_LOCALLY','directory':str(NEW),'GPU_used':False}))
