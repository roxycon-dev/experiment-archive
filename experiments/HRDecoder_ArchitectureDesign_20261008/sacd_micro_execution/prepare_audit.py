"""Adapt the original strict audit to the new candidate and conditional deadline evidence."""
from pathlib import Path
import py_compile
ROOT=Path(__file__).resolve().parent.parent
old=ROOT/'sacd_execution/audit_final_outcome.py'
new=ROOT/'sacd_isolated_execution/audit_final_outcome.py'
text=old.read_text(encoding='utf-8')
text=text.replace("server_receipts/sacd'", "server_receipts/sacd_isolated'")
text=text.replace('training_runs/sacd/', 'training_runs/sacd_isolated/')
text=text.replace('reports/sacd/', 'reports/sacd_isolated/')
text=text.replace('sacd_execution/', 'sacd_isolated_execution/')
start=text.index("registry=read(")
end=text.index("complete=read(",start)
text=text[:start]+'''registry=read('reports/sacd_isolated/SCREEN_STARTED.json')
screen=read('training_runs/sacd_isolated/SCREEN_PASS.json')
extension=read('reports/sacd_isolated/DEADLINE_EXTENSION_ACTIVATED.json')
check('one fresh candidate registered and screen passed before full40k continuation',
      None if not registry or not screen or not extension else
      registry['candidate_count']==registry['candidate_limit']==1 and registry['identity_SHA']==digest
      and not registry['warm_diagnostic_checkpoint_used'] and not registry['full40k_authorized_now']
      and screen['status']=='PASS' and screen['identity_SHA']==digest and screen['screen_updates']==8000
      and screen['full_val_counts_verified'] and all(screen['decisions'].values())
      and screen['time_unix']<1791568800 and extension['full40k_authorized']
      and extension['screen_identity_SHA']==digest and extension['same_fresh_run_continues'],
      'SCREEN_STARTED + SCREEN_PASS + DEADLINE_EXTENSION_ACTIVATED; original02:00 cutoff superseded conditionally')
check('reviewed fresh screening gate source matches training identity',None if not protocol else
      identity['screen_gate_code_SHA']==hashlib.sha256((HERE/'screen_gate.py').read_bytes()).hexdigest()
      and identity['screen_protocol_SHA']==hashlib.sha256((HERE/'screen_protocol.json').read_bytes()).hexdigest(),
      'identity.screen_gate_code_SHA + identity.screen_protocol_SHA')
isolation=read('sacd_isolated_execution/FEATURE_ISOLATION_CPU_PASS.json')
check('pinned CPU feature-path cut and unchanged forward/direct-base gradients',None if not isolation else
      isolation['status']=='PASS' and isolation['torch']=='2.0.1+cu117'
      and isolation['private_feature_gradient_cut'] and isolation['forward_and_loss_bitwise_equal']
      and isolation['direct_z0_gradient_preserved_bitwise'],
      'FEATURE_ISOLATION_CPU_PASS.json; synthetic proof is not an accuracy claim')

'''+text[end:]
needle="check('final test completed before user deadline',None if not test else\n      test['completed_unix']<1791568800,'test.completed_unix < Beijing2026-10-10 02:00')"
assert needle in text
text=text.replace(needle,"check('test follows the user-authorized conditional extension',None if not test or not extension else\n      extension['status']=='ACTIVE' and extension['screen_pass_unix']<1791568800\n      and test['completed_unix']>extension['screen_pass_unix'],\n      'User: through screening, server can be extended until completion; no obsolete02:00 final-test gate')")
# The shared original val reference was independently counted before selection.
text=text.replace("reference_path=HERE/'val_label_reference.json'", "reference_path=ROOT/'sacd_execution/val_label_reference.json'")
new.write_text(text,encoding='utf-8')
py_compile.compile(str(new),doraise=True)
print('CONDITIONAL_EXTENSION_AUDIT_PREPARED')
