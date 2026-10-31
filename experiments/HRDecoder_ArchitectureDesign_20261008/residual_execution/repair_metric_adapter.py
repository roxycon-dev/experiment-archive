"""Repair scalar metric validation only, retaining and proving the training identity."""
import json
from pathlib import Path
import shlex
import sys

HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
REMOTE='/root/autodl-tmp/HRDecoder_v3_20261008'
sys.path.insert(0,str(ROOT/'local_preparation'))
from remote import connect

client=connect()
try:
    with client.open_sftp() as ftp:
        # Preserve the exact original helper before replacing its API adapter.
        with ftp.open(REMOTE+'/sacd_micro_execution/gate.py','rb') as f:original=f.read()
        (HERE/'gate_before_metric_adapter.py').write_bytes(original)
        with ftp.open(REMOTE+'/reports/residual/gate_before_metric_adapter.py','wb') as f:f.write(original)
        ftp.put(str(ROOT/'sacd_micro_execution/gate.py'),REMOTE+'/sacd_micro_execution/gate.py')
        ftp.put(str(HERE/'verify_metric_adapter.py'),REMOTE+'/residual_execution/verify_metric_adapter.py')
    code='''import copy,hashlib,json,subprocess,sys,tempfile,time
from pathlib import Path
r=Path('/root/autodl-tmp/HRDecoder_v3_20261008');h=r/'residual_execution';w=r/'training_runs/residual';p=r/'reports/residual'
protocol=json.loads((w/'protocol.json').read_text());digest=protocol['identity_SHA'];material=protocol['identity']
assert digest=='96cd1bc0ce79e9e5f5f8247958f868a4e00023f072bc6fea920092ad5a5003ec'
for n,d in material['code'].items():assert hashlib.sha256((h/n).read_bytes()).hexdigest()==d,n
assert hashlib.sha256((h/'screen_gate.py').read_bytes()).hexdigest()==material['screen_gate_code_SHA']
assert hashlib.sha256((h/'screen_protocol.json').read_bytes()).hexdigest()==material['screen_protocol_SHA']
assert hashlib.sha256((h/'architecture_spec.json').read_bytes()).hexdigest()==material['spec_SHA']
subprocess.run([sys.executable,str(h/'verify_metric_adapter.py')],cwd=str(h),check=True)
sys.path.insert(0,str(h));from screen_gate import screen_gate
# Test the entire pure gate with a deliberately non-improving synthetic8k row.
# The fixture never enters training_runs or user-visible validation histories.
with tempfile.TemporaryDirectory(prefix='metric_gate_fixture_',dir=str(p)) as temp:
 t=Path(temp);assert p.resolve() in t.resolve().parents
 for name in ('protocol.json','GPU_SMOKE_PASS.json','val_04000.json'):(t/name).write_bytes((w/name).read_bytes())
 fake=json.loads((w/'val_04000.json').read_text());fake['completed']=8000
 (t/'val_08000.json').write_text(json.dumps(fake))
 decision=screen_gate(t,h/'screen_protocol.json',digest)
 assert decision['status']=='REJECTED' and not decision['decisions']['4k_to8k_gain']
 assert decision['full_val_counts_verified']
cpu=json.loads((h/'METRIC_ADAPTER_CPU_PASS.json').read_text())
validator={'validation_helper_SHA256':cpu['validation_helper_SHA256'],
 'screen_gate_code_SHA256':material['screen_gate_code_SHA'],
 'screen_protocol_SHA256':material['screen_protocol_SHA'],'training_identity_SHA':digest}
receipt={'status':'PASS','repair_kind':'validation-only API adapter, complete76 record vs standalone base row',
 'time_unix':time.time(),'validator_identity':validator,
 'validator_identity_SHA':hashlib.sha256(json.dumps(validator,sort_keys=True).encode()).hexdigest(),
 'original_helper_SHA256':hashlib.sha256((p/'gate_before_metric_adapter.py').read_bytes()).hexdigest(),
 'training_identity_SHA':digest,'all_core_network_forward_loss_backward_sampler_optimizer_sources_unchanged':True,
 'original_GPU_SMOKE_still_applicable':True,'GPU_SMOKE_identity_SHA':json.loads((w/'GPU_SMOKE_PASS.json').read_text())['identity_SHA'],
 'full_screen_gate_CPU_fixture_PASS':True,'fixture_is_not_real8k_validation':True,
 'all_metric_negative_controls_rejected':cpu['negative_controls_rejected'],
 'screen_criteria_changed':False,'raw_counts_or_val_artifacts_changed':False,
 'optimizer_updates_by_repair':0,'server_kept_running':True,'external_diagnostics_controlled':False}
(p/'METRIC_VALIDATOR_REPAIR.json').write_text(json.dumps(receipt,indent=2)+'\\n')
print(json.dumps(receipt))
'''
    _,o,e=client.exec_command('/root/miniconda3/bin/python -c '+shlex.quote(code))
    raw,error=o.read().decode(),e.read().decode();assert o.channel.recv_exit_status()==0,error
    lines=raw.strip().splitlines();receipt=json.loads(lines[-1])
    (HERE/'METRIC_VALIDATOR_REPAIR.json').write_text(json.dumps(receipt,indent=2)+'\n')
    with client.open_sftp() as ftp:
        for rel in ('reports/residual/METRIC_VALIDATOR_REPAIR.json','residual_execution/METRIC_ADAPTER_CPU_PASS.json'):
            target=ROOT/'cache_execution/server_receipts/residual'/rel;target.parent.mkdir(parents=True,exist_ok=True)
            ftp.get(REMOTE+'/'+rel,str(target))
    print(json.dumps(receipt))
finally:client.close()
