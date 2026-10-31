import json,time,sys
from pathlib import Path
r=Path('/root/autodl-tmp/HRDecoder_FinalDecision_20261005');sys.path.insert(0,str(r))
from qlib.common import read_json,write_json,sha,utc_now,assert_source
old='d97c2c25f965dfa8b464e7876103eb125b67521382e952e0b974b59e07f26606';new='169c48fb058b0c8aeeb88c8b914243cb8adf540dd29f568ca73158661ee00431'
archive=r/'reports/preparation_repairs/endpoint_policy_20261005'
for _ in range(14400):
    gate=read_json(r/'reports/gpu_readiness.json')
    if gate['status']=='READY':break
    if read_json(r/'reports/driver_state.json')['stage']=='TECHNICAL_PAUSE':raise RuntimeError('GPU acceptance paused; do not advance receipt')
    time.sleep(.5)
else:raise TimeoutError('GPU readiness did not complete')
assert not list((r/'runs').glob('*/identity.json')),'Formal identity already exists; require explicit diagnosis'
assert assert_source(r)['sha256']==new
assert gate['source_sha256']==old and set(gate['benchmarks'])=={'C','S','D','SD'}
assert all(row['complete_updates']==100 for row in gate['benchmarks'].values())
early=read_json(r/'reports/early_gpu_gates.json');assert early['status']=='PASS' and early['source_sha256']==old
write_json(archive/'gpu_readiness.original.json',gate);write_json(archive/'early_gpu_gates.original.json',early)
extension={'status':'PASS','only_changed_file':'decision.py','training_and_GPU_gate_files_unchanged':True,
    'original_validated_source_sha256':old,'source_sha256':new,'time':utc_now(),
    'reason':'Protocol section8 allows both completed paired20k endpoints after a later scientific stop; no missing32k is fabricated.'}
for name,receipt in [('early_gpu_gates.json',early),('gpu_readiness.json',gate)]:
    receipt.update(source_sha256=new,non_training_source_extension=extension)
    write_json(r/'reports'/name,receipt)
write_json(r/'reports/endpoint_policy_revision.json',extension)
print('ENDPOINT_POLICY_SOURCE_EXTENSION_PASS',flush=True)
