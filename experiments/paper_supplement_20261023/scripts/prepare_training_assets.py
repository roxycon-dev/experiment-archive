"""Lossless parent export and isolated copy of the already-used C/S training implementation."""
import hashlib
import json
from pathlib import Path
import zipfile
import torch

ROOT=Path(__file__).resolve().parents[1]; EXP=ROOT.parent
OUT=ROOT/'transfer'; OUT.mkdir(exist_ok=True)
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
torch.set_num_threads(2)
parent=EXP/'HRDecoder_NextPlan_20261004/execution/server_results/runs/F12_seed42/checkpoint_u12000.pth'
expected='68d7b5821fab1210b3336cf2a63b9de893f24fc32876f5255d7d662084b5e728'
assert sha(parent)==expected
saved=torch.load(parent,map_location='cpu',mmap=True,weights_only=False)
assert saved['completed']==12000
compact=OUT/'F12_parent_model_exact.pth'
if not compact.exists():
    torch.save({'model':saved['model'],'completed':12000,'original_identity':saved['identity'],
                'source_full_sha256':expected,'source_state_key':'model'},compact)
check=torch.load(compact,map_location='cpu',mmap=True,weights_only=False)
assert set(check['model'])==set(saved['model']) and all(torch.equal(check['model'][k],v) for k,v in saved['model'].items())
(ROOT/'evidence/training_parent_export.json').write_text(json.dumps({'status':'PASS','source':str(parent),
    'source_sha256':expected,'compact_sha256':sha(compact),'model_tensors_bitwise_equal':True,
    'state_tensors':len(check['model']),'completed':12000,'uses_raw_model_as_historical_C_S_initialization':True},indent=2))
src=EXP/'HRDecoder_FinalDecision_20261005/execution/server_results'
files={p:Path('paper_supplement_20261023/training_v1')/p.relative_to(src) for p in src.glob('*.py')}
for folder in [src/'qlib',src/'vendor/msda_compat']:
    for p in folder.rglob('*'):
        if p.is_file() and p.suffix in ('.py','.h','.cuh','.cu','.cpp'):
            files[p]=Path('paper_supplement_20261023/training_v1')/p.relative_to(src)
header=src.parent/'vendor/msda_compat/src/cuda/ms_deform_im2col_cuda.cuh'
receipt=json.loads((src/'reports/kernel_source_identity.json').read_text())
assert sha(header)==next(r['compat_sha256'] for r in receipt['files'] if r['file']=='cuda/ms_deform_im2col_cuda.cuh')
files[header]=Path('paper_supplement_20261023/training_v1/vendor/msda_compat/src/cuda/ms_deform_im2col_cuda.cuh')
for name in ['kernel_source_identity.json','data_identity.json','pretraining_identity.json']:
    p=src/'reports'/name; files[p]=Path('paper_supplement_20261023/training_v1/reports')/name
with zipfile.ZipFile(OUT/'training_sources.zip','w',zipfile.ZIP_DEFLATED,compresslevel=3) as z:
    for p,name in sorted(files.items()):z.write(p,name.as_posix())
print(json.dumps({'parent_compact_bytes':compact.stat().st_size,'training_source_files':len(files),
                  'bundle_bytes':(OUT/'training_sources.zip').stat().st_size}),flush=True)
