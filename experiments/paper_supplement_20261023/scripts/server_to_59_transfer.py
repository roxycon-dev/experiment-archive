"""Copy existing legacy MMCV binary and immutable DDR tar directly between authorized servers."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import zipfile
import mmcv

ROOT=Path(__file__).resolve().parents[1];EXP=ROOT.parent
assets=ROOT/'server_transfer_assets';assets.mkdir(exist_ok=True)
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024**2),b''):h.update(chunk)
    return h.hexdigest()
package=Path(mmcv.__file__).resolve().parent
dist=Path(importlib.metadata.distribution('mmcv-full')._path)
archive=assets/'mmcv_legacy_runtime.zip'
records=[]
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=3) as z:
    for folder in [package,dist]:
        for p in sorted(folder.rglob('*')):
            if p.is_file() and '__pycache__' not in p.parts:
                name=p.relative_to(folder.parent).as_posix()
                z.write(p,name);records.append({'name':name,'sha256':sha(p),'bytes':p.stat().st_size})
receipt={'status':'EXACT_EXISTING_RUNTIME_COPY','mmcv':mmcv.__version__,'archive_sha256':sha(archive),
         'archive_bytes':archive.stat().st_size,'files':records,'target_compatibility':'CPython3.8,torch2.0.1+cu117; must pass actual2080Ti CUDA forward/backward gate'}
(assets/'mmcv_legacy_runtime_receipt.json').write_text(json.dumps(receipt,indent=2))
key='/root/.ssh/codex_supp_to59_20261023'
opts=['-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-i',key]
target='studentuser001@127.0.0.1'
remote='/home/studentuser001/paper_supplement_20261023/assets/'
files=[assets/'mmcv_legacy_runtime_receipt.json',archive,EXP/'ddr_split613.tar',ROOT/'training_v1/hrnetv2_w48-d2186c55.pth']
for p in files:
    while not p.exists():__import__('time').sleep(5)
    if p.name.startswith('hrnetv2_'):
        while p.stat().st_size!=310659148:__import__('time').sleep(5)
    print(json.dumps({'transferring':p.name,'bytes':p.stat().st_size}),flush=True)
    subprocess.run(['scp',*opts,'-P','40559',str(p),target+':'+remote+p.name+'.part'],check=True)
    subprocess.run(['ssh',*opts,'-p','40559',target,'mv -- '+remote+p.name+'.part '+remote+p.name],check=True)
    print(json.dumps({'transferred':p.name,'sha256':sha(p)}),flush=True)
print('ALL_BASELINE_ASSETS_TRANSFERRED',flush=True)
