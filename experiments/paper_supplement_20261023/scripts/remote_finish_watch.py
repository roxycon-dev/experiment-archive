"""Start fusion as soon as verified inputs arrive, then package only small evidence outputs."""
import json
from pathlib import Path
import subprocess
import time
import zipfile

ROOT=Path(__file__).resolve().parents[1];EXP=ROOT.parent
cache=ROOT/'results/remote_probability_cache'
weights=EXP/'HRDecoder_Ready_20261002/best_pipeline/weights'
started=time.monotonic()
status=ROOT/'remote_queue_status.json'
while True:
    ready=all((cache/a/'completion.json').exists() for a in ['S','C','U']) and (weights/'exact_transport_receipt.json').exists() and (weights/'H4_best.pth').exists()
    if ready:break
    status.write_text(json.dumps({'stage':'WAITING_FOR_REQUIRED_WEIGHTS_AND_BRANCHES','seconds':time.monotonic()-started}),encoding='utf-8')
    if time.monotonic()-started>3600:raise TimeoutError('Required inputs did not arrive within one hour')
    time.sleep(5)
status.write_text(json.dumps({'stage':'FUSING_SYSTEM'}),encoding='utf-8')
with (ROOT/'log_fusion.txt').open('w') as log:
    process=subprocess.Popen(['/root/miniconda3/bin/python','-u',str(ROOT/'scripts/remote_fuse_system.py')],cwd=str(EXP),stdout=log,stderr=subprocess.STDOUT)
    code=process.wait()
if code:
    status.write_text(json.dumps({'stage':'FUSION_FAILED','code':code}),encoding='utf-8');raise RuntimeError('Fusion failed; inspect log_fusion.txt')
audit={'branches':{a:{'manifest':json.loads((cache/a/'manifest.json').read_text()),
                     'completion':json.loads((cache/a/'completion.json').read_text()),
                     'validation_canary':json.loads((cache/a/'canary_007-1774-100.json').read_text())} for a in ['S','C','U']},
       'GPU_query':subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,driver_version','--format=csv,noheader'],text=True),
       'probability_caches_retained_on_server':True,'new_training':False}
(ROOT/'evidence/remote_branch_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
archive=EXP/'external_results.zip'
with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=3) as z:
    for p in (ROOT/'results/idrid_holdout_v1').rglob('*'):
        if p.is_file():z.write(p,p.relative_to(EXP).as_posix())
    for p in list(ROOT.glob('log_*.txt'))+[ROOT/'evidence/remote_branch_audit.json',weights/'exact_transport_receipt.json']:
        if p.is_file():z.write(p,p.relative_to(EXP).as_posix())
status.write_text(json.dumps({'stage':'COMPLETE_READY_TO_DOWNLOAD','archive':str(archive),'bytes':archive.stat().st_size}),encoding='utf-8')
