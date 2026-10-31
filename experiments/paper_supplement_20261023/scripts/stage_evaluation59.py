"""Server-to-server, hash-verified staging in a new isolated inference workspace."""
from pathlib import Path
import hashlib
import json
import shlex
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1];EXP=ROOT.parent
LAN='/home/studentuser001/paper_supplement_20261023'
DEST=LAN+'/evaluation_workspace'
SSH=['ssh','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-i','/root/.ssh/codex_supp_to59_20261023','-p','40560','studentuser001@127.0.0.1']
SCP=['scp','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-i','/root/.ssh/codex_supp_to59_20261023','-P','40560']

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()

def remote(code):
    return subprocess.check_output(SSH+['/usr/bin/python3 -c '+shlex.quote(code)],text=True)

records=[]
files=[(EXP/'remote_seed.zip','assets/remote_seed.zip'),(EXP/'A_exact.zip','assets/A_exact.zip')]
for name in ['U0_inference.pth','S20_ema_inference.pth','C20_ema_inference.pth']:
    files.append((EXP/'HRDecoder_ArchitectureDesign_20261008/cache_execution/inference_weights'/name,'assets/'+name))
files.append((EXP/'HRDecoder_Ready_20261002/best_pipeline/weights/H4_best.pth','assets/H4_best.pth'))
for name in ['bootstrap_evaluation59.sh','cache_common_system59.py','evaluation59_common.py','launch_common_cache59.py','exact_weight_transport.py']:
    files.append((ROOT/'scripts'/name,'paper_supplement_20261023/scripts/'+name))
files.append((ROOT/'plan/paired_evaluation_runtime.md','paper_supplement_20261023/plan/paired_evaluation_runtime.md'))
for src,rel in files:
    expected=sha(src);dst=DEST+'/'+rel
    code="from pathlib import Path; import hashlib,json; p=Path("+repr(dst)+"); p.parent.mkdir(parents=True,exist_ok=True); print(json.dumps({'exists':p.exists(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None}))"
    existing=json.loads(remote(code))
    if existing['exists']:
        assert existing['sha256']==expected,('Existing staging file differs',dst)
    else:
        subprocess.run(SCP+[str(src),'studentuser001@127.0.0.1:'+dst+'.part'],check=True)
        code="from pathlib import Path;import hashlib; p=Path("+repr(dst+'.part')+");assert hashlib.sha256(p.read_bytes()).hexdigest()=="+repr(expected)+";p.replace("+repr(dst)+")"
        remote(code)
    records.append({'source':str(src),'target':dst,'bytes':src.stat().st_size,'sha256':expected})
    print(json.dumps({'staged':rel,'bytes':src.stat().st_size}),flush=True)

code=r'''
from pathlib import Path
import hashlib,json,os,subprocess,zipfile
exp=Path(DEST_VALUE);root=exp/'paper_supplement_20261023';task=exp.parent
archive=exp/'assets/remote_seed.zip'
marker=root/'evidence/source_bundle_staged59.json'
if not marker.exists():
 with zipfile.ZipFile(archive) as z:
  for m in z.infolist():
   (exp/m.filename).resolve().relative_to(exp.resolve())
   assert not m.filename.startswith('/')
  assert z.testzip() is None
  # Keep freshly staged evaluation files separate from the older archived scripts.
  for m in z.infolist():
   p=exp/m.filename
   if not p.exists():z.extract(m,exp)
 marker.write_text(json.dumps({'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'CRC_all_entries':'PASS','isolated_new_workspace':str(exp)},indent=2))
ddr=exp/'data/DDR_split_613'
if not ddr.exists():ddr.symlink_to(task/'data/DDR_split_613',target_is_directory=True)
weights=exp/'HRDecoder_ArchitectureDesign_20261008/cache_execution/inference_weights';weights.mkdir(parents=True,exist_ok=True)
for name in ['U0_inference.pth','S20_ema_inference.pth','C20_ema_inference.pth']:
 p=weights/name
 if not p.exists():p.symlink_to(exp/'assets'/name)
legacy=task/'legacy_env/bin/python'
target=exp/'HRDecoder_Ready_20261002/best_pipeline/weights';target.mkdir(parents=True,exist_ok=True)
if not (target/'exact_transport_receipt.json').exists():
 subprocess.run([str(legacy),str(root/'scripts/exact_weight_transport.py'),'decode',str(exp/'assets/A_exact.zip'),'--target',str(target)],check=True)
h4=target/'H4_best.pth'
if not h4.exists():h4.symlink_to(exp/'assets/H4_best.pth')
envlog=(task/'logs/evaluation_environment_install.log').open('ab',buffering=0)
if not (task/'logs/evaluation_environment.json').exists():
 p=subprocess.Popen(['/bin/bash',str(root/'scripts/bootstrap_evaluation59.sh')],stdout=envlog,stderr=subprocess.STDOUT,start_new_session=True)
 (task/'logs/evaluation_environment_install_pid.json').write_text(json.dumps({'pid':p.pid}))
cachelog=(task/'logs/common_cache59_supervisor.log').open('ab',buffering=0)
p=subprocess.Popen(['/usr/bin/python3','-u',str(root/'scripts/launch_common_cache59.py')],stdout=cachelog,stderr=subprocess.STDOUT,start_new_session=True)
(task/'logs/common_cache59_supervisor_pid.json').write_text(json.dumps({'pid':p.pid}))
print(json.dumps({'stage':'SOURCE_AND_WEIGHTS_READY','evaluation_env_preparing':True,'workspace':str(exp)}))
'''.replace('DEST_VALUE',repr(DEST))
print(remote(code),flush=True)
(ROOT/'evidence/evaluation59_staging_receipt.json').write_text(json.dumps({'stage':'STAGED','files':records,'workspace':DEST},indent=2))
# Reuse only the already audited FP32 external U cache, while A/U DDR inference is running.
source=ROOT/'results/remote_probability_cache/U'
destination=DEST+'/paper_supplement_20261023/results/common_cache59/idrid/U'
remote('from pathlib import Path; Path('+repr(destination)+').mkdir(parents=True,exist_ok=True)')
for record in sorted(source.glob('IDRiD_*.json')):
    row=json.loads(record.read_text());probability=record.with_suffix('.npy')
    assert sha(probability)==row['probability_sha256']
    dst=destination+'/'+probability.name
    existing=json.loads(remote("from pathlib import Path;import hashlib,json;p=Path("+repr(dst)+");print(json.dumps({'exists':p.exists(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None}))"))
    if existing['exists']:assert existing['sha256']==row['probability_sha256']
    else:
        subprocess.run(SCP+[str(probability),'studentuser001@127.0.0.1:'+dst+'.part'],check=True)
        remote("from pathlib import Path;import hashlib;p=Path("+repr(dst+'.part')+");assert hashlib.sha256(p.read_bytes()).hexdigest()=="+repr(row['probability_sha256'])+";p.replace("+repr(dst)+")")
    row.update(cohort='idrid',branch='U',origin='Reused audited4080 FP32 external U; no new inference',source_manifest=json.loads((source/'manifest.json').read_text()))
    remote("from pathlib import Path;p=Path("+repr(destination+'/'+record.name)+");p.write_text("+repr(json.dumps(row,indent=2))+")")
    print(json.dumps({'external_U_staged':record.stem}),flush=True)
remote("from pathlib import Path;import json;p=Path("+repr(destination+'/completion.json')+");p.write_text(json.dumps({'status':'COMPLETE','images':27,'origin':'Audited4080 FP32 cache, exact SHA copy'}))")
print('EVALUATION59_ALL_SOURCE_AND_FROZEN_ASSETS_STAGED',flush=True)
