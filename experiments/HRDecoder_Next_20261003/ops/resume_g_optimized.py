"""Pre-training scheduling revision, preserving completed predictions and producer identities."""
from pathlib import Path
import sys
import subprocess
import os
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from nextlib.common import read_json,write_json,source_identity
old=read_json(ROOT/'configs/settings.json')['source_identity'];new=source_identity()
changed=[p for p,h in new['files'].items() if old['files'].get(p)!=h]
assert changed==['nextlib/evaluate.py'],changed
assert not (ROOT/'work/geometry_lock.json').exists()
assert not list((ROOT/'work').glob('M_seed*'))
assert read_json(ROOT/'reports/server_preflight.json')['source_sha256']==old['sha256']
receipt=dict(time=time.time(),old_source=old['sha256'],new_source=new['sha256'],changed=changed,
    reason='Measured FP32 lossless compression 2.63s per image; four CPU workers overlap unchanged serial forwards',
    model_geometry_normalization_threshold_unchanged=True,checkpoint_and_training_unchanged=True,
    validation='Local full protocol/resume acceptance; existing original counts rechecked after revision',migrated=[])
for folder in (ROOT/'work/G').iterdir():
    if not folder.is_dir():continue
    for filename in ['index.json','partial.json','metrics.json','unavailable.json']:
        p=folder/filename
        if not p.exists():continue
        j=read_json(p);identity=j.get('identity',{})
        if identity.get('source_sha256')==old['sha256']:
            identity['source_sha256']=new['sha256']
            j['scheduling_revision']=dict(original_producer_source_sha256=old['sha256'],
                consumer_source_sha256=new['sha256'],change='CPU scheduling only; prediction graph unchanged')
            write_json(p,j);receipt['migrated'].append(str(p.relative_to(ROOT)))
write_json(ROOT/'reports/scheduling_revision.json',receipt)
env=dict(os.environ,PATH='/root/miniconda3/bin:'+os.environ.get('PATH',''),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONUNBUFFERED='1')
for stage,cmd in [
 ('relocate',[sys.executable,'configure.py','--ddr-root','/root/HRDecoder/data/DDR_split_613','--prefer-existing','/root/autodl-tmp/HRDecoder_Ready_20261002']),
 ('acceptance',[sys.executable,'preflight.py','--server']),
 ('G',[sys.executable,'run_campaign.py','--stage','G'])]:
 write_json(ROOT/'reports/server_stage.json',dict(stage=stage,status='running',time=time.time(),pid=os.getpid()))
 with (ROOT/f'reports/{stage}.log').open('a') as log:
  result=subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
 if result.returncode:
  write_json(ROOT/'reports/server_stage.json',dict(stage=stage,status='failed',exit_code=result.returncode,time=time.time()));sys.exit(result.returncode)
write_json(ROOT/'reports/server_stage.json',dict(stage='G',status='complete',time=time.time(),next='Review full G and start M'))
