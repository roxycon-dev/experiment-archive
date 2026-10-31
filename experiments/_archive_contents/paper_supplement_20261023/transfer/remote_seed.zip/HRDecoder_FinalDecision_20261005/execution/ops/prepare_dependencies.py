"""Isolated exact Torch dependency; preserve legacy P0 environment."""
import sys,json,shlex
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from ops.remote import connect,command
from qlib.common import write_json,utc_now
code="""
from pathlib import Path
import subprocess,os,json
prefix=Path('/root/miniconda3/envs/hrdecoder_q_20261003')
log=Path('/root/autodl-tmp/hrdecoder_final_dependency_install.log')
assert not Path('/root/autodl-tmp/hrdecoder_final_dependency_install.pid').exists(),'Existing installer: inspect before duplicate'
worker='''import subprocess
subprocess.run(['/root/miniconda3/bin/conda','create','-y','-p','/root/miniconda3/envs/hrdecoder_q_20261003','python=3.11','pip'],check=True)
subprocess.run(['/root/miniconda3/envs/hrdecoder_q_20261003/bin/python','-m','pip','install','--no-cache-dir','torch==2.7.1','torchvision==0.22.1','--index-url','https://download.pytorch.org/whl/cu128'],check=True)
subprocess.run(['/root/miniconda3/envs/hrdecoder_q_20261003/bin/python','-m','pip','install','--no-cache-dir','numpy==1.26.4','Pillow','opencv-python-headless==4.11.0.86','timm==1.0.20','ninja'],check=True)
'''
with log.open('ab') as out:
 p=subprocess.Popen(['/root/miniconda3/bin/python','-u','-c',worker],stdin=subprocess.DEVNULL,stdout=out,stderr=out,start_new_session=True)
Path('/root/autodl-tmp/hrdecoder_final_dependency_install.pid').write_text(str(p.pid))
print(json.dumps({'pid':p.pid,'log':str(log),'new_env':str(prefix),'base_legacy_unchanged':True}))
"""
with connect() as client:result=json.loads(command(client,"/root/miniconda3/bin/python - <<'PY'\n"+code+"\nPY",30))
result['time']=utc_now();write_json(ROOT/'reports/dependency_install.json',result);print(result)
