"""Pin the proven Torch stack; then start the assigned node's acceptance."""
import sys,subprocess,json,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from qlib.common import write_json,utc_now
py=sys.executable
pins=ROOT/'reports/parallel_dependency_constraints.txt'
pins.write_text('torch==2.7.1+cu128\ntorchvision==0.22.1+cu128\nnumpy==1.26.4\ntimm==1.0.20\nPyYAML==6.0.3\nhuggingface_hub==2.1.1\nsafetensors==0.8.0\n')
subprocess.run([py,'-m','pip','install','--no-cache-dir','--no-deps','--index-url','https://download.pytorch.org/whl/cu128','torchvision==0.22.1'],check=True)
subprocess.run([py,'-m','pip','install','--no-cache-dir','-c',str(pins),'numpy==1.26.4','timm==1.0.20','PyYAML==6.0.3','huggingface_hub==2.1.1','safetensors==0.8.0'],check=True)
import torch,timm,numpy,torchvision
assert torch.__version__=='2.7.1+cu128' and torchvision.__version__=='0.22.1+cu128' and timm.__version__=='1.0.20' and numpy.__version__=='1.26.4'
assert not (ROOT/'reports/driver.lock').exists()
with (ROOT/'reports/parallel_sd_worker.log').open('ab') as log:
 p=subprocess.Popen([py,'-u','reports/parallel_sd_worker.py'],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
write_json(ROOT/'reports/parallel_worker_launch.json',{'pid':p.pid,'stage':'SECOND_NODE_GPU_ACCEPTANCE_LAUNCHED','time':utc_now(),'pinned_proven_dependencies':True})
print('WORKER_STARTED',p.pid,flush=True)
