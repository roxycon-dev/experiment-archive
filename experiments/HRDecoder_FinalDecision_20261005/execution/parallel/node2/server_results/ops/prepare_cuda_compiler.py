"""Install the absent CUDA12.8 compiler only in this experiment's isolated environment."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from qlib.common import write_json,utc_now
from ops.remote import connect,command
code="""
from pathlib import Path
import subprocess,json
pidfile=Path('/root/autodl-tmp/hrdecoder_final_cuda_install.pid');assert not pidfile.exists()
prefix='/root/miniconda3/envs/hrdecoder_q_20261003'
assert not (Path(prefix)/'bin/nvcc').exists()
worker="import subprocess;subprocess.run(['/root/miniconda3/bin/conda','install','-y','-p','/root/miniconda3/envs/hrdecoder_q_20261003','-c','nvidia','cuda-nvcc=12.8','cuda-cudart-dev=12.8','cuda-cccl=12.8'],check=True)"
with Path('/root/autodl-tmp/hrdecoder_final_cuda_install.log').open('ab') as log:
 p=subprocess.Popen(['/root/miniconda3/bin/python','-u','-c',worker],stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
pidfile.write_text(str(p.pid));print(json.dumps({'pid':p.pid,'log':'/root/autodl-tmp/hrdecoder_final_cuda_install.log','reason':'Torch2.7.1+cu128 installed but nvcc absent; required S/SD CUDA kernel compiler'}))
"""
with connect() as c:result=json.loads(command(c,"/root/miniconda3/bin/python - <<'PY'\n"+code+"\nPY",30))
result['time']=utc_now();write_json(ROOT/'reports/cuda_compiler_install.json',result);print(result)
