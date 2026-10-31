#!/usr/bin/env bash
set -euo pipefail
task_root=/home/studentuser001/paper_supplement_20261023
env_prefix="$task_root/evaluation_env"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
if [ ! -x "$env_prefix/bin/python" ]; then
  "$task_root/legacy_env/bin/conda" create -y -p "$env_prefix" --override-channels -c https://repo.anaconda.com/pkgs/main 'python=3.11' pip
fi
"$env_prefix/bin/python" -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu118 'torch==2.7.1+cu118' 'torchvision==0.22.1+cu118'
"$env_prefix/bin/python" -m pip install --no-cache-dir 'numpy==1.26.4' 'pillow==11.1.0' 'scipy==1.15.3' 'opencv-python-headless==4.9.0.80' 'einops==0.8.0' 'timm==0.9.12' pyyaml packaging ninja
"$env_prefix/bin/python" - <<'PY'
from pathlib import Path
import json,sys,torch,numpy,cv2
root=Path('/home/studentuser001/paper_supplement_20261023')
receipt={'stage':'EVALUATION_ENVIRONMENT_INSTALLED','python':sys.version,'torch':torch.__version__,'CUDA':torch.version.cuda,'numpy':numpy.__version__,'OpenCV':cv2.__version__,
         'GPU_inference_gate':'PENDING_FIXED_VALIDATION_CANARY','uses_GPU_for_install':False}
(root/'logs/evaluation_environment.json').write_text(json.dumps(receipt,indent=2))
print(json.dumps(receipt),flush=True)
PY
