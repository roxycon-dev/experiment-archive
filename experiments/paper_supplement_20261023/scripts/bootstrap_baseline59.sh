#!/usr/bin/env bash
set -euo pipefail
task_root=/home/studentuser001/paper_supplement_20261023
env_prefix="$task_root/legacy_env"
mkdir -p "$task_root/assets" "$task_root/logs"
if [ ! -x "$env_prefix/bin/python" ]; then
  curl -fL --retry 3 --connect-timeout 15 --max-time 600 https://repo.anaconda.com/miniconda/Miniconda3-py38_23.11.0-2-Linux-x86_64.sh -o "$task_root/assets/Miniconda3-py38_23.11.0-2-Linux-x86_64.sh"
  sha256sum "$task_root/assets/Miniconda3-py38_23.11.0-2-Linux-x86_64.sh" > "$task_root/logs/miniconda_download.sha256"
  bash "$task_root/assets/Miniconda3-py38_23.11.0-2-Linux-x86_64.sh" -b -p "$env_prefix"
fi
"$env_prefix/bin/python" -m pip install --no-cache-dir 'numpy==1.24.4' 'torch==2.0.1+cu117' 'torchvision==0.15.2+cu117' --extra-index-url https://download.pytorch.org/whl/cu117
"$env_prefix/bin/python" -m pip install --no-cache-dir 'numpy==1.24.4' 'opencv-python-headless==4.9.0.80' 'scipy==1.10.1' 'matplotlib==3.7.5' 'pillow<11' 'scikit-learn==1.3.2' 'timm==0.9.12' 'yapf==0.40.1' 'einops==0.8.0' prettytable terminaltables pandas addict pyyaml packaging
# Reuse the exact already-working CPython3.8/torch2.0.1/cu117 MMCV1.7.1
# binary and its metadata, instead of silently compiling a different version.
while [ ! -f "$task_root/assets/mmcv_legacy_runtime.zip" ]; do sleep 5; done
"$env_prefix/bin/python" - <<'PY'
from pathlib import Path
import hashlib,json,site,zipfile
root=Path('/home/studentuser001/paper_supplement_20261023/assets')
receipt=json.loads((root/'mmcv_legacy_runtime_receipt.json').read_text())
archive=root/'mmcv_legacy_runtime.zip'
assert hashlib.sha256(archive.read_bytes()).hexdigest()==receipt['archive_sha256']
with zipfile.ZipFile(archive) as z:z.extractall(site.getsitepackages()[0])
PY
"$env_prefix/bin/python" - <<'PY'
import json,torch,mmcv,numpy,cv2
from mmcv.ops import roi_align
x=torch.ones((1,2,8,8),device='cuda',requires_grad=True)
r=roi_align(x,torch.tensor([[0.,0.,0.,7.,7.]],device='cuda'),(3,3));r.sum().backward()
assert torch.isfinite(x.grad).all()
from pathlib import Path
receipt={'status':'READY','torch':torch.__version__,'mmcv':mmcv.__version__,'numpy':numpy.__version__,'OpenCV':cv2.__version__,
         'GPU_count':torch.cuda.device_count(),'ROIAlign_CUDA_forward_backward':'PASS','new_training_updates':0}
Path('/home/studentuser001/paper_supplement_20261023/logs/environment_ready.json').write_text(json.dumps(receipt,indent=2))
print(json.dumps(receipt),flush=True)
PY
