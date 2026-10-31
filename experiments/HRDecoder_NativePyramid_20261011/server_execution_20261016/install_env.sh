set -eu
BASE=/root/autodl-tmp/nativepyramid_sp_v3_20261016
export TMPDIR="$BASE/tmp"
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_NO_CACHE_DIR=1
/root/miniconda3/envs/hrdecoder_q_20261003/bin/python -m venv "$BASE/venv"
PY="$BASE/venv/bin/python"
"$PY" -m pip install --upgrade pip --index-url https://pypi.org/simple
"$PY" -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
"$PY" -m pip install timm==1.0.27 numpy==2.4.1 opencv-python==4.13.0.90 Pillow==12.1.0 --index-url https://pypi.org/simple
"$PY" -m pip check
"$PY" -m pip freeze > "$BASE/receipts/pip_freeze.txt"
"$PY" - <<'PY'
import torch,timm,cv2,numpy,sys,json
print(json.dumps({'python':sys.version,'torch':torch.__version__,'timm':timm.__version__,
                  'cv2':cv2.__version__,'numpy':numpy.__version__,
                  'bf16':torch.cuda.is_bf16_supported(),'gpu':torch.cuda.get_device_name()},indent=2))
PY
touch "$BASE/receipts/environment_ready"
