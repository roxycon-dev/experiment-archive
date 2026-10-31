set -eu
BASE=/root/autodl-tmp/nativepyramid_sp_v3_20261016
test ! -e "$BASE"
mkdir -p "$BASE/code" "$BASE/assets" "$BASE/data" "$BASE/receipts" "$BASE/packages" "$BASE/tmp"
printf '\n--- Alternative Python ---\n'
ls -l /root/miniconda3/envs/hrdecoder_q_20261003/bin/python* 2>/dev/null || true
/root/miniconda3/envs/hrdecoder_q_20261003/bin/python - <<'PY'
import sys,json,importlib.metadata as m
r={'python':sys.version,'executable':sys.executable,'dependencies':{}}
for name in ('torch','torchvision','timm','numpy','opencv-python','Pillow'):
    try:r['dependencies'][name]=m.version(name)
    except m.PackageNotFoundError:r['dependencies'][name]=None
print(json.dumps(r,indent=2))
PY
printf '\n--- Bootstrap availability ---\n'
/root/miniconda3/bin/conda --version
ls -l /root/HRDecoder/data/DDR_split_613/images/train | head -5 || true
printf '\n--- Startup receipt ---\n'
nvidia-smi -q > "$BASE/receipts/nvidia_smi_start.txt"
df -hT > "$BASE/receipts/disk_start.txt"
