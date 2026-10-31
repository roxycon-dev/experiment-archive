set -eu
export PATH=/root/miniconda3/bin:$PATH
python - <<'PY'
import importlib.metadata as m, json, sys
r={'python':sys.version,'executable':sys.executable,'dependencies':{}}
for name in ('torch','torchvision','timm','numpy','opencv-python','Pillow'):
    try:r['dependencies'][name]=m.version(name)
    except m.PackageNotFoundError:r['dependencies'][name]=None
try:
    import torch
    r.update(cuda_available=torch.cuda.is_available(),cuda_build=torch.version.cuda,
             bf16_supported=torch.cuda.is_available() and torch.cuda.is_bf16_supported())
except Exception as e:r['torch_import_error']=str(e)
print(json.dumps(r,indent=2))
PY
printf '\n--- Existing assets ---\n'
find /root/HRDecoder /root/autodl-tmp /root/transfer -maxdepth 6 -type f \( -name '*w48*.pth' -o -name 'data_identity.json' -o -name 'split_assignment.json' -o -name 'python3.11' -o -name 'python3.12' \) -print 2>/dev/null | head -60 || true
du -sh /root/HRDecoder/data/* 2>/dev/null || true
printf '\n--- Shutdown support ---\n'
command -v shutdown || true
command -v poweroff || true
command -v autodl || true
cat /root/finalize_and_shutdown.sh | tail -30
printf '\n--- Python environments ---\n'
ls /root/miniconda3/envs 2>/dev/null || true
printf '\n--- Python binaries ---\n'
ls /usr/bin/python* /usr/local/bin/python* 2>/dev/null || true
