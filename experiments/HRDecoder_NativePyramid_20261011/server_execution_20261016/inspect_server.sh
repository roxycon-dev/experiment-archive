set -eu
date -u
hostname
id
printf '\n--- GPU ---\n'
nvidia-smi --query-gpu=name,uuid,memory.total,memory.free,driver_version --format=csv
printf '\n--- Storage ---\n'
df -hT / /root /root/autodl-tmp /tmp 2>/dev/null || df -hT
printf '\n--- Active workloads ---\n'
ps -eo pid,etimes,comm,args | grep -E 'python|train|torchrun|verify' | grep -v grep | head -30 || true
printf '\n--- Roots ---\n'
ls -la /root
printf '\n--- Python ---\n'
command -v python || true
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
printf '\n--- Existing data/weight paths ---\n'
find /root/autodl-tmp /root/autodl-fs /data /workspace -maxdepth 5 -type f \( -name '*w48*.pth' -o -name 'data_identity.json' -o -name 'split_assignment.json' \) -print 2>/dev/null | head -60 || true
printf '\n--- Shutdown tools ---\n'
command -v shutdown || true
command -v poweroff || true
command -v supervisorctl || true
