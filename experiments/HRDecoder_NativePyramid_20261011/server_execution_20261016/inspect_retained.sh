set -eu
BASE=/root/autodl-tmp/nativepyramid_sp_v3_20261016
ls -ld "$BASE" "$BASE/venv" "$BASE/code" "$BASE/data" "$BASE/assets"
find "$BASE" -maxdepth 2 -type f -name '*.json' -print
"$BASE/venv/bin/python" - <<'PY'
import json,sys,importlib.metadata as m
from pathlib import Path
import torch
b=Path('/root/autodl-tmp/nativepyramid_sp_v3_20261016')
r={'python':sys.version,'deps':{n:m.version(n) for n in ('torch','torchvision','timm','numpy','opencv-python','Pillow')},
   'cuda':torch.cuda.is_available(),'bf16':torch.cuda.is_bf16_supported()}
for n in ('job_result.json','job_status.json','receipts/shutdown_request.json'):
    p=b/n
    r[n]=json.loads(p.read_text()) if p.is_file() else None
print(json.dumps(r,indent=2))
PY
