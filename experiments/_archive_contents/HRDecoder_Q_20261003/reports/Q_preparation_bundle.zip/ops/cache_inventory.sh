/root/miniconda3/bin/python - <<'PY'
from pathlib import Path
import hashlib,json
base=Path('/root/autodl-tmp/HRDecoder_Ready_20261002/cache/val/H1').resolve()
expected=Path('/root/autodl-tmp/HRDecoder_Ready_20261002/cache/val/H1')
if base!=expected: raise ValueError('Unexpected cleanup candidate symlink')
rows=[]
for p in sorted(base.glob('*.npz')):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''): h.update(b)
 rows.append({'name':p.name,'sha256':h.hexdigest(),'bytes':p.stat().st_size})
print(json.dumps({'directory':str(base),'files':rows}))
PY
