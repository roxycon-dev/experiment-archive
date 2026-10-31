set -eu
/root/miniconda3/envs/hrdecoder_q_20261003/bin/python - <<'PY'
from pathlib import Path
import json
root=Path('/root/autodl-tmp/HRDecoder_Ready_20261002')
out={'test_label_pixels_read':False,'directories':{},'test_metadata_files':[]}
for rel in ('cache/test','cold_storage'):
    base=root/rel
    if base.is_dir():
        for p in sorted(base.iterdir()):
            if p.is_dir():
                out['directories'][str(p)]={'npz_files_here':len(list(p.glob('*.npz'))),
                    'child_directories':[c.name for c in p.iterdir() if c.is_dir()]}
for base in (root/'reports',root/'results',root/'cache/test'):
    if base.is_dir():
        out['test_metadata_files'] += [str(p) for p in base.glob('**/*test*.json')]
print(json.dumps(out))
PY
