set -eu
/root/miniconda3/envs/hrdecoder_q_20261003/bin/python - <<'PY'
from pathlib import Path
import hashlib,json,re,shlex
base=Path('/root/autodl-tmp/nativepyramid_sp_v3_20261016')
source=Path('/usr/bin/shutdown')
raw=source.read_text()
lines=raw.splitlines()
removed=[line for line in lines if line.strip()=='rm -rf /root/.local/share/Trash']
clean=[line for line in lines if line not in removed]
assert len(removed)==1,'Unexpected provider shutdown cleanup line'
body='\n'.join(clean)+'\n'
assert not re.search(r'(^|[;&|\n])\s*(rm|rmdir|mkfs|dd)\b',body),'Unexpected filesystem mutation in provider wrapper'
target=base/'receipts/provider_shutdown_preserve_files.sh'
target.write_text('#!/bin/bash\nset -e\n'+body)
target.chmod(0o700)
receipt={'source_sha256':hashlib.sha256(raw.encode()).hexdigest(),
         'wrapper_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
         'removed_unrelated_trash_cleanup':True,'provider_shutdown_action_preserved':True,
         'wrapper':str(target)}
(base/'receipts/shutdown_wrapper_preparation.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
for name in ('packages/data.tar','assets/hrnetv2_w48-d2186c55.pth'):
    p=base/name
    print(name,p.stat().st_size if p.exists() else 'not present')
PY
