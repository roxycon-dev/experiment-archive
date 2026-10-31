from pathlib import Path
import sys,json
sys.path.insert(0,str(Path(__file__).resolve().parent))
from mirror_remote import connect,remote_exec
CODE=r'''
from pathlib import Path
import sys,json
r=Path('/root/autodl-tmp/HRDecoder_Next_20261003');sys.path.insert(0,str(r))
from nextlib.common import trusted_load,read_json,digest
out={}
for arm in ['M0','M1']:
 p=r/'work/M_seed42'/arm;ck=trusted_load(p/'last.pth')
 out[arm]=dict(checkpoint_identity=ck['identity'],identity_file=read_json(p/'identity.json'),
  update=ck['update'],cursor=ck['stream_cursor'],history=[x['update'] for x in ck['history']])
print(json.dumps(out))
'''
c=connect()
try:print(remote_exec(c,"/root/miniconda3/bin/python - <<'PY'\n"+CODE+"\nPY",30))
finally:c.close()
