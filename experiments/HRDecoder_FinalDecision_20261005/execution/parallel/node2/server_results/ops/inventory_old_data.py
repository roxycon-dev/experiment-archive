"""Concrete old data candidates by directory and binary suffix; scripts preserved."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from ops.remote import connect,command
from qlib.common import write_json
code="""
from pathlib import Path
import os,json,collections
out={}
for base in ('/root/autodl-tmp/HRDecoder_Next_20261003','/root/autodl-tmp/HRDecoder_Ready_20261002'):
 groups=collections.defaultdict(lambda:{'bytes':0,'files':0,'examples':[]})
 for d,_,names in os.walk(base):
  for name in names:
   p=Path(d)/name
   if p.is_symlink() or p.suffix not in ('.npz','.npy','.pth','.pt','.pkl','.jpg','.png'):continue
   rel=p.relative_to(base);key='/'.join(rel.parts[:min(len(rel.parts)-1,3)])+':'+p.suffix
   entry=groups[key];entry['bytes']+=p.stat().st_size;entry['files']+=1
   if len(entry['examples'])<3:entry['examples'].append(str(p))
 out[base]=dict(groups)
print(json.dumps(out))
"""
with connect() as client:result=json.loads(command(client,"/root/miniconda3/bin/python - <<'PY'\n"+code+"\nPY",90))
write_json(ROOT/'reports/old_data_candidates.json',result);print(json.dumps(result))
