import json,tarfile,hashlib
from pathlib import Path
from collections import Counter
base=Path(r'D:\dataset\experiment\autodl100')
a=Path(r'C:\Users\GuanXingRen\AppData\experment\analysis\split_assignment.json')
m=json.loads(a.read_text(encoding='utf-8'))
root=Path(r'D:\HRDecoder\data\DDR')
rows=[]
with tarfile.open(base/'bundle/ddr_split613.tar','w') as t:
 for name,v in sorted(m.items()):
  for kind,filename in [('images',name),('labels',Path(name).stem+'.png')]:
   p=root/kind/v['original_subset']/filename
   assert p.is_file(),str(p)
   dest=f"data/DDR_split_613/{kind}/{v['split']}/{filename}"
   t.add(p,arcname=dest,recursive=False)
   rows.append({'path':dest,'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
(base/'bundle/split_assignment.json').write_bytes(a.read_bytes())
(base/'bundle/data_manifest.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
work=base/'work/HRDecoder'
work.mkdir(parents=True,exist_ok=True)
with tarfile.open(base/'bundle/hrdecoder_source.tar') as t:
 t.extractall(work,filter='data')
print(json.dumps({'split':dict(Counter(v['split'] for v in m.values())),'files':len(rows),'archive_bytes':(base/'bundle/ddr_split613.tar').stat().st_size}))
