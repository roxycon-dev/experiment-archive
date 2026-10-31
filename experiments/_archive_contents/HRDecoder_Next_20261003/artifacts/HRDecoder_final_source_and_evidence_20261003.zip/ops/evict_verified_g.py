"""Release completed, nonwinning new G arrays only after verified local SHA backup."""
from pathlib import Path
import sys,json
sys.path.insert(0,str(Path(__file__).resolve().parent))
from mirror_remote import ROOT,LOCAL,REMOTE,connect,remote_exec

def main():
    assert (ROOT/'reports/allow_g_evict.json').exists()
    manifest=json.loads((ROOT/'reports/mirror_manifest.json').read_text(encoding='utf-8'))
    groups={}
    for rel,row in manifest.items():
        if rel.startswith('work/G/') and rel.endswith('.npz'):
            groups.setdefault(str(Path(rel).parent).replace('\\','/'),[]).append(dict(path=rel,**row))
    c=connect()
    try:
        state=json.loads(remote_exec(c,"/root/miniconda3/bin/python - <<'PY'\nfrom pathlib import Path\nimport json\nr=Path('"+REMOTE+"');assert (r/'reports/G_analysis_complete.json').exists()\nj=json.loads((r/'reports/G_deployment_candidates.json').read_text());print(json.dumps(j['best']))\nPY"))
        for cache,items in groups.items():
            name=cache.split('/')[-1]
            if name in ['champion','U0_native_whole',state['key']] or name.startswith('U0_highres'):continue
            if len(items)!=76:continue
            if any(not (LOCAL/x['path']).exists() or (LOCAL/x['path']).stat().st_size!=x['size'] for x in items):continue
            code="""import pathlib,json,hashlib,time
r=pathlib.Path('/root/autodl-tmp/HRDecoder_Next_20261003');items=json.loads(PAYLOAD)
assert (r/'reports/G_analysis_complete.json').exists()
best=json.loads((r/'reports/G_deployment_candidates.json').read_text())['best']['key']
cache=r/items[0]['path'];cache=cache.parent;assert cache.name not in ['champion','U0_native_whole',best]
assert not cache.name.startswith('U0_highres') and len(items)==76
released=0
for item in items:
 p=r/item['path']
 if not p.exists():continue
 assert (r/'work').resolve() in p.resolve().parents or (r/'cold_storage').resolve() in p.resolve().parents
 assert not p.is_symlink() and p.suffix=='.npz'
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 assert h.hexdigest()==item['sha256'] and p.stat().st_size==item['size']
 released+=item['size'];p.unlink()
if released:
 p=cache/'index.json';j=json.loads(p.read_text());j['probabilities_backed_up_then_evicted']=True;p.write_text(json.dumps(j,indent=2))
 record=r/'reports/verified_G_evictions.json';rows=json.loads(record.read_text()) if record.exists() else []
 rows.append(dict(cache=str(cache),bytes=released,time=time.time(),count=76));record.write_text(json.dumps(rows,indent=2))
print(json.dumps(dict(cache=str(cache),bytes_released=released)))
""".replace('PAYLOAD',repr(json.dumps(items)))
            print(remote_exec(c,"/root/miniconda3/bin/python - <<'PY'\n"+code+"\nPY",180),flush=True)
    finally:c.close()

if __name__=='__main__':main()
