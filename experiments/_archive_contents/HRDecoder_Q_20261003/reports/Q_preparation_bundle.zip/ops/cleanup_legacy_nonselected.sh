set -eu
/root/miniconda3/bin/python - <<'PY'
from pathlib import Path
import hashlib,json,time
root=Path('/root/autodl-tmp')
base=root/'HRDecoder_Ready_20261002/cache/val'
# Explicit user authorization: experiment-unrelated disposable server data.
# These are generated probability caches of unselected historical candidates;
# P0 champion/U0, ALL weights, all GT, all logs and per-image JSON are preserved.
names=('A0','H1','U1')
all_rows=[]
for name in names:
 directory=base/name
 if directory.resolve()!=directory or directory.is_symlink(): raise ValueError('Unexpected path')
 paths=sorted(directory.glob('*.npz'))
 if len(paths)!=76: raise ValueError('Unexpected legacy cache count')
 for p in paths:
  if p.is_symlink() or p.resolve().parent!=directory: raise ValueError('Escaping cleanup target')
  h=hashlib.sha256()
  with p.open('rb') as f:
   for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
  all_rows.append({'path':str(p),'sha256':h.hexdigest(),'bytes':p.stat().st_size})
record={'authorization':'Current user authorized deletion of experiment-unrelated data',
 'kind':'regenerable_legacy_nonselected_probability_caches_only',
 'raw_probability_backup_claimed':False,'keep':['champion','U0','all_weights','all_labels','all_metadata'],
 'files':all_rows,'bytes':sum(r['bytes'] for r in all_rows),'time':time.time(),'status':'VERIFIED_TARGETS'}
out=root/'HRDecoder_Q_20261003/reports/legacy_cache_cleanup.json'
out.parent.mkdir(parents=True,exist_ok=True)
out.write_text(json.dumps(record,indent=2))
for row in all_rows:Path(row['path']).unlink()
record['status']='DELETED_AUTHORIZED_DISPOSABLE_CACHES'
out.write_text(json.dumps(record,indent=2))
print(json.dumps({'status':record['status'],'files':len(all_rows),'freed_bytes':record['bytes']}))
PY
df -h /root/autodl-tmp
