"""Extend the prior archive audit to both user-specified trees, including ignored assets."""
import json
from pathlib import Path
import subprocess
import tarfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
def norm(p): return str(p).replace('\\','/').lower()
old = {norm(r['path']):r for r in json.loads((ROOT/'evidence/archive_checkpoint_audit.json').read_text(encoding='utf-8'))}
paths = [Path(p) for p in (ROOT/'evidence/all_two_roots_unignored_assets.txt').read_text(encoding='utf-8-sig').splitlines()]
results=[]
for p in paths:
    if p.suffix.lower() not in ('.zip','.rar','.7z','.tar','.gz'):
        continue
    if norm(p) in old:
        results.append({**old[norm(p)],'prior_audit_reused':True})
        continue
    row={'path':str(p),'bytes':p.stat().st_size,'prior_audit_reused':False}
    try:
        if p.suffix.lower()=='.rar':
            names=subprocess.run(['tar','-tf',str(p)],capture_output=True,text=True,check=True,encoding='utf-8').stdout.splitlines()
            members=[(n,None) for n in names]
        elif zipfile.is_zipfile(p):
            with zipfile.ZipFile(p) as z:members=[(m.filename,m.file_size) for m in z.infolist() if not m.is_dir()]
        else:
            with tarfile.open(p,'r:*') as z:members=[(m.name,m.size) for m in z if m.isfile()]
        row.update(status='HEADERS_LISTED_NO_EXTRACTION',members=len(members),
                   checkpoint_members=[{'name':n,'bytes':s} for n,s in members if n.lower().endswith(('.pth','.pt','.ckpt','.safetensors')) or 'best_miou' in n.lower()])
    except Exception as exc:row.update(status='UNREADABLE_CONTAINER',error=str(exc))
    results.append(row)
    print(json.dumps({'path':str(p),'status':row['status'],'candidate_count':len(row.get('checkpoint_members',[]))},ensure_ascii=False),flush=True)
(ROOT/'evidence/two_roots_archive_audit.json').write_text(json.dumps(results,indent=2,ensure_ascii=False),encoding='utf-8')
print(json.dumps({'archives':len(results),'new_archives':sum(not r['prior_audit_reused'] for r in results),
                  'unreadable':[r['path'] for r in results if r['status'] not in ['HEADERS_LISTED_NO_EXTRACTION']],
                  'candidate_32000':[r for r in results if any('32000' in m['name'] for m in r.get('checkpoint_members',[]))]},ensure_ascii=False),flush=True)
