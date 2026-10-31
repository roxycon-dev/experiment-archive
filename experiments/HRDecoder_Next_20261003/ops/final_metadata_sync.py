"""Exact SHA mirror of completed experiment metadata, never reruns inference."""
from pathlib import Path
import sys,json
sys.path.insert(0,str(Path(__file__).resolve().parent))
from mirror_remote import ROOT,LOCAL,REMOTE,connect,remote_exec,file_sha,write_json
CODE=r'''
from pathlib import Path
import json,hashlib
r=Path('/root/autodl-tmp/HRDecoder_Next_20261003')
assert json.loads((r/'reports/server_stage.json').read_text())['status']=='complete'
assert json.loads((r/'reports/campaign_completion.json').read_text())['status']=='EXPERIMENTS_COMPLETE_BACKUP_REVIEW_PENDING'
files=[]
for base in [r/'configs',r/'work',r/'reports']:
 for p in base.rglob('*'):
  if not p.is_file() or p.suffix not in ['.json','.log','.txt','.md','.png','.csv','.html','.pdf']:continue
  if 'mirror_manifests' in p.parts or 'remote_responses' in p.parts:continue
  if p.name in ['verified_G_evictions.json']:continue
  h=hashlib.sha256()
  with p.open('rb') as f:
   for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
  files.append(dict(path=p.relative_to(r).as_posix(),sha256=h.hexdigest(),size=p.stat().st_size))
print(json.dumps(files))
'''
def main():
 c=connect();verified={};downloads=0
 try:
  files=json.loads(remote_exec(c,"/root/miniconda3/bin/python - <<'PY'\n"+CODE+"\nPY",60))
  with c.open_sftp() as sftp:
   for row in files:
    target=LOCAL/row['path'];target.parent.mkdir(parents=True,exist_ok=True)
    if not target.exists() or target.stat().st_size!=row['size'] or file_sha(target)!=row['sha256']:
     temp=target.with_suffix(target.suffix+'.final_download')
     sftp.get(REMOTE+'/'+row['path'],str(temp));assert file_sha(temp)==row['sha256']
     temp.replace(target);downloads+=1
    verified[row['path']]=dict(sha256=row['sha256'],size=row['size'])
 finally:c.close()
 write_json(ROOT/'reports/final_metadata_backup.json',dict(status='PASS',files=verified))
 print(json.dumps(dict(status='PASS',files=len(verified),downloads=downloads)))
if __name__=='__main__':main()
