from pathlib import Path
import json,sys,hashlib
sys.path.insert(0,str(Path(__file__).resolve().parent))
from mirror_remote import ROOT,REMOTE,connect,write_json,file_sha

def main():
 c=connect();folder='reports/M_error_gallery';target=ROOT/'server_results'/folder;target.mkdir(parents=True,exist_ok=True);rows=[]
 try:
  with c.open_sftp() as sftp:
   for item in sftp.listdir_attr(REMOTE+'/'+folder):
    assert Path(item.filename).name==item.filename
    if Path(item.filename).suffix not in ['.png','.json','.csv','.html']:continue
    local=target/item.filename;sftp.get(REMOTE+'/'+folder+'/'+item.filename,str(local))
    with sftp.open(REMOTE+'/'+folder+'/'+item.filename,'rb') as f:
     h=hashlib.sha256()
     for chunk in iter(lambda:f.read(1024**2),b''):h.update(chunk)
    assert file_sha(local)==h.hexdigest()
    rows.append(dict(file=item.filename,sha256=h.hexdigest(),size=local.stat().st_size))
 finally:c.close()
 write_json(ROOT/'reports/M_review_backup.json',dict(status='PASS',files=rows))
 print(json.dumps(dict(files=len(rows),images=len(list(target.glob('*.png'))))))
if __name__=='__main__':main()
