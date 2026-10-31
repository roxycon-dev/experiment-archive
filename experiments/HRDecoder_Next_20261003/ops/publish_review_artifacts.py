"""Copy compact reviewed artifacts to this same experiment's server archive."""
from pathlib import Path
import sys,hashlib,json
sys.path.insert(0,str(Path(__file__).resolve().parent))
from mirror_remote import ROOT,REMOTE,connect,remote_exec
def main():
 names=['G_阶段结论.md','G_manual_review.csv','G_visual_review.json','G_fixed_gain_uncertainty.json',
        'G_PR_IoU_diagnostics.png','G_PR_IoU_diagnostics.pdf']
 c=connect();receipt={}
 try:
  with c.open_sftp() as sftp:
   for name in names:
    source=ROOT/'reports'/name;dest=REMOTE+'/reports/'+name;temp=dest+'.upload'
    sftp.put(str(source),temp);sftp.posix_rename(temp,dest)
    receipt[name]=hashlib.sha256(source.read_bytes()).hexdigest()
  code="import json,pathlib,hashlib\nr=pathlib.Path('"+REMOTE+"')\nexpected=json.loads("+repr(json.dumps(receipt))+ ")\nfor name,h in expected.items():assert hashlib.sha256((r/'reports'/name).read_bytes()).hexdigest()==h\nprint('Review artifacts verified',len(expected))"
  print(remote_exec(c,"/root/miniconda3/bin/python - <<'PY'\n"+code+"\nPY",30))
 finally:c.close()
 (ROOT/'reports/G_review_upload.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
if __name__=='__main__':main()
