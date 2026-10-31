"""Only this task's exact C-drive downloads; no broad deletion or glob purge."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from qlib.common import read_json,write_json,sha,utc_now
downloads=Path.home()/'Downloads';receipt=read_json(ROOT/'reports/TJDR_download_receipt.json')
assert receipt['stage']=='DOWNLOAD_COMPLETE' and len(receipt['completed_files'])==896
for row in receipt['completed_files']:assert sha(ROOT/row['path'])==row['sha256']
paths=[downloads/'train.zip',downloads/'train (1).zip',downloads/'未确认 626605.crdownload']
results=[]
for path in paths:
 assert path.resolve().parent==downloads.resolve()
 if not path.exists():continue
 if path.suffix=='.zip':
  if sha(path)!=receipt['archive_sha256']:
   import zipfile,hashlib
   expected={row['path'].split('inputs/TJDR/',1)[1]:row['sha256'] for row in receipt['completed_files']}
   with zipfile.ZipFile(path) as z:
    names={p.filename for p in z.infolist() if not p.is_dir()};assert names==set(expected)
    for name in sorted(names):
     h=hashlib.sha256()
     with z.open(name) as f:
      for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
     assert h.hexdigest()==expected[name]
 else:
  with path.open('rb') as f:assert b'train/annotation/TJDR_train_' in f.read(200)
 row={'path':str(path),'bytes':path.stat().st_size}
 try:path.unlink();row['removed']=True
 except PermissionError:row.update(removed=False,reason='Download still holds file; retry only this exact task-created path after it releases')
 results.append(row)
write_json(ROOT/'reports/C_drive_download_cleanup.json',{'time':utc_now(),'files':results,'other_C_files_untouched':True})
print(results)
