"""Seal the SHA-verified completed-node snapshot before retention flags advance."""
from pathlib import Path
import json,shutil,sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from mirror_remote import ROOT,LOCAL,file_sha,write_json
receipt=json.loads((ROOT/'reports/final_metadata_backup.json').read_text(encoding='utf-8-sig'))
assert receipt['status']=='PASS'
sealed=ROOT/'artifacts/closed_node_metadata';rows={}
for rel,row in receipt['files'].items():
 if not rel.endswith('.json') or not (rel.startswith('work/') or rel.startswith('configs/')):continue
 source=LOCAL/rel;target=sealed/rel;target.parent.mkdir(parents=True,exist_ok=True)
 assert file_sha(source)==row['sha256'],rel
 shutil.copyfile(source,target);assert file_sha(target)==row['sha256']
 rows[rel]=row
write_json(ROOT/'reports/closed_node_metadata_manifest.json',dict(status='PASS',root=str(sealed.resolve()),files=rows,
 interpretation='Exact completed-node snapshot; live cache indexes may later gain retention acknowledgements'))
print(json.dumps(dict(status='PASS',files=len(rows))))
