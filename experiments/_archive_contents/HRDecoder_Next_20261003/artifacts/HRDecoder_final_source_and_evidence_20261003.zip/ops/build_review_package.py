"""Archive source/provenance and exact local frozen asset references after completion."""
from pathlib import Path
import sys,json,zipfile
sys.path.insert(0,str(Path(__file__).resolve().parent))
from mirror_remote import ROOT,LOCAL,file_sha,write_json
sys.path.insert(0,str(ROOT))
from nextlib.common import source_identity
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def main():
 final=read(LOCAL/'work/final_lock.json');test=read(LOCAL/'work/final_test.json')
 assert test['n_images']==227 and test['lock_sha256']==file_sha(LOCAL/'work/final_lock.json')
 assert source_identity()['sha256']==final['source_sha256']
 spec=read(ROOT/'configs/assets.json');mapping={}
 for name,asset in final['assets'].items():
  candidates=[ROOT/'assets'/spec[name]['file'],Path(spec[name]['source'])]
  path=next((p for p in candidates if p.is_file() and file_sha(p)==asset['sha256']),None)
  assert path is not None,name+' lacks exact frozen local asset'
  mapping[name]=dict(remote_path=asset['path'],local_path=str(path.resolve()),sha256=asset['sha256'])
 write_json(ROOT/'reports/frozen_asset_local_map.json',dict(status='PASS',assets=mapping,
     inference='Frozen G pipeline: champion 75% + U0_highres native slide 25%; threshold strictly >0.5',
     original_winner_retained=True,serialized_sha_exact=True))
 # Explicitly package prepared and executed config identities separately.
 package=ROOT/'artifacts/HRDecoder_final_source_and_evidence_20261003.zip';package.parent.mkdir(exist_ok=True)
 files=[]
 for folder in ['nextlib','trainer','legacy','tools','vendor','ops']:
  files += [(p,p.relative_to(ROOT).as_posix()) for p in (ROOT/folder).rglob('*.py') if '__pycache__' not in p.parts]
 files += [(p,p.name) for p in ROOT.glob('*.py')]
 files += [(p,'prepared/'+p.relative_to(ROOT).as_posix()) for p in (ROOT/'configs').rglob('*.json')]
 sealed=ROOT/'artifacts/closed_node_metadata'
 for folder in ['configs','work']:
  files += [(p,'executed/'+p.relative_to(sealed).as_posix()) for p in (sealed/folder).rglob('*.json')]
 files += [(p,'reviewed/'+p.relative_to(ROOT).as_posix()) for p in (ROOT/'reports').glob('*')
           if p.is_file() and p.suffix in ['.md','.csv','.png','.pdf','.json']
           and p.name not in ['mirror_manifest.json','critical_backup_manifest.json','latest_direct_probe.json','completion_ready.json','reproducibility_package.json']]
 files += [(p,'reviewed/'+p.relative_to(ROOT).as_posix()) for p in (ROOT/'reports/M_contact_sheets').glob('*.png')]
 for gallery in ['G_error_gallery','M_error_gallery']:
  files += [(p,'visual_evidence/'+gallery+'/'+p.name) for p in (LOCAL/'reports'/gallery).glob('*')
            if p.is_file() and p.suffix in ['.png','.csv','.json','.html']]
 manifest={}
 with zipfile.ZipFile(package,'w',zipfile.ZIP_DEFLATED,compresslevel=4) as z:
  for p,rel in files:
   assert p.suffix not in ['.pem','.key']
   z.write(p,rel);manifest[rel]=dict(sha256=file_sha(p),bytes=p.stat().st_size)
  z.writestr('archive_manifest.json',json.dumps(manifest,ensure_ascii=False,indent=2))
 with zipfile.ZipFile(package) as z:assert z.testzip() is None
 write_json(ROOT/'reports/reproducibility_package.json',dict(status='PASS',path=str(package.resolve()),
    sha256=file_sha(package),files=len(manifest),source_sha256=final['source_sha256'],
    weight_storage='Exact frozen originals/compact files remain in local workspace; references verified in frozen_asset_local_map.json',
    secrets_included=False))
 print(json.dumps(dict(status='PASS',files=len(manifest),bytes=package.stat().st_size)))
if __name__=='__main__':main()
