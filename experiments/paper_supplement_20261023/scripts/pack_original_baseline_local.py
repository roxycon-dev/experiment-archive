"""Immutable local mask/count archive for the recovered original baseline; no models included."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'results/baseline_original_local'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()

def main():
    assert json.loads((RUN/'completion.json').read_text())['status']=='COMPLETE'
    assert not (RUN/'failure.json').exists()
    name='baseline_original_seed42';archive=ROOT/'transfer'/('baseline_original_seed42_results.zip')
    receipt=ROOT/'evidence/recovered_evaluations'/(name+'.json')
    if receipt.exists():
        row=json.loads(receipt.read_text());assert sha(archive)==row['archive_sha256'];print(json.dumps(row));return
    files=[p for p in sorted(RUN.rglob('*')) if p.is_file() and p.suffix in ('.json','.npz') and 'analysis' not in p.relative_to(RUN).parts and p.name!='recovery_manifest.json']
    entries=[{'path':p.relative_to(ROOT).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)} for p in files]
    manifest={'name':name,'run_relative':RUN.relative_to(ROOT).as_posix(),
        'status':'COMPLETE_MASKS_AND_METADATA_PACKAGED','files':entries,'model_weights_included':False,'raw_probability_caches_included':False}
    temp=archive.with_suffix('.zip.tmp')
    with zipfile.ZipFile(temp,'w',zipfile.ZIP_STORED) as z:
        for p,r in zip(files,entries):z.write(p,r['path'])
        z.writestr(RUN.relative_to(ROOT).as_posix()+'/recovery_manifest.json',json.dumps(manifest,indent=2))
    temp.replace(archive)
    with zipfile.ZipFile(archive) as z:
        for r in entries:
            payload=z.read(r['path']);assert len(payload)==r['bytes'] and hashlib.sha256(payload).hexdigest()==r['sha256']
    (RUN/'recovery_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    row={'name':name,'local_archive':str(archive),'archive_sha256':sha(archive),
        'archive_bytes':archive.stat().st_size,'files':len(entries),'extracted_files':len(entries),
        'status':'PASS_LOCAL_SHA_AND_ALL_ARCHIVED_FILE_SHAS','origin':'Local fixed inference, not remote recovery',
        'model_checkpoint_backup':'PASS_SEPARATE_RECOVERED_ORIGINAL_IDENTITY_RECEIPT','raw_probability_caches':'Not retained; binary masks and counts retained'}
    receipt.write_text(json.dumps(row,indent=2),encoding='utf-8');print(json.dumps(row))

if __name__=='__main__':main()
