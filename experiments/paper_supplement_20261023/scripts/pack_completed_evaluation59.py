"""Package completed masks and provenance only; large reusable probability caches stay on the server."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

TASK=Path('/home/studentuser001/paper_supplement_20261023')
ROOT=TASK/'evaluation_workspace/paper_supplement_20261023'

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()

parser=argparse.ArgumentParser();parser.add_argument('--name',required=True);args=parser.parse_args()
assert args.name in ['baseline_seed42']+[a+'_seed'+str(s) for s in (42,2024,3407) for a in ('C','S')]
run=ROOT/'results/baseline_retrained59' if args.name=='baseline_seed42' else ROOT/'results/paired_seed59'/args.name
assert json.loads((run/'completion.json').read_text())['status']=='COMPLETE'
assert json.loads((run/'status.json').read_text())['stage']=='COMPLETE'
assert not (run/'failure.json').exists()
folder=TASK/'export_bundles';folder.mkdir(exist_ok=True)
archive=folder/(args.name+'_results.zip');receipt=archive.with_suffix('.receipt.json')
if receipt.exists():
    value=json.loads(receipt.read_text());assert sha(archive)==value['archive_sha256'];print(json.dumps(value));raise SystemExit(0)
files=[p for p in sorted(run.rglob('*')) if p.is_file() and p.suffix in ('.json','.npz') and 'analysis' not in p.relative_to(run).parts]
entries=[{'path':p.relative_to(ROOT).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)} for p in files]
manifest={'name':args.name,'run_relative':run.relative_to(ROOT).as_posix(),'status':'COMPLETE_MASKS_AND_METADATA_PACKAGED',
          'files':entries,'model_weights_included':False,'raw_probability_caches_included':False}
temp=archive.with_suffix('.zip.tmp')
with zipfile.ZipFile(temp,'w',zipfile.ZIP_STORED) as z:
    for p,r in zip(files,entries):z.write(p,r['path'])
    z.writestr(run.relative_to(ROOT).as_posix()+'/recovery_manifest.json',json.dumps(manifest,indent=2))
temp.replace(archive)
value={'name':args.name,'archive':str(archive),'archive_sha256':sha(archive),'archive_bytes':archive.stat().st_size,
       'files':len(entries),'status':'READY_FOR_LOCAL_RECOVERY','model_checkpoint_backup':'SEPARATE_PENDING'}
receipt.write_text(json.dumps(value,indent=2));print(json.dumps(value))
