"""Read immutable endpoint markers and package exact training source/provenance, without loading a GPU."""
from pathlib import Path
import hashlib
import json
import zipfile

ROOT=Path(__file__).resolve().parents[1];TRAIN=ROOT/'training_v1'
PARENT='68d7b5821fab1210b3336cf2a63b9de893f24fc32876f5255d7d662084b5e728'

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()

def archive(paths,target):
    receipt=target.with_suffix('.receipt.json')
    if receipt.exists():return json.loads(receipt.read_text())
    rows=[{'path':p.relative_to(ROOT).as_posix(),'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(set(paths))]
    temp=target.with_suffix('.zip.tmp');target.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        for row in rows:z.write(ROOT/row['path'],row['path'])
        z.writestr('backup_manifest.json',json.dumps({'files':rows,'scope':'Exact source or completed training provenance; model checkpoint handled separately'},indent=2))
    temp.replace(target)
    result={'remote':str(target),'sha256':sha(target),'bytes':target.stat().st_size,'archive_entries':len(rows)}
    receipt.write_text(json.dumps(result,indent=2));return result

items=[];folder=ROOT/'source_backup';folder.mkdir(exist_ok=True)
first=TRAIN/'runs/C_seed2024/identity.json'
if first.exists():
    identity=json.loads(first.read_text());assert identity['parent_full_sha256']==PARENT
    source=[]
    for rel,digest in identity['source_files'].items():
        p=TRAIN/rel;assert sha(p)==digest;source.append(p)
    source+=[TRAIN/'protocol.json']+list((TRAIN/'reports').glob('*.json'))
    value=archive(source,folder/'exact_training_source.zip');items.append(dict(value,name='exact_training_source',kind='source'))
completed=[];exported=[]
for seed in (2024,3407):
    stream=TRAIN/f'reports/new_stream_DDR_seed{seed}.jsonl'
    identity_file=stream.with_suffix('.identity.json')
    if identity_file.exists():
        ident=json.loads(identity_file.read_text());assert ident['seed']==seed and ident['rows']==160008 and ident['schedule_horizon']==32000
        assert sha(stream)==ident['sha256']
        value=archive([stream,identity_file],folder/f'sampling_stream_seed{seed}.zip')
        items.append(dict(value,name=f'sampling_stream_seed{seed}',kind='sampling_stream'))
for arm in ('C','S'):
    for seed in (2024,3407):
        name=arm+'_seed'+str(seed);run=TRAIN/'runs'/name;done=run/'completion.json'
        if not done.exists():continue
        if json.loads((run/'status.json').read_text()).get('stage')!='COMPLETE':continue
        finish=json.loads(done.read_text());identity=json.loads((run/'identity.json').read_text())
        assert finish['status']=='COMPLETE' and finish['completed']==finish['EMA_completed']==20000
        assert finish['schedule_horizon']==32000 and identity['parent_full_sha256']==PARENT
        p=run/'last.pth';items.append({'name':name+'_full_u20000','kind':'full_checkpoint','remote':str(p),
               'sha256':finish['endpoint_checkpoint_sha256'],'bytes':p.stat().st_size,'arm':arm,'seed':seed})
        evidence=[p for p in run.glob('*') if p.is_file() and p.suffix in ('.json','.jsonl')]
        value=archive(evidence,folder/(name+'_training_evidence.zip'));items.append(dict(value,name=name+'_training_evidence',kind='training_evidence'))
        completed.append(name)
        exported_receipt=ROOT/'export_weights'/(name+'_export.json')
        if exported_receipt.exists():
            export=json.loads(exported_receipt.read_text());assert export['source_checkpoint_sha256']==finish['endpoint_checkpoint_sha256']
            p=ROOT/'export_weights'/(name+'_EMA20k.pth')
            items.append({'name':name+'_EMA20k','kind':'EMA_export','remote':str(p),'sha256':export['weight_sha256'],'bytes':p.stat().st_size})
            value=archive([exported_receipt],folder/(name+'_EMA_export_receipt.zip'));items.append(dict(value,name=name+'_EMA_export_receipt',kind='export_receipt'))
            exported.append(name)
print(json.dumps({'items':items,'new_completed_runs':completed,'new_exported_runs':exported,'campaign_complete':False}))
