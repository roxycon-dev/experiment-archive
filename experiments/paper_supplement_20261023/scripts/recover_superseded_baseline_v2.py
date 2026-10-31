"""Preserve the superseded partial training attempt; does not resume or evaluate it."""
import json
from pathlib import Path
import shlex
import zipfile
import hashlib
import torch
from training_remote_control import ROOT,REMOTE,LAN,connect,execute
from verified_ssh_download import download,sha,write
from verified_parallel_download import download as parallel_download

def main():
    receipt_path=ROOT/'evidence/superseded_baseline_v2_recovery.json'
    if receipt_path.exists():
        receipt=json.loads(receipt_path.read_text());assert sha(Path(receipt['checkpoint_local']))==receipt['checkpoint_sha256']
        print(json.dumps({'status':receipt['status'],'already_preserved':True}));return
    with connect() as client:
        code=r'''
import hashlib,json,zipfile
from pathlib import Path
r=Path(ROOT_VALUE);base=r/'baseline_v2';stop=r/'logs/baseline_v2_superseded_by_recovered_original.json'
assert json.loads(stop.read_text())['status']=='SUPERSEDED_BY_RECOVERED_ORIGINAL_BASELINE'
checkpoint=base/'runs/baseline_seed42/iter_2000.pth'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
paths=[base/'baseline_reproduction_ddp2.py',base/'baseline_train_entry.py']+list((base/'runs/baseline_seed42').glob('*.log*'))+list((r/'logs').glob('baseline_v2*.json'))+list((r/'logs').glob('baseline_v2*.log'))
paths=sorted(set(p for p in paths if p.is_file()))
manifest={'files':[{'path':p.relative_to(r).as_posix(),'sha256':sha(p),'bytes':p.stat().st_size} for p in paths]}
folder=r/'export_bundles';folder.mkdir(exist_ok=True);archive=folder/'superseded_baseline_v2_history.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
 for p,row in zip(paths,manifest['files']):z.write(p,row['path'])
 z.writestr('backup_manifest.json',json.dumps(manifest,indent=2))
print(json.dumps({'checkpoint':{'remote':str(checkpoint),'sha256':sha(checkpoint),'bytes':checkpoint.stat().st_size},'history':{'remote':str(archive),'sha256':sha(archive),'bytes':archive.stat().st_size}}))
'''.replace('ROOT_VALUE',repr(LAN))
        result=execute(client,'/usr/bin/python3 -c '+shlex.quote(code),'59',timeout=120)
        assert result['exit_code']==0,result
        files=json.loads(result['stdout']);local={}
        for kind,item in files.items():
            target=REMOTE+'/recovered_archives/'+Path(item['remote']).name
            copied=execute(client,shlex.join(['scp','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-i','/root/.ssh/codex_supp_to59_20261023','-P','40560','studentuser001@127.0.0.1:'+item['remote'],target]),timeout=180)
            assert copied['exit_code']==0,copied
            destination=ROOT/('models/superseded_baseline_v2_u2000.pth' if kind=='checkpoint' else 'transfer/superseded_baseline_v2_history.zip')
            method=parallel_download if kind=='checkpoint' else download
            method(client,target,destination,item['sha256'],item['bytes'],ROOT/'results/superseded_baseline_v2_transfer.json')
            local[kind]=destination
    with zipfile.ZipFile(local['history']) as z:
        manifest=json.loads(z.read('backup_manifest.json'))
        for row in manifest['files']:
            payload=z.read(row['path']);assert hashlib.sha256(payload).hexdigest()==row['sha256'] and len(payload)==row['bytes']
            destination=(ROOT/'evidence/superseded_baseline_v2_history'/row['path']).resolve()
            destination.relative_to((ROOT/'evidence/superseded_baseline_v2_history').resolve())
            destination.parent.mkdir(parents=True,exist_ok=True)
            if destination.exists():assert destination.read_bytes()==payload
            else:destination.write_bytes(payload)
    torch.set_num_threads(2)
    x=torch.load(local['checkpoint'],map_location='cpu',weights_only=False,mmap=True)
    assert x['meta']['iter']==1999 and x['optimizer']['state']
    def finite(value):
        if isinstance(value,torch.Tensor):
            if value.is_floating_point():assert bool(torch.isfinite(value).all())
            return 1
        if isinstance(value,dict):return sum(finite(v) for v in value.values())
        if isinstance(value,(tuple,list)):return sum(finite(v) for v in value)
        return 0
    checked=finite({'state':x['state_dict'],'optimizer':x['optimizer']})
    receipt={'status':'PASS_SUPERSEDED_PARTIAL_ATTEMPT_PRESERVED_LOCALLY',
        'checkpoint_local':str(local['checkpoint']),'checkpoint_sha256':files['checkpoint']['sha256'],
        'checkpoint_update':2000,'last_logged_training_update':2980,'not_a_completed40k_run':True,
        'history_archive':str(local['history']),'history_archive_sha256':files['history']['sha256'],
        'history_files':manifest['files'],'finite_model_optimizer_tensors':checked,
        'used_as_primary_baseline':False,'training_resumed':False,'overall_campaign_complete':False}
    write(receipt_path,receipt)
    write(ROOT/'evidence/terminal_backups/superseded_baseline_v2_u2000.json',
          {'path':str(local['checkpoint']),'sha256':files['checkpoint']['sha256'],
           'kind':'superseded_partial_baseline_attempt','not_primary_baseline':True})
    print(json.dumps({'status':receipt['status'],'history_files':len(manifest['files']),'finite_tensors':checked}))

if __name__=='__main__':main()
