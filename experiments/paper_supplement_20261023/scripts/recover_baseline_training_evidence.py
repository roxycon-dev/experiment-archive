"""Archive completed baseline logs/config/selection, without touching training or selecting a model."""
import hashlib
import json
from pathlib import Path
import shlex
import zipfile
from training_remote_control import ROOT, REMOTE, LAN, connect, execute
from verified_ssh_download import download, sha, write


def main():
    receipt_path = ROOT/'evidence/baseline_training_history_receipt.json'
    if receipt_path.exists():
        saved = json.loads(receipt_path.read_text(encoding='utf-8'))
        assert sha(Path(saved['archive'])) == saved['archive_sha256']
        print(json.dumps(saved)); return
    remote_code = r'''
from pathlib import Path
import hashlib,json,zipfile
task=Path(TASK_VALUE);base=task/'baseline_v1'
job=task/'evaluation_workspace/paper_supplement_20261023/eval_jobs/baseline_seed42.json'
if not job.exists():
 print(json.dumps({'status':'WAITING_ORIGINAL40K_AND_VALIDATION_SELECTION'}));raise SystemExit
selection=json.loads(job.read_text());assert selection['training_finished_at']==40000
assert {r['training_node'] for r in selection['all_validation_records']}==set(range(4000,40001,4000))
paths=[job,base/'baseline_reproduction_ddp2.py']
paths+=list((base/'runs/baseline_seed42').glob('*.log'))+list((base/'runs/baseline_seed42').glob('*.log.json'))
paths+=list((base/'runs/baseline_seed42').glob('*.py'))
paths+=[p for p in (task/'logs').glob('baseline*') if p.is_file() and p.suffix in ('.json','.log','.txt')]
paths=sorted(set(paths));assert any(p.name.endswith('.log.json') for p in paths)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
index={'status':'COMPLETE_BASELINE_TRAINING_HISTORY','selected_node':selection['selected_node'],
 'files':[{'path':str(p.relative_to(task)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in paths]}
target=task/'export_bundles/baseline_training_history.zip';target.parent.mkdir(exist_ok=True)
with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for p in paths:z.write(p,str(p.relative_to(task)))
 z.writestr('baseline_history_manifest.json',json.dumps(index,indent=2))
print(json.dumps({'status':'READY','remote':str(target),'sha256':sha(target),'bytes':target.stat().st_size}))
'''.replace('TASK_VALUE', repr(LAN))
    with connect() as client:
        response=execute(client, '/usr/bin/python3 -c '+shlex.quote(remote_code), '59', timeout=180)
        assert response['exit_code']==0, response
        item=json.loads(response['stdout'])
        if item['status']!='READY':print(json.dumps(item));return
        target=REMOTE+'/recovered_archives/baseline_training_history.zip'
        command=shlex.join(['scp','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes',
            '-i','/root/.ssh/codex_supp_to59_20261023','-P','40560',
            'studentuser001@127.0.0.1:'+item['remote'],target])
        assert execute(client,command,timeout=180)['exit_code']==0
        local=ROOT/'transfer/baseline_training_history.zip'
        download(client,target,local,item['sha256'],item['bytes'],ROOT/'results/baseline_history_transfer.json')
    destination=ROOT/'evidence/baseline_training_history'
    with zipfile.ZipFile(local) as z:
        index=json.loads(z.read('baseline_history_manifest.json'))
        for row in index['files']:
            payload=z.read(row['path']);assert len(payload)==row['bytes'] and hashlib.sha256(payload).hexdigest()==row['sha256']
            p=(destination/row['path']).resolve();p.relative_to(destination.resolve())
            if p.exists():assert sha(p)==row['sha256']
            else:p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(payload)
    write(receipt_path,{'status':'PASS_LOCAL_ARCHIVE_AND_ALL_HISTORY_FILE_SHAS',
        'archive':str(local),'archive_sha256':item['sha256'],'files':index['files'],
        'selected_node':index['selected_node'],'source':'Original completed40k training and original validation hook; no new selection'})
    print(json.dumps({'status':'RECOVERED_BASELINE_TRAINING_HISTORY','files':len(index['files'])}))


if __name__=='__main__':main()
