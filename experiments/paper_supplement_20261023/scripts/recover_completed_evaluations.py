"""Recover stable completed masks through the authorized pinned SSH relay and verify every file."""
import hashlib
import json
from pathlib import Path
import shlex
import zipfile
from verified_ssh_download import download
from training_remote_control import connect,execute,ROOT,REMOTE,LAN

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()

LANROOT=LAN+'/evaluation_workspace/paper_supplement_20261023'
with connect() as client:
    code="from pathlib import Path;import json;r=Path("+repr(LANROOT)+");names=[];paths=list((r/'results/paired_seed59').glob('*/completion.json'))+list((r/'results/baseline_retrained59').glob('completion.json'));[(names.append('baseline_seed42' if p.parent.name=='baseline_retrained59' else p.parent.name)) for p in paths if (p.parent/'status.json').exists() and json.loads((p.parent/'status.json').read_text())['stage']=='COMPLETE'];print(json.dumps(names))"
    result=execute(client,'/usr/bin/python3 -c '+shlex.quote(code),'59');assert result['exit_code']==0,result
    names=json.loads(result['stdout']);recovered=[]
    with client.open_sftp() as s:
        s.put(str(ROOT/'scripts/pack_completed_evaluation59.py'),REMOTE+'/scripts/pack_completed_evaluation59.py')
    command=shlex.join(['scp','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-i','/root/.ssh/codex_supp_to59_20261023','-P','40560',REMOTE+'/scripts/pack_completed_evaluation59.py','studentuser001@127.0.0.1:'+LANROOT+'/scripts/pack_completed_evaluation59.py'])
    assert execute(client,command)['exit_code']==0
    assert execute(client,'mkdir -p '+shlex.quote(REMOTE+'/recovered_archives'))['exit_code']==0
    for name in names:
        local_receipt=ROOT/'evidence/recovered_evaluations'/(name+'.json')
        if local_receipt.exists():recovered.append({'name':name,'already_recovered':True});continue
        command='/usr/bin/python3 '+shlex.quote(LANROOT+'/scripts/pack_completed_evaluation59.py')+' --name '+shlex.quote(name)
        packed=execute(client,command,'59');assert packed['exit_code']==0,packed
        receipt=json.loads(packed['stdout']);remote_archive=REMOTE+'/recovered_archives/'+Path(receipt['archive']).name
        # A channel timeout may leave the immutable staging copy running to completion.
        # Reuse it only after its full SHA matches the source receipt; never overwrite it
        # just because the control connection timed out during a slow transfer.
        check="from pathlib import Path;import hashlib,json;p=Path("+repr(remote_archive)+");print(json.dumps(p.exists() and p.stat().st_size=="+str(receipt['archive_bytes'])+" and hashlib.sha256(p.read_bytes()).hexdigest()=="+repr(receipt['archive_sha256'])+"))"
        staged=execute(client,'/root/miniconda3/bin/python -c '+shlex.quote(check));assert staged['exit_code']==0,staged
        if not json.loads(staged['stdout']):
            command=shlex.join(['scp','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes',
                '-o','ConnectTimeout=15','-o','ServerAliveInterval=20','-o','ServerAliveCountMax=3',
                '-i','/root/.ssh/codex_supp_to59_20261023','-P','40560','studentuser001@127.0.0.1:'+receipt['archive'],remote_archive])
            copied=execute(client,command,timeout=600);assert copied['exit_code']==0,copied
        local=ROOT/'transfer'/Path(receipt['archive']).name;part=local.with_suffix('.zip.part')
        download(client,remote_archive,local,receipt['archive_sha256'],receipt['archive_bytes'],ROOT/'results/evaluation_recovery_transfer.json')
        with zipfile.ZipFile(local) as z:
            manifest_name=next(n for n in z.namelist() if n.endswith('/recovery_manifest.json'))
            index=json.loads(z.read(manifest_name));assert index['name']==name
            for r in index['files']:
                payload=z.read(r['path']);assert len(payload)==r['bytes'] and hashlib.sha256(payload).hexdigest()==r['sha256']
                destination=(ROOT/r['path']).resolve();destination.relative_to((ROOT/'results').resolve())
                if destination.exists():assert sha(destination)==r['sha256'],('Different existing result',destination)
                else:destination.parent.mkdir(parents=True,exist_ok=True);destination.write_bytes(payload)
            target=ROOT/manifest_name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(z.read(manifest_name))
        local_receipt.parent.mkdir(parents=True,exist_ok=True)
        receipt.update(local_archive=str(local),status='PASS_LOCAL_SHA_AND_ALL_ARCHIVED_FILE_SHAS',extracted_files=len(index['files']),
                       raw_probability_caches='Retained on2080Ti; not claimed locally backed up',model_checkpoint_backup='SEPARATE_PENDING')
        local_receipt.write_text(json.dumps(receipt,indent=2),encoding='utf-8');recovered.append(receipt)
        print(json.dumps({'recovered':name,'files':len(index['files']),'bytes':local.stat().st_size}),flush=True)
    print(json.dumps({'completed_runs_seen':names,'recovery':recovered}))
