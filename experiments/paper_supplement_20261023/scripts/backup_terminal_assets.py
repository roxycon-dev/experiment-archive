"""Event-based local artifact backup; no model calls, no shutdown, and no training changes."""
import argparse
import json
import msvcrt
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time
import zipfile
import paramiko
from training_remote_control import ROOT,REMOTE,LAN,connect,execute
from verified_ssh_download import download,sha,write
from verified_parallel_download import download as download_parallel

def root_manifest(client):
    response=execute(client,'/root/miniconda3/bin/python '+shlex.quote(REMOTE+'/scripts/prepare_terminal_backup_manifest.py'),timeout=180)
    assert response['exit_code']==0,response
    return json.loads(response['stdout'])

def baseline_manifest(client):
    recovered=ROOT/'evidence/recovered_original_baseline_identity.json'
    if recovered.exists():
        identity=json.loads(recovered.read_text(encoding='utf-8'))
        assert identity['status']=='PASS_RECOVERED_HISTORICAL_BASELINE_IDENTITY_AND_LOCAL_BACKUP'
        return dict(kind='baseline_original',checkpoint_path=identity['local_checkpoint'],
                    checkpoint_sha256=identity['source_checkpoint_sha256'],selected_node=32000,
                    config_sha256=identity['inference_config_sha256'],
                    selected_by='Historical original validation hook, recovered without new selection')
    corrected=(ROOT/'plan/baseline_reconstruction_amendment.json').exists()
    path=LAN+'/evaluation_workspace/paper_supplement_20261023/eval_jobs/'+('baseline_v2_seed42.json' if corrected else 'baseline_seed42.json')
    code="from pathlib import Path;import json;p=Path("+repr(path)+");print(json.dumps(json.loads(p.read_text()) if p.exists() else None))"
    response=execute(client,'/usr/bin/python3 -c '+shlex.quote(code),'59');assert response['exit_code']==0,response
    return json.loads(response['stdout'])

def stage_baseline(client,job):
    folder=REMOTE+'/model_recovery';items=[]
    corrected=job['kind']=='baseline_v2';prefix='baseline_v2' if corrected else 'baseline';base='baseline_v2' if corrected else 'baseline_v1'
    for name,source,digest in [(prefix+'_best_u'+str(job['selected_node']),job['checkpoint_path'],job['checkpoint_sha256']),
                               (prefix+'_endpoint_u40000',LAN+'/'+base+'/runs/baseline_seed42/iter_40000.pth',job['endpoint_checkpoint_sha256'])]:
        assert source.startswith(LAN+'/'+base+'/runs/baseline_seed42/')
        target=folder+'/'+name+'.pth'
        code=r'''
from pathlib import Path
import hashlib,json,shutil,subprocess
src=SOURCE_VALUE;dst=Path(TARGET_VALUE);digest=DIGEST_VALUE
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
dst.parent.mkdir(exist_ok=True)
if not dst.exists():
 assert shutil.disk_usage(dst.parent).free>2*1024**3+5*1024**3,'Retain disk headroom before staging baseline'
 command=['scp','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-i','/root/.ssh/codex_supp_to59_20261023','-P','40560','studentuser001@127.0.0.1:'+src,str(dst)+'.part']
 subprocess.run(command,check=True);p=Path(str(dst)+'.part');assert sha(p)==digest;p.replace(dst)
assert sha(dst)==digest
print(json.dumps({'remote':str(dst),'sha256':digest,'bytes':dst.stat().st_size}))
'''.replace('SOURCE_VALUE',repr(source)).replace('TARGET_VALUE',repr(target)).replace('DIGEST_VALUE',repr(digest))
        response=execute(client,'/root/miniconda3/bin/python -c '+shlex.quote(code),timeout=900);assert response['exit_code']==0,response
        items.append(dict(json.loads(response['stdout']),name=name,kind='baseline_checkpoint'))
    write(ROOT/('evidence/terminal_backups/baseline_v2_selection.json' if corrected else 'evidence/terminal_backups/baseline_selection.json'),job)
    return items

def extract_verified_archive(path):
    destination=ROOT/'recovered_training_provenance'
    with zipfile.ZipFile(path) as z:
        manifest=json.loads(z.read('backup_manifest.json'))
        for row in manifest['files']:
            target=(destination/row['path']).resolve();target.relative_to(destination.resolve())
            payload=z.read(row['path']);import hashlib
            assert len(payload)==row['bytes'] and hashlib.sha256(payload).hexdigest()==row['sha256']
            if target.exists():assert sha(target)==row['sha256']
            else:target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(payload)

def once():
    receipts=ROOT/'evidence/terminal_backups';receipts.mkdir(parents=True,exist_ok=True)
    with connect() as client:
        with client.open_sftp() as s:s.put(str(ROOT/'scripts/prepare_terminal_backup_manifest.py'),REMOTE+'/scripts/prepare_terminal_backup_manifest.py')
        manifest=root_manifest(client);job=baseline_manifest(client)
        items=manifest['items']
        if job and job['kind']=='baseline_original':
            assert sha(Path(job['checkpoint_path']))==job['checkpoint_sha256']
            write(receipts/'baseline_original_selection.json',job)
        elif job:
            old=receipts/('baseline_v2_staging.json' if job['kind']=='baseline_v2' else 'baseline_staging.json')
            if old.exists():items+=json.loads(old.read_text())
            else:
                staged=stage_baseline(client,job);write(old,staged);items+=staged
        complete=[]
        for item in items:
            receipt=receipts/(item['name']+'.json')
            dest=ROOT/'models'/(item['name']+('.pth' if item['kind'] in ('full_checkpoint','EMA_export','baseline_checkpoint') else '.zip'))
            if receipt.exists():
                previous=json.loads(receipt.read_text());assert previous['sha256']==item['sha256']
                assert dest.exists() and dest.stat().st_size==item['bytes'] and dest.stat().st_mtime_ns==previous['local_mtime_ns']
                complete.append(item['name']);continue
            assert shutil.disk_usage(ROOT).free>item['bytes']+3*1024**3
            transfer=download_parallel if item['bytes']>=32*1024**2 else download
            result=transfer(client,item['remote'],dest,item['sha256'],item['bytes'],ROOT/'results/model_backup_transfer.json')
            if dest.suffix=='.zip':extract_verified_archive(dest)
            result.update(kind=item['kind'],source=item['remote'],local_mtime_ns=dest.stat().st_mtime_ns,campaign_complete=False)
            write(receipt,result);complete.append(item['name'])
        all_done=len(manifest['new_completed_runs'])==4 and len(manifest['new_exported_runs'])==4 and job is not None and len(complete)==len(items)
        value={'stage':'ALL_TERMINAL_ASSETS_BACKED_UP' if all_done else 'WAITING_FIXED_TERMINAL_ARTIFACTS',
               'backed_up':complete,'new_completed_runs':manifest['new_completed_runs'],'new_exported_runs':manifest['new_exported_runs'],
               'baseline40k_available':job is not None and job['kind']!='baseline_original',
               'recovered_original_baseline_available':job is not None and job['kind']=='baseline_original',
               'all_terminal_backups_complete':all_done,'overall_campaign_complete':False,
               'updated_unix':time.time(),'no_shutdown':True}
        write(ROOT/'results/terminal_backup_status.json',value);return value

def main():
    p=argparse.ArgumentParser();p.add_argument('--watch',action='store_true');p.add_argument('--max-hours',type=float,default=30);args=p.parse_args()
    lock_path=ROOT/'logs/terminal_backup.lock';lock_path.parent.mkdir(exist_ok=True)
    with lock_path.open('a+b') as lock:
        if lock.tell()==0:lock.write(b'0');lock.flush()
        lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        started=time.monotonic()
        while True:
            try:
                value=once();print(json.dumps(value),flush=True)
            except (paramiko.SSHException,OSError,TimeoutError) as e:
                write(ROOT/'results/terminal_backup_status.json',{'stage':'NETWORK_RETRY','error':repr(e),'overall_campaign_complete':False,'updated_unix':time.time()})
                if not args.watch:raise
            except BaseException as e:
                write(ROOT/'results/terminal_backup_status.json',{'stage':'TECHNICAL_PAUSE','error':repr(e),'overall_campaign_complete':False,'updated_unix':time.time()});raise
            else:
                if value['all_terminal_backups_complete'] or not args.watch:break
            if time.monotonic()-started>args.max_hours*3600:
                write(ROOT/'results/terminal_backup_status.json',{'stage':'WATCH_WINDOW_ENDED_INCOMPLETE','overall_campaign_complete':False,'updated_unix':time.time()});break
            time.sleep(180)

if __name__=='__main__':main()
