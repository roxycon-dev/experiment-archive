"""Bounded remote disk, immutable endpoint requests, SHA-verified local custody."""
import time,shutil
from pathlib import Path
from qlib.common import read_json,write_json,sha,utc_now,digest
ROOT=Path(__file__).resolve().parent
GiB=1024**3

def check_storage_plan():
    plan=read_json(ROOT/'reports/storage_plan.json')
    assert plan['user_authorized_local_cold_archive'] and plan['initial_local_free_bytes']>=100*GiB
    assert plan['preserve_full_precision_and_all_required_endpoints']
    assert shutil.disk_usage(ROOT).free>=12*GiB,'Active server needs at least12GiB after input preparation'
    return plan

def publish_endpoint(runpath,update):
    assert update in (20000,32000)
    runpath=Path(runpath);assert runpath.parent==ROOT/'runs'
    name=runpath.name;assert name in {f'{a}_seed{s}' for a in ('C','S','D','SD') for s in (42,2024)}
    metrics=read_json(runpath/f'val_u{update}/metrics.json');assert metrics['n_images']==76
    checkpoint=runpath/f'checkpoint_u{update}.pth';assert checkpoint.exists()
    files=[{'path':checkpoint.relative_to(ROOT).as_posix(),'bytes':checkpoint.stat().st_size,'sha256':sha(checkpoint)}]
    assert len(metrics['probability_files'])==76
    for p in metrics['probability_files']:
        path=runpath/f'val_u{update}/{p["sid"]}.npz'
        assert path.stat().st_size==p['bytes']
        files.append({'path':path.relative_to(ROOT).as_posix(),'bytes':p['bytes'],'sha256':p['sha256']})
    request={'run':name,'update':update,'files':files,'metrics_sha256':sha(runpath/f'val_u{update}/metrics.json'),
             'identity_sha256':sha(runpath/'identity.json'),'time':utc_now(),'mutable_last_checkpoint_excluded':True}
    request['request_sha256']=digest(request)
    path=ROOT/f'reports/storage_requests/{name}_u{update}.json'
    if path.exists():assert read_json(path)['files']==files,'Immutable endpoint changed after publishing'
    else:write_json(path,request)

def wait_for_space(minimum=8*GiB,runpath=None,completed=None):
    started=time.monotonic()
    while shutil.disk_usage(ROOT).free<minimum:
        requests=list((ROOT/'reports/storage_requests').glob('*.json'))
        assert requests,'Insufficient disk with no completed endpoint eligible for verified offload'
        if runpath:
            write_json(Path(runpath)/'status.json',{'stage':'WAITING_FOR_VERIFIED_OFFLOAD','completed':completed,
                'minimum_free_bytes':minimum,'free_disk_bytes':shutil.disk_usage(ROOT).free,'updated_at':utc_now()})
        if time.monotonic()-started>3600:raise OSError('Verified offload did not release the required space within1h; preserve current checkpoint and diagnose receiver')
        time.sleep(10)

def publish_parent_probabilities(runpath):
    runpath=Path(runpath);assert runpath==ROOT/'runs/PARENT_F12_seed2024'
    assert read_json(runpath/'status.json')['stage']=='COMPLETE'
    metrics=read_json(runpath/'val_u12000/metrics.json');assert len(metrics['probability_files'])==76
    files=[]
    for row in metrics['probability_files']:
        p=runpath/f'val_u12000/{row["sid"]}.npz';assert p.stat().st_size==row['bytes']
        files.append({'path':p.relative_to(ROOT).as_posix(),'bytes':row['bytes'],'sha256':row['sha256']})
    request={'run':runpath.name,'update':12000,'files':files,'metrics_sha256':sha(runpath/'val_u12000/metrics.json'),
             'identity_sha256':sha(runpath/'identity.json'),'time':utc_now(),'protected_parent_checkpoint_excluded':True}
    request['request_sha256']=digest(request);write_json(ROOT/'reports/storage_requests/PARENT_F12_seed2024_u12000.json',request)

def checkpoint_sha(relative):
    p=ROOT/relative
    if p.exists():return sha(p)
    receipts=list((ROOT/'reports/storage_offload').glob('*.json'))
    for receiptfile in receipts:
        receipt=read_json(receiptfile)
        if receipt.get('stage')!='LOCAL_SHA_VERIFIED_REMOTE_RELEASED':continue
        for item in receipt['files']:
            if item['path']==relative:return item['sha256']
    raise FileNotFoundError('No endpoint or verified custody receipt: '+relative)

def restore_selection(lock):
    missing=[row for row in lock['checkpoints'].values() if not (ROOT/row['path']).exists()]
    if not missing:return
    request={'lock_sha256':lock['lock_sha256'],'files':missing,'stage':'RESTORE_SELECTED_ONLY','time':utc_now()}
    write_json(ROOT/'reports/selected_restore_request.json',request)
    started=time.monotonic()
    while not all((ROOT/row['path']).exists() for row in missing):
        if time.monotonic()-started>3600:raise TimeoutError('Selected frozen weights not restored within1h; no test was started')
        time.sleep(10)
    for row in missing:assert sha(ROOT/row['path'])==row['sha256']
    write_json(ROOT/'reports/selected_restore_complete.json',{'stage':'SHA_VERIFIED','lock_sha256':lock['lock_sha256'],'time':utc_now()})
