"""Best-effort local small-evidence sync; shutdown never depends on this watcher."""
import json
import re
from pathlib import Path
import time
from remote_ops import connect

BASE='/root/autodl-tmp/nativepyramid_sp_v3_20261016'
HERE=Path(__file__).resolve().parent/'observed_server_attempt03'
SOURCE_SHA='d3032fc8b20f30a058adbcf3e6bc5a3485b83adafe9b7cdce0682a2d55d7f627'
FILES=('job_status.json','job_result.json','receipts/shutdown_request.json','receipts/shutdown_wrapper_preparation.json',
       'preflight_attempt03/preflight.json','preflight_attempt03/diagnostic_progress.json','preflight_attempt03/unit_tests.json',
       'preflight_attempt03/precision_micro_parity.json','preflight_attempt03/input_pipeline.json',
       'preflight_attempt03/real_crop_learning.json','preflight_attempt03/full_resolution_smoke.json',
       'run_seed42_launch02/progress.json','run_seed42_launch02/result.json','run_seed42_launch02/endpoint_lock.json',
       'run_seed42_launch02/failure.json','run_seed42_launch02/test_result.json','run_seed42_launch02/test_failure.json')


def save(relative,raw):
    path=HERE/relative;path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.writing');temp.write_bytes(raw);temp.replace(path)


if __name__=='__main__':
    HERE.mkdir(exist_ok=True)
    client=sftp=None;seen={};last_event=None;shutdown_seen=False;errors=0;current_started_at=None
    pending=set();last_discovery=0
    while True:
        try:
            if client is None:
                client=connect();sftp=client.open_sftp()
            if time.monotonic()-last_discovery>=30:
                for directory,pattern in (('run_seed42_launch02/probes',r'train_\d{5}\.json'),
                                          ('run_seed42_launch02',r'val_\d{5}\.json')):
                    try:names=sftp.listdir(BASE+'/'+directory)
                    except OSError:names=[]
                    pending.update(directory+'/'+name for name in names if re.fullmatch(pattern,name)
                                   and directory+'/'+name not in seen)
                last_discovery=time.monotonic()
            for relative in (*FILES,*pending):
                remote=BASE+'/'+relative
                try:stat=sftp.stat(remote)
                except OSError:continue
                fingerprint=(stat.st_mtime,stat.st_size)
                if seen.get(relative)==fingerprint:continue
                with sftp.open(remote,'rb') as stream:raw=stream.read()
                json.loads(raw)
                save(relative,raw);seen[relative]=fingerprint
                pending.discard(relative)
                if relative=='job_status.json':
                    status=json.loads(raw)
                    if status.get('source_sha256')!=SOURCE_SHA or status.get('controller_version')!=3:
                        continue
                    current_started_at=status.get('started_at')
                    phase=status.get('status');progress=status.get('formal_progress') or status.get('diagnostic_progress') or {}
                    update=progress.get('executed_updates',progress.get('completed_updates',0))
                    event=(phase,int(update)//(1000 if phase=='FORMAL_TRAINING' else 50))
                    if event!=last_event:
                        print(json.dumps({'phase':phase,'update':update,'outcome':status.get('outcome')}),flush=True)
                        last_event=event
                    shutdown_seen=shutdown_seen or phase=='SHUTDOWN_REQUESTED'
                if relative=='receipts/shutdown_request.json':
                    request=json.loads(raw)
                    shutdown_seen=shutdown_seen or bool(current_started_at and request.get('requested_at','')>=current_started_at)
            if shutdown_seen:
                for relative in ('shutdown.log','controller_attempt03.log','preflight_attempt03.log','training_attempt03.log','test_attempt03.log'):
                    try:
                        with sftp.open(BASE+'/'+relative,'rb') as stream:raw=stream.read()
                        save(relative,raw)
                    except OSError:pass
            errors=0
            time.sleep(1)
        except BaseException as error:
            if isinstance(error,KeyboardInterrupt):raise
            errors+=1
            try:
                if sftp is not None:sftp.close()
                if client is not None:client.close()
            except Exception:pass
            client=sftp=None
            record={'status':'CONNECTION_CLOSED_AFTER_SHUTDOWN_REQUEST' if shutdown_seen else 'CONNECTION_RETRY',
                    'shutdown_request_observed':shutdown_seen,'error_type':type(error).__name__,
                    'error':str(error),'consecutive_errors':errors,'unix_time':time.time()}
            save('watcher_status.json',(json.dumps(record,indent=2)+'\n').encode())
            if shutdown_seen and errors>=3:
                print(json.dumps(record),flush=True)
                break
            time.sleep(10)
