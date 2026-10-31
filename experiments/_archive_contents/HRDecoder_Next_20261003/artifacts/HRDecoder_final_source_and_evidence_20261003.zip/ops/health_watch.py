"""Light status and an independent SSH backup channel for best/resume weights."""
from pathlib import Path
import json
import os
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parent))
from mirror_remote import ROOT,LOCAL,REMOTE,connect,remote_exec,file_sha,write_json

PROBE=r'''
import pathlib,json,time,hashlib,subprocess
r=pathlib.Path('/root/autodl-tmp/HRDecoder_Next_20261003');files=[];progress={}
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()
for base in [r/'configs',r/'reports']:
 for p in base.glob('*.json'):
  if p.stat().st_size<20*1024**2:files.append(dict(path=p.relative_to(r).as_posix(),kind='metadata'))
for p in (r/'work').glob('*.json'):files.append(dict(path=p.relative_to(r).as_posix(),kind='metadata'))
for p in (r/'work').glob('M_seed*'):
 for q in p.glob('*.json'):files.append(dict(path=q.relative_to(r).as_posix(),kind='metadata'))
 for arm in p.glob('M*'):
  for q in arm.glob('*.json'):files.append(dict(path=q.relative_to(r).as_posix(),kind='metadata'))
  q=arm/'progress.json';row=json.loads(q.read_text()) if q.exists() else dict(update=0)
  progress[arm.relative_to(r).as_posix()]=row
  for name in ['best.pth','last.pth']:
   q=arm/name
   if q.exists():files.append(dict(path=q.relative_to(r).as_posix(),kind='weight',name=name,
       update=row.get('update',0),sha256=sha(q),size=q.stat().st_size,
       terminal=(p/'selection.json').exists()))
stage=r/'reports/server_stage.json'
for item in files:
 if item['kind']=='metadata':
  stat=(r/item['path']).stat();item.update(size=stat.st_size,mtime=stat.st_mtime_ns)
print(json.dumps(dict(files=files,progress=progress,time=time.time(),stage=json.loads(stage.read_text()),
 memory_current=int(pathlib.Path('/sys/fs/cgroup/memory.current').read_text()),
 memory_events=pathlib.Path('/sys/fs/cgroup/memory.events').read_text(),
 disk=subprocess.check_output(['df','-h','/dev/shm','/root/autodl-tmp'],text=True),
 gpu=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used','--format=csv,noheader'],text=True).strip())))
'''

def main():
    ledger_path=ROOT/'reports/critical_backup_manifest.json'
    ledger=json.loads(ledger_path.read_text(encoding='utf-8-sig')) if ledger_path.exists() else {}
    write_json(ROOT/'reports/critical_backup_active.json',dict(pid=os.getpid(),time=time.time()))
    while not (ROOT/'reports/stop_mirror.json').exists():
        c=None
        try:
            c=connect();snapshot=json.loads(remote_exec(c,"/root/miniconda3/bin/python - <<'PY'\n"+PROBE+"\nPY",120))
            write_json(ROOT/'reports/monitor_health.json',dict(time=time.time(),remote_time=snapshot['time'],
                stage=snapshot['stage'],progress=snapshot['progress'],gpu=snapshot['gpu'],status='connected',
                memory_current=snapshot['memory_current'],memory_events=snapshot['memory_events'],disk=snapshot['disk']))
            for item in sorted(snapshot['files'],key=lambda x:x['kind']!='metadata'):
                rel=item['path'];target=LOCAL/rel;old=ledger.get(rel,{})
                if item['kind']=='metadata' and target.exists() and old.get('mtime')==item.get('mtime') and old.get('size')==item.get('size'):continue
                if item['kind']=='weight':
                    if item['name']=='last.pth' and not item.get('terminal') and item['update']<old.get('update',-1000)+1000:continue
                    if target.exists() and file_sha(target)==item['sha256']:continue
                target.parent.mkdir(parents=True,exist_ok=True)
                temp=target.with_suffix(target.suffix+'.critical_download')
                with c.open_sftp() as sftp:
                    try:sftp.get(REMOTE+'/'+rel,str(temp),max_concurrent_prefetch_requests=64)
                    except TypeError:sftp.get(REMOTE+'/'+rel,str(temp))
                h=file_sha(temp)
                if item.get('sha256') and item['sha256']!=h:
                    temp.unlink();print('Atomic weight advanced during backup; retry latest',rel,flush=True);continue
                temp.replace(target);ledger[rel]=dict(sha256=h,update=item.get('update'),
                    mtime=item.get('mtime'),size=item.get('size'),time=time.time())
                write_json(ledger_path,ledger)
                if item['kind']=='weight':print('CRITICAL_WEIGHT_VERIFIED',rel,item['update'],h,flush=True)
            write_json(ROOT/'reports/critical_backup_status.json',dict(status='PASS',time=time.time(),files=len(ledger)))
        except Exception as exc:
            write_json(ROOT/'reports/critical_backup_status.json',dict(status='needs_attention',time=time.time(),error=str(exc)))
            print('HEALTH_RETRY',type(exc).__name__,str(exc),flush=True)
        finally:
            if c:c.close()
        time.sleep(60)

if __name__=='__main__':main()
