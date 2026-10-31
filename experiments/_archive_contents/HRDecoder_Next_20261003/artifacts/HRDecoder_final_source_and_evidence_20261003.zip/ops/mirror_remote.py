"""Continuous verified local backup; derived G RAM arrays evict only after acknowledgement.

Password is never used. Original assets and original cache files are never
deleted; only new, completed, backed-up G arrays may be released after the
agent writes the explicit allow_g_evict marker. Models remain server-side.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import threading
import time
import paramiko

ROOT=Path(__file__).resolve().parents[1]
REMOTE='/root/autodl-tmp/HRDecoder_Next_20261003'
LOCAL=ROOT/'server_results'
KEY=Path.home()/'.ssh/codex_hrdecoder_next_20261003'

def connect():
    c=paramiko.SSHClient();c.load_host_keys(str(ROOT/'ops/known_hosts'))
    c.connect('connect.nmb2.seetacloud.com',port=36431,username='root',key_filename=str(KEY),
              timeout=25,allow_agent=False,look_for_keys=False)
    c.get_transport().set_keepalive(30)
    return c

def remote_exec(c,cmd,timeout=180):
    _,out,err=c.exec_command(cmd,timeout=timeout)
    stdout=out.read().decode('utf-8','replace');stderr=err.read().decode('utf-8','replace')
    status=out.channel.recv_exit_status()
    if status:raise RuntimeError(str(status)+' '+stderr[-3000:]+stdout[-3000:])
    return stdout

def write_json(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp')
    tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(p)

def file_sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()

INVENTORY=r'''
import pathlib,json,hashlib,os,time,subprocess
r=pathlib.Path('/root/autodl-tmp/HRDecoder_Next_20261003');w=r/'work'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()
files=[]
for base in [r/'configs',r/'reports',w]:
 if not base.exists():continue
 for p in base.rglob('*'):
  if not p.is_file() or p.suffix not in ['.json','.log','.txt','.md','.png','.csv','.html']:continue
  if 'remote_responses' in p.parts or 'mirror_manifests' in p.parts:continue
  rel=p.relative_to(r).as_posix();st=p.stat()
  files.append(dict(path=rel,size=st.st_size,mtime=st.st_mtime_ns,kind='metadata'))
for p in w.glob('M_seed*/M*/best.pth'):
 st=p.stat();files.append(dict(path=p.relative_to(r).as_posix(),size=st.st_size,mtime=st.st_mtime_ns,kind='weight',sha256=sha(p)))
for p in w.glob('M_seed*/M*/last.pth'):
 st=p.stat();files.append(dict(path=p.relative_to(r).as_posix(),size=st.st_size,mtime=st.st_mtime_ns,kind='resume_weight',sha256=sha(p)))
caches=[p.parent for p in (w/'G').glob('*/index.json')]
for p in w.glob('M_seed*/M*/history.json'):
 if not (p.parent.parent/'selection.json').exists():continue
 history=json.loads(p.read_text());rows=[]
 for node in history:
  for mode in ['solo','fixed_half']:rows.append((-node[mode]['mIoU_lesion4'],node['update'],mode!='solo'))
 if rows:caches.append(p.parent/('val_u'+str(sorted(rows)[0][1])))
for cache in caches:
 for p in cache.glob('*.json'):
  rel=p.relative_to(r).as_posix();st=p.stat()
  if not any(x['path']==rel for x in files):files.append(dict(path=rel,size=st.st_size,mtime=st.st_mtime_ns,kind='metadata'))
 paths=list(cache.glob('*.npz'))
 if not paths:continue
 rel=cache.relative_to(r).as_posix()
 saved=r/'reports/mirror_manifests'/(hashlib.sha256(rel.encode()).hexdigest()+'.json')
 if saved.exists():data=json.loads(saved.read_text())
 else:
  data=[dict(path=p.relative_to(r).as_posix(),size=p.stat().st_size,sha256=sha(p),kind='probability',cache=rel) for p in paths]
  saved.parent.mkdir(parents=True,exist_ok=True);saved.write_text(json.dumps(data))
 files.extend(data)
stage=r/'reports/server_stage.json'
print(json.dumps(dict(files=files,stage=json.loads(stage.read_text()) if stage.exists() else None,
 time=time.time(),gpu=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used','--format=csv,noheader'],text=True).strip())))
'''

def once(c,manifest):
    write_json(ROOT/'reports/mirror_phase.json',dict(time=time.time(),phase='inventory'))
    inventory=json.loads(remote_exec(c,"/root/miniconda3/bin/python - <<'PY'\n"+INVENTORY+"\nPY",300))
    write_json(ROOT/'reports/monitor_health.json',dict(time=time.time(),remote_time=inventory['time'],
        stage=inventory['stage'],gpu=inventory['gpu'],status='connected'))
    files=inventory['files'];metadata=[x for x in files if x['kind']=='metadata']
    write_json(ROOT/'reports/mirror_phase.json',dict(time=time.time(),phase='metadata',files=len(metadata)))
    # Metadata and recovery checkpoints take precedence over diagnostic arrays.
    def fetch(item):
        rel=item['path'];target=LOCAL/rel;old=manifest.get(rel,{})
        if target.exists():
            if item['kind']=='metadata' and old.get('mtime')==item.get('mtime') and old.get('size')==item['size']:return
            if item.get('sha256') and old.get('sha256')==item['sha256'] and target.stat().st_size==item['size']:return
            if item.get('sha256') and target.stat().st_size==item['size'] and file_sha(target)==item['sha256']:
                manifest[rel]=dict(size=target.stat().st_size,sha256=item['sha256'],
                    mtime=item.get('mtime'),verified_at=time.time())
                return
        target.parent.mkdir(parents=True,exist_ok=True);temp=target.with_suffix(target.suffix+'.download')
        with c.open_sftp() as sftp:
            try:sftp.get(REMOTE+'/'+rel,str(temp),max_concurrent_prefetch_requests=64)
            except TypeError:sftp.get(REMOTE+'/'+rel,str(temp))
        h=file_sha(temp)
        if item.get('sha256') and h!=item['sha256']:
            temp.unlink();raise ValueError('Remote file changed or SHA mismatch: '+rel)
        temp.replace(target)
        manifest[rel]=dict(size=target.stat().st_size,sha256=h,mtime=item.get('mtime'),verified_at=time.time())
    with ThreadPoolExecutor(max_workers=4) as pool:
        for result in pool.map(fetch,metadata):pass
    for kind in ['weight','resume_weight']:
        if (ROOT/'reports/critical_backup_active.json').exists():continue
        chosen=[x for x in files if x['kind']==kind]
        with ThreadPoolExecutor(max_workers=4) as pool:
            for result in pool.map(fetch,chosen):pass
        write_json(ROOT/'reports/mirror_manifest.json',manifest)
    groups={}
    for item in files:
        if item['kind']=='probability':groups.setdefault(item['cache'],[]).append(item)
    # Archive the almost-backed-up cold assets first, then free owned RAM caches.
    # M arrays are stable only once that paired seed has completed.
    def priority(group):
        cache,items=group
        existing=sum((LOCAL/x['path']).stat().st_size for x in items if (LOCAL/x['path']).exists())
        total=sum(x['size'] for x in items)
        return (0 if cache.startswith('work/M_seed') else 1,-existing/max(total,1),cache)
    for cache,items in sorted(groups.items(),key=priority):
        write_json(ROOT/'reports/mirror_phase.json',dict(time=time.time(),phase='probabilities',
            cache=cache,files=len(items),bytes=sum(x['size'] for x in items)))
        with ThreadPoolExecutor(max_workers=4) as pool:
            for result in pool.map(fetch,items):pass
        write_json(ROOT/'reports/mirror_manifest.json',manifest)
        print('CACHE_BACKED_UP',cache,flush=True)
        if (ROOT/'reports/allow_g_evict.json').exists():
            from evict_verified_g import main as release_verified
            release_verified()
    # Allowing eviction is a separate, explicit decision after G analysis.
    if (ROOT/'reports/allow_g_evict.json').exists():
        safety=json.loads(remote_exec(c,"/root/miniconda3/bin/python - <<'PY'\nfrom pathlib import Path\nimport json\nr=Path('"+REMOTE+"');p=r/'reports/G_deployment_candidates.json'\nj=json.loads(p.read_text()) if p.exists() else {}\nprint(json.dumps(dict(analysis_complete=(r/'reports/G_analysis_complete.json').exists(),best=j.get('best',{}).get('key'))))\nPY"))
        if not safety['analysis_complete']:return
        groups={}
        for item in files:
            if item['kind']=='probability' and item.get('cache','').startswith('work/G/'):
                if item['cache'].split('/')[-1] not in ['champion','U0_native_whole',safety['best']] and not item['cache'].split('/')[-1].startswith('U0_highres'):
                    groups.setdefault(item['cache'],[]).append(item)
        for cache,items in groups.items():
            if len(items)!=76 or any(manifest.get(x['path'],{}).get('sha256')!=x['sha256'] for x in items):continue
            payload=json.dumps(items)
            code="""import json,pathlib,hashlib
r=pathlib.Path('/root/autodl-tmp/HRDecoder_Next_20261003');items=json.loads(PAYLOAD)
for item in items:
 p=r/item['path'];assert (r/'work').resolve() in p.resolve().parents or (r/'cold_storage').resolve() in p.resolve().parents
 assert not p.is_symlink()
 if not p.exists():continue
 h=hashlib.sha256(p.read_bytes()).hexdigest();assert h==item['sha256'];p.unlink()
cache=r/items[0]['cache'];p=cache/'index.json';j=json.loads(p.read_text());j['probabilities_backed_up_then_evicted']=True;p.write_text(json.dumps(j,indent=2))
print('Backed-up new G arrays released',str(cache))
""".replace('PAYLOAD',repr(payload))
            print(remote_exec(c,"/root/miniconda3/bin/python - <<'PY'\n"+code+"\nPY"),flush=True)
    write_json(ROOT/'reports/mirror_status.json',dict(status='PASS',time=time.time(),files=len(manifest),
        bytes_verified=sum(x['size'] for x in manifest.values()),stage=inventory['stage']))
    print('MIRROR_PASS',len(manifest),round(sum(x['size'] for x in manifest.values())/1024**3,2),'GiB',flush=True)
    write_json(ROOT/'reports/mirror_phase.json',dict(time=time.time(),phase='pass_complete'))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--once',action='store_true');args=ap.parse_args()
    path=ROOT/'reports/mirror_manifest.json';manifest=json.loads(path.read_text(encoding='utf-8-sig')) if path.exists() else {}
    while True:
        c=None
        try:
            c=connect();once(c,manifest)
        except Exception as exc:
            write_json(ROOT/'reports/mirror_manifest.json',manifest)
            write_json(ROOT/'reports/mirror_status.json',dict(status='needs_attention',time=time.time(),error=str(exc)))
            print('MIRROR_RETRY',type(exc).__name__,str(exc),flush=True)
        finally:
            if c:c.close()
        if args.once or (ROOT/'reports/stop_mirror.json').exists():break
        time.sleep(60)

if __name__=='__main__':main()
