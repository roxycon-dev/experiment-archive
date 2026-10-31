"""Four independent authenticated streams for immutable large artifacts, with verified range resume."""
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib
import json
import os
from pathlib import Path
import shlex
import threading
import time
from training_remote_control import ROOT,REMOTE,connect,execute
from verified_ssh_download import sha,write

def download(client,remote,destination,expected_sha256,expected_bytes,progress_path=None):
    destination=Path(destination).resolve();destination.relative_to(ROOT.resolve());expected_bytes=int(expected_bytes)
    assert remote.startswith(REMOTE+'/') and '/..' not in remote and expected_bytes>0
    if destination.exists():
        assert destination.stat().st_size==expected_bytes and sha(destination)==expected_sha256
        return {'status':'PASS_ALREADY_LOCAL','path':str(destination),'sha256':expected_sha256,'bytes':expected_bytes}
    chunks=destination.with_suffix(destination.suffix+'.chunks');chunks.mkdir(parents=True,exist_ok=True)
    layout_path=chunks/'layout.json';streams=min(4,expected_bytes)
    if layout_path.exists():
        layout=json.loads(layout_path.read_text());assert layout['source']==remote and layout['sha256']==expected_sha256 and layout['bytes']==expected_bytes
    else:
        code=r'''
import hashlib,json,os,sys
path=sys.argv[1];size=int(sys.argv[2]);count=int(sys.argv[3]);expected=sys.argv[4]
assert os.stat(path).st_size==size
h=hashlib.sha256();rows=[];offset=0
with open(path,'rb') as f:
 for i in range(count):
  n=size//count if i<count-1 else size-offset;c=hashlib.sha256();remaining=n
  while remaining:
   b=f.read(min(8388608,remaining));assert b;remaining-=len(b);h.update(b);c.update(b)
  rows.append({'index':i,'offset':offset,'bytes':n,'sha256':c.hexdigest()});offset+=n
assert h.hexdigest()==expected
print(json.dumps({'source':path,'sha256':expected,'bytes':size,'ranges':rows}))
'''
        command=shlex.join(['/root/miniconda3/bin/python','-c',code,remote,str(expected_bytes),str(streams),expected_sha256])
        response=execute(client,command,timeout=180);assert response['exit_code']==0,response
        layout=json.loads(response['stdout']);write(layout_path,layout)
    began=time.monotonic();mutex=threading.Lock();cancel=threading.Event();received={};last=[0.]
    for r in layout['ranges']:
        done=chunks/(str(r['index'])+'.bin');part=done.with_suffix('.bin.part')
        received[r['index']]=done.stat().st_size if done.exists() else (part.stat().st_size if part.exists() else 0)
    initial=sum(received.values())
    def progress(index,n):
        with mutex:
            received[index]=n;t=time.monotonic()
            if t-last[0]>=30:
                total=sum(received.values());value={'stage':'DOWNLOADING','file':destination.name,'parallel_streams':4,'bytes':total,
                    'total_bytes':expected_bytes,'resumed_bytes':initial,'MiB_per_second':(total-initial)/1024**2/max(t-began,.001),'updated_unix':time.time()}
                if progress_path:write(progress_path,value)
                print(json.dumps(value),flush=True);last[0]=t
    def worker(r):
        index=r['index'];done=chunks/(str(index)+'.bin');part=done.with_suffix('.bin.part')
        if done.exists():
            assert done.stat().st_size==r['bytes'] and sha(done)==r['sha256'];return
        offset=part.stat().st_size if part.exists() else 0;assert offset<=r['bytes']
        with connect() as c:
            if offset:
                code="import sys,hashlib;f=open(sys.argv[1],'rb');f.seek(int(sys.argv[2]));n=int(sys.argv[3]);h=hashlib.sha256();\nwhile n:\n b=f.read(min(n,8388608));assert b;n-=len(b);h.update(b)\nprint(h.hexdigest())"
                response=execute(c,shlex.join(['/root/miniconda3/bin/python','-c',code,remote,str(r['offset']),str(offset)]),timeout=120)
                assert response['exit_code']==0 and response['stdout'].strip()==sha(part),'Resumed range prefix differs from its pinned source'
            code="import os,sys;f=open(sys.argv[1],'rb');assert os.fstat(f.fileno()).st_size==int(sys.argv[4]);f.seek(int(sys.argv[2]));n=int(sys.argv[3]);\nwhile n:\n b=f.read(min(n,1048576));assert b;n-=len(b);sys.stdout.buffer.write(b);sys.stdout.buffer.flush()"
            ch=c.get_transport().open_session(window_size=128*1024**2,max_packet_size=1024**2);ch.settimeout(60)
            ch.exec_command(shlex.join(['/root/miniconda3/bin/python','-u','-c',code,remote,str(r['offset']+offset),str(r['bytes']-offset),str(expected_bytes)]))
            n=offset
            try:
                with part.open('ab',buffering=1024**2) as f:
                    while True:
                        if cancel.is_set():raise OSError('Another transfer range failed; preserve partials for verified resume')
                        b=ch.recv(1024**2)
                        if not b:break
                        f.write(b);n+=len(b);assert n<=r['bytes'];progress(index,n)
                    f.flush();os.fsync(f.fileno())
                assert ch.recv_exit_status()==0,ch.makefile_stderr('rb').read().decode('utf-8','replace')
            finally:ch.close()
        assert n==r['bytes'] and sha(part)==r['sha256'];part.replace(done);progress(index,n)
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(worker,r) for r in layout['ranges']]
        try:
            for f in as_completed(futures):f.result()
        except BaseException:
            cancel.set();raise
    final_part=destination.with_suffix(destination.suffix+'.part')
    with final_part.open('wb',buffering=8*1024**2) as out:
        for r in layout['ranges']:
            p=chunks/(str(r['index'])+'.bin');assert sha(p)==r['sha256']
            with p.open('rb') as f:
                while True:
                    data=f.read(8*1024**2)
                    if not data:break
                    out.write(data)
        out.flush();os.fsync(out.fileno())
    assert final_part.stat().st_size==expected_bytes and sha(final_part)==expected_sha256
    final_part.replace(destination)
    result={'status':'PASS_LOCAL_FULL_AND_RANGE_SHAS','path':str(destination),'sha256':expected_sha256,'bytes':expected_bytes,
            'resumed_bytes':initial,'seconds':time.monotonic()-began,'transport':'Four pinned-key independent SSH binary streams',
            'verified_range_parts_retained':str(chunks)}
    write(destination.with_suffix(destination.suffix+'.verification.json'),result)
    if progress_path:write(progress_path,result)
    return result
