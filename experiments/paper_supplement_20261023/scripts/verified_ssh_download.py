"""Hash-verified resumable binary SSH stream; avoids per-block SFTP round trips."""
import hashlib
import json
import os
from pathlib import Path
import shlex
import time
from training_remote_control import ROOT,REMOTE,execute

def sha(path,limit=None):
    h=hashlib.sha256();remaining=limit
    with Path(path).open('rb') as f:
        while remaining is None or remaining>0:
            b=f.read(8*1024**2 if remaining is None else min(8*1024**2,remaining))
            if not b:break
            h.update(b)
            if remaining is not None:remaining-=len(b)
    return h.hexdigest()

def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2),encoding='utf-8');temp.replace(path)

def download(client,remote,destination,expected_sha256,expected_bytes,progress_path=None):
    destination=Path(destination).resolve();destination.relative_to(ROOT.resolve())
    assert remote.startswith(REMOTE+'/') and '/..' not in remote
    expected_bytes=int(expected_bytes)
    if destination.exists():
        assert destination.stat().st_size==expected_bytes and sha(destination)==expected_sha256
        return {'status':'PASS_ALREADY_LOCAL','path':str(destination),'sha256':expected_sha256,'bytes':expected_bytes}
    destination.parent.mkdir(parents=True,exist_ok=True)
    part=destination.with_suffix(destination.suffix+'.part');identity=part.with_suffix(part.suffix+'.source.json')
    offset=part.stat().st_size if part.exists() else 0
    assert offset<=expected_bytes
    ident={'remote':remote,'sha256':expected_sha256,'bytes':expected_bytes}
    if identity.exists():assert json.loads(identity.read_text())==ident,'Partial file belongs to a different immutable source'
    if offset:
        code="import hashlib,sys;f=open(sys.argv[1],'rb');n=int(sys.argv[2]);h=hashlib.sha256();\nwhile n:\n b=f.read(min(n,8388608));assert b;n-=len(b);h.update(b)\nprint(h.hexdigest())"
        command=shlex.join(['/root/miniconda3/bin/python','-c',code,remote,str(offset)])
        response=execute(client,command,timeout=120);assert response['exit_code']==0,response
        assert response['stdout'].strip()==sha(part),'Partial prefix differs from the pinned server source'
    write(identity,ident)
    code="import os,sys,shutil; p=sys.argv[1];n=int(sys.argv[2]);size=int(sys.argv[3]);assert os.stat(p).st_size==size;f=open(p,'rb');f.seek(n);shutil.copyfileobj(f,sys.stdout.buffer,1048576);sys.stdout.buffer.flush()"
    command=shlex.join(['/root/miniconda3/bin/python','-u','-c',code,remote,str(offset),str(expected_bytes)])
    channel=client.get_transport().open_session(window_size=128*1024**2,max_packet_size=1024**2)
    channel.settimeout(60);channel.exec_command(command)
    began=time.monotonic();last=0.;total=offset
    try:
        with part.open('ab',buffering=1024**2) as f:
            while True:
                data=channel.recv(1024**2)
                if not data:break
                f.write(data);total+=len(data)
                assert total<=expected_bytes
                if time.monotonic()-last>=10:
                    value={'stage':'DOWNLOADING','file':destination.name,'bytes':total,'total_bytes':expected_bytes,
                           'resumed_bytes':offset,'MiB_per_second':(total-offset)/1024**2/max(time.monotonic()-began,.001),'updated_unix':time.time()}
                    if progress_path:write(progress_path,value)
                    print(json.dumps(value),flush=True);last=time.monotonic()
            f.flush();os.fsync(f.fileno())
        code=channel.recv_exit_status()
        assert code==0,channel.makefile_stderr('rb').read().decode('utf-8','replace')
    finally:channel.close()
    assert total==expected_bytes and part.stat().st_size==expected_bytes
    assert sha(part)==expected_sha256,'Full on-disk SHA differs; preserve partial and source identity'
    part.replace(destination)
    result={'status':'PASS_LOCAL_FULL_SHA','path':str(destination),'sha256':expected_sha256,'bytes':expected_bytes,
            'resumed_bytes':offset,'seconds':time.monotonic()-began,'transport':'Binary SSH stream,128MiB window,1MiB packets'}
    write(destination.with_suffix(destination.suffix+'.verification.json'),result)
    if progress_path:write(progress_path,result)
    return result
