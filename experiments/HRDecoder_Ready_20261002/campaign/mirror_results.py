"""Independent metadata and checked weight backups, without model calls."""
import argparse,getpass,hashlib,json,os,posixpath,shlex,stat,threading,time
from pathlib import Path
import paramiko
ROOT=Path(__file__).resolve().parents[1]
REMOTE='/root/autodl-tmp/HRDecoder_Ready_20261002'
OUT=ROOT/'server_results'

def write(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.part')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(path)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4194304),b''):h.update(b)
    return h.hexdigest()

def walk(sftp,rel):
    for a in sftp.listdir_attr(posixpath.join(REMOTE,rel)):
        p=posixpath.join(rel,a.filename)
        if stat.S_ISDIR(a.st_mode):yield from walk(sftp,p)
        elif stat.S_ISREG(a.st_mode):yield p,a

def main():
    global ROOT, REMOTE, OUT
    from monitor_once import REMOTE_CHECK
    ap=argparse.ArgumentParser();ap.add_argument('--host',required=True);ap.add_argument('--port',type=int,required=True)
    ap.add_argument('--metadata-once',action='store_true')
    ap.add_argument('--remote-root',default=REMOTE)
    ap.add_argument('--local-root',default=str(ROOT))
    args=ap.parse_args()
    REMOTE=args.remote_root
    ROOT=Path(args.local_root).resolve()
    OUT=ROOT/'server_results'
    REMOTE_CHECK=REMOTE_CHECK.replace('/root/autodl-tmp/HRDecoder_Ready_20261002',REMOTE)
    password=os.environ.pop('HRD_MIRROR_PASSWORD',None)
    if password is None:password=getpass.getpass('SSH password: ')
    OUT.mkdir(exist_ok=True);stop=threading.Event();ready=threading.Event();lock=threading.Lock();confirmed={}
    backup=dict(active=None,transferred_bytes=0,total_bytes=0,last_error=None,last_progress_epoch=None,verified_files=0)
    def connect():
        c=paramiko.SSHClient();c.load_system_host_keys();c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        c.connect(args.host,port=args.port,username='root',password=password,allow_agent=False,look_for_keys=False,timeout=20,banner_timeout=20,auth_timeout=20)
        c.get_transport().set_keepalive(15);s=c.open_sftp();s.get_channel().settimeout(20);return c,s
    def change(**kw):
        with lock:backup.update(kw)
    def weights():
        c=s=None;seen={}
        try:
            while not stop.is_set():
                try:
                    if c is None:c,s=connect()
                    for rel,a in walk(s,'work_dirs'):
                        if not rel.endswith('/ckpt_best.pth'):continue
                        try:s.stat(posixpath.join(REMOTE,posixpath.dirname(rel),'completion_summary.json'))
                        except IOError:continue
                        key=(a.st_mtime,a.st_size)
                        if seen.get(rel)==key:continue
                        local=OUT/rel;local.parent.mkdir(parents=True,exist_ok=True)
                        change(active=rel,transferred_bytes=0,total_bytes=a.st_size,last_error=None,last_progress_epoch=time.time())
                        code='import hashlib,sys;h=hashlib.sha256();f=open(sys.argv[1],"rb");[h.update(b) for b in iter(lambda:f.read(4194304),b"")];print(h.hexdigest())'
                        _,o,e=c.exec_command('/root/miniconda3/bin/python -c '+shlex.quote(code)+' '+shlex.quote(posixpath.join(REMOTE,rel)),timeout=30)
                        expected=o.read().decode().strip()
                        if o.channel.recv_exit_status() or len(expected)!=64:raise RuntimeError('Remote hash failed: '+e.read().decode(errors='replace'))
                        if not(local.exists() and local.stat().st_size==a.st_size and sha(local)==expected):
                            tmp=local.with_suffix('.pth.part')
                            def progress(done,total):change(transferred_bytes=done,total_bytes=total,last_progress_epoch=time.time())
                            offset=tmp.stat().st_size if tmp.exists() else 0
                            if offset>a.st_size:
                                tmp.unlink();offset=0
                            # Interrupted backups resume their partial file. Only
                            # the final full SHA can admit it as a valid backup.
                            with s.open(posixpath.join(REMOTE,rel),'rb') as remote_file, tmp.open('ab') as local_file:
                                remote_file.seek(offset)
                                remote_file.prefetch(a.st_size,max_concurrent_requests=32)
                                progress(offset,a.st_size)
                                while offset<a.st_size:
                                    block=remote_file.read(min(1048576,a.st_size-offset))
                                    if not block:raise IOError('Unexpected EOF: '+rel)
                                    local_file.write(block);local_file.flush();offset+=len(block);progress(offset,a.st_size)
                            if tmp.stat().st_size!=a.st_size or sha(tmp)!=expected:
                                tmp.unlink();raise RuntimeError('Weight size/hash mismatch: '+rel)
                            tmp.replace(local)
                        journal=ROOT/'reports/verified_backups.json'
                        d=json.loads(journal.read_text(encoding='utf-8')) if journal.exists() else {}
                        d[rel[10:]]=dict(sha256=expected,bytes=a.st_size,verified_at_epoch=time.time(),remote=posixpath.join(REMOTE,rel),local=str(local))
                        write(journal,d);seen[rel]=key
                        with lock:confirmed[rel]=key
                        change(active=None,transferred_bytes=a.st_size,verified_files=len(seen),last_error=None)
                    ready.set();stop.wait(60)
                except Exception as error:
                    ready.clear();change(last_error=f'{type(error).__name__}: {error}')
                    if s is not None:s.close()
                    if c is not None:c.close()
                    c=s=None;stop.wait(15)
        finally:
            if s is not None:s.close()
            if c is not None:c.close()
    if not args.metadata_once:threading.Thread(target=weights,daemon=True,name='independent-weight-backup').start()
    c=s=None;seen={}
    try:
        while not stop.is_set():
            try:
                if c is None:c,s=connect()
                # State is fresh before any slower historical evidence copy.
                for rel in ('work_dirs/campaign_state.json','work_dirs/finish_state.json'):
                    local=OUT/rel;local.parent.mkdir(parents=True,exist_ok=True);tmp=local.with_suffix('.json.part')
                    s.get(posixpath.join(REMOTE,rel),str(tmp),prefetch=False);tmp.replace(local)
                state=json.loads((OUT/'work_dirs/finish_state.json').read_text(encoding='utf-8'))
                with lock:snapshot=dict(updated_at_epoch=time.time(),finish=state.get('status'),mirrored_files=len(seen),backup=dict(backup),metadata_independent_of_backups=True,metadata_phase='priority_state')
                write(OUT/'mirror_status.json',snapshot);wanted={}
                # A minute-scale health check costs no model quota. Its SSH
                # channel and state publication are independent of large weights.
                _,health_out,health_err=c.exec_command('/root/miniconda3/bin/python -c '+shlex.quote(REMOTE_CHECK),timeout=30)
                health_text=health_out.read().decode('utf-8')
                if health_out.channel.recv_exit_status():raise RuntimeError('Health check failed: '+health_err.read().decode('utf-8',errors='replace'))
                write(ROOT/'reports/monitor_health.json',json.loads(health_text))
                for base in ('work_dirs','reports','configs'):
                    for rel,a in walk(s,base):
                        if rel.endswith('/ckpt_best.pth'):
                            try:s.stat(posixpath.join(REMOTE,posixpath.dirname(rel),'completion_summary.json'))
                            except IOError:pass
                            else:wanted[rel]=(a.st_mtime,a.st_size)
                        if Path(rel).suffix.lower() not in {'.json','.jsonl','.csv','.yaml','.txt','.log','.md'}:continue
                        key=(a.st_mtime,a.st_size)
                        if seen.get(rel)==key:continue
                        local=OUT/rel;local.parent.mkdir(parents=True,exist_ok=True);tmp=local.with_suffix(local.suffix+'.part')
                        if rel not in seen and local.exists() and local.stat().st_size==a.st_size and local.stat().st_mtime>=a.st_mtime:
                            seen[rel]=key;continue
                        s.get(posixpath.join(REMOTE,rel),str(tmp),prefetch=False);tmp.replace(local);seen[rel]=key
                state=json.loads((OUT/'work_dirs/finish_state.json').read_text(encoding='utf-8'))
                with lock:
                    all_backed=all(confirmed.get(rel)==key for rel,key in wanted.items())
                    snapshot=dict(updated_at_epoch=time.time(),finish=state.get('status'),mirrored_files=len(seen),backup=dict(backup),metadata_independent_of_backups=True,metadata_phase='full_evidence')
                write(OUT/'mirror_status.json',snapshot)
                if ROOT.name=='round2_highres':
                    from render_round2 import render
                    render(ROOT)
                if args.metadata_once or (state.get('status')=='complete' and all_backed):break
                stop.wait(60)
            except Exception as error:
                print(f'METADATA_RETRY: {type(error).__name__}: {error}',flush=True)
                if args.metadata_once:raise
                if s is not None:s.close()
                if c is not None:c.close()
                c=s=None;stop.wait(15)
    finally:
        stop.set()
        if s is not None:s.close()
        if c is not None:c.close()

if __name__=='__main__':main()
