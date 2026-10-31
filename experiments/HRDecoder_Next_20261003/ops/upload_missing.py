"""Concurrent transactional uploads over the task's SSH key; no passwords."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib
import time
import paramiko

ROOT = Path(__file__).resolve().parents[1]
KEY = Path.home()/'.ssh/codex_hrdecoder_next_20261003'

def upload(name):
    local = ROOT/'assets'/name
    remote = '/root/autodl-tmp/HRDecoder_Next_20261003/assets/'+name
    for attempt in range(4):
        client = paramiko.SSHClient()
        client.load_host_keys(str(ROOT/'ops/known_hosts'))
        try:
            client.connect('connect.nmb2.seetacloud.com',port=36431,username='root',
                           key_filename=str(KEY),allow_agent=False,look_for_keys=False,timeout=25)
            client.get_transport().set_keepalive(30)
            sftp = paramiko.SFTPClient.from_transport(client.get_transport(),window_size=64*1024**2)
            partial = remote+'.upload'
            try: offset=sftp.stat(partial).st_size
            except IOError: offset=0
            if offset>local.stat().st_size: offset=0
            with local.open('rb') as src, sftp.file(partial,'r+' if offset else 'w') as dst:
                src.seek(offset);dst.seek(offset);dst.set_pipelined(True)
                last=time.monotonic()
                while True:
                    block=src.read(1024**2)
                    if not block:break
                    dst.write(block)
                    if time.monotonic()-last>20:
                        print(name,round(src.tell()/local.stat().st_size*100,1),'%',flush=True)
                        last=time.monotonic()
            h=hashlib.sha256(local.read_bytes()).hexdigest()
            _,out,err=client.exec_command('sha256sum '+partial,timeout=120)
            actual=out.read().decode().split()[0]
            if actual!=h:raise ValueError('Upload SHA mismatch '+name)
            sftp.posix_rename(partial,remote)
            print('VERIFIED_UPLOAD',name,h,flush=True)
            return
        except Exception as exc:
            print('RETRY',name,attempt,type(exc).__name__,str(exc),flush=True)
            if attempt==3:raise
            time.sleep(5)
        finally:client.close()

if __name__=='__main__':
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(upload,['U1.pth','H1.pth']))
