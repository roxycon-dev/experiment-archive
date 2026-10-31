"""Small bounded transport diagnostic for required multi-GB endpoint backup; no GPU work."""
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib
import json
import shlex
import threading
import time
from training_remote_control import ROOT,REMOTE,connect

source=REMOTE+'/recovered_archives/S_seed42_results.zip'
reference=ROOT/'transfer/S_seed42_results.zip'
assert reference.exists()
chunk_bytes=2*1024**2;barrier=threading.Barrier(4);start=time.monotonic()
def worker(i):
    offset=i*chunk_bytes
    with reference.open('rb') as f:f.seek(offset);expected=hashlib.sha256(f.read(chunk_bytes)).hexdigest()
    target=ROOT/'transfer/transport_probe'/('range_'+str(i)+'.bin');target.parent.mkdir(exist_ok=True)
    with connect() as c:
        barrier.wait(timeout=30)
        ch=c.get_transport().open_session(window_size=128*1024**2,max_packet_size=1024**2);ch.settimeout(60)
        code="import sys;f=open(sys.argv[1],'rb');f.seek(int(sys.argv[2]));n=int(sys.argv[3]);\nwhile n:\n b=f.read(min(n,1048576));assert b;n-=len(b);sys.stdout.buffer.write(b);sys.stdout.buffer.flush()"
        ch.exec_command(shlex.join(['/root/miniconda3/bin/python','-u','-c',code,source,str(offset),str(chunk_bytes)]))
        t=time.monotonic();h=hashlib.sha256();n=0
        with target.open('wb') as f:
            while True:
                data=ch.recv(1024**2)
                if not data:break
                h.update(data);f.write(data);n+=len(data)
        assert ch.recv_exit_status()==0 and n==chunk_bytes and h.hexdigest()==expected
        return {'range':i,'bytes':n,'seconds':time.monotonic()-t,'SHA_verified':True}
with ThreadPoolExecutor(max_workers=4) as pool:
    futures=[pool.submit(worker,i) for i in range(4)];rows=[]
    for f in as_completed(futures):rows.append(f.result());print(json.dumps(rows[-1]),flush=True)
elapsed=time.monotonic()-start
result={'status':'PASS','scope':'8MiB immutable already-recovered archive, four independent authorized SSH streams',
        'bytes':4*chunk_bytes,'seconds_including_handshakes':elapsed,'aggregate_MiB_per_second':4*chunk_bytes/1024**2/elapsed,
        'per_connection':rows,'no_model_inference_or_training':True}
(ROOT/'evidence/parallel_transport_probe.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
