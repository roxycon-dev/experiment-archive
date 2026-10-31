"""Stream only required TRAIN files between assigned nodes; no local archive."""
import sys,json,hashlib,time,threading,shlex
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'reports'))
from parallel_remote import connect_node2,command
from ops.remote import connect
from qlib.common import read_json,write_json,sha,utc_now
r='/root/autodl-tmp/HRDecoder_FinalDecision_20261005'

def deps_and_launch():
    with connect() as a,connect_node2() as b:
        with a.open_sftp() as af:binary=af.file(r+'/vendor/msda_build/MultiScaleDeformableAttention.so','rb').read()
        h=hashlib.sha256(binary).hexdigest()
        command(b,'mkdir -p '+shlex.quote(r+'/vendor/msda_build'),30)
        with b.open_sftp() as bf:
            with bf.file(r+'/vendor/msda_build/MultiScaleDeformableAttention.so','wb') as f:f.write(binary)
            for rel in ['reports/single_seed_parallel_amendment.json','reports/parallel_sd_worker.py']:
                bf.put(str(ROOT/rel),r+'/'+rel)
            receipt={'sha256':h,'bytes':len(binary),'source_instance':'connect.nmb2.seetacloud.com:36431','source_file':'vendor/msda_build/MultiScaleDeformableAttention.so','target_instance':'connect.nmb1.seetacloud.com:40920','identical_torch_and_python_ABI':True,'mandatory_kernel_reference_gate_will_run':True}
            with bf.file(r+'/reports/parallel_kernel_binary.json','w') as f:f.write(json.dumps(receipt))
        write_json(ROOT/'reports/parallel_kernel_binary.json',receipt)
        print(command(b,'/root/miniconda3/envs/hrdecoder_q_20261003/bin/python -m pip install --no-cache-dir numpy==1.26.4 timm==1.0.20',300)[-800:],flush=True)
        code="""
import subprocess,json,hashlib
from pathlib import Path
r=Path(%r);assert not (r/'reports/driver.lock').exists()
assert not (r/'runs/SD_seed42/identity.json').exists()
with (r/'reports/parallel_sd_worker.log').open('ab') as log:
 p=subprocess.Popen(['/root/miniconda3/envs/hrdecoder_q_20261003/bin/python','-u','reports/parallel_sd_worker.py'],cwd=r,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
print(json.dumps({'stage':'SECOND_NODE_GPU_ACCEPTANCE_LAUNCHED','pid':p.pid}))
"""%r
        out=json.loads(command(b,"/root/miniconda3/envs/hrdecoder_q_20261003/bin/python - <<'PY'\n"+code+"\nPY",30));write_json(ROOT/'reports/parallel_node2_worker_launch.json',{**out,'time':utc_now()});print(out,flush=True)

errors=[]
def prep():
    try:deps_and_launch()
    except BaseException as e:errors.append(str(e));print('DEPENDENCY_OR_LAUNCH_FAILURE',str(e),flush=True)
t=threading.Thread(target=prep);t.start()
with connect() as a,connect_node2() as b:
    command(b,'mkdir -p '+shlex.quote(r+'/inputs/TJDR'),30)
    ai,ao,ae=a.exec_command('tar -C '+shlex.quote(r)+' -cf - inputs/TJDR/train',timeout=120)
    bi,bo,be=b.exec_command('tar -C '+shlex.quote(r)+' -xf -',timeout=120)
    n=0;last=time.monotonic()
    while True:
        block=ao.read(1024**2)
        if not block:break
        bi.write(block);n+=len(block)
        if time.monotonic()-last>10:
            write_json(ROOT/'reports/parallel_train_input_transfer.json',{'stage':'TRANSFERRING_TRAIN_ONLY','bytes':n,'time':utc_now()});last=time.monotonic();print('TRAIN_BYTES',n,flush=True)
    bi.flush();bi.channel.shutdown_write()
    assert ao.channel.recv_exit_status()==0,ae.read().decode()
    assert bo.channel.recv_exit_status()==0,be.read().decode()
    code="""
import json,hashlib,shutil
from pathlib import Path
r=Path(%r);m=json.loads((r/'reports/TJDR_manifest.json').read_text());n=0
for row in m['samples']:
 for directory,key in [('image','image_sha256'),('annotation','label_sha256')]:
  p=r/'inputs/TJDR/train'/directory/(row['sid']+'.png');assert hashlib.sha256(p.read_bytes()).hexdigest()==row[key];n+=1
assert n==896
result={'stage':'ALL896_TRAIN_FILES_SHA_VERIFIED','files':n,'test_downloaded':False,'free_disk_bytes':shutil.disk_usage(r).free}
assert result['free_disk_bytes']>=12*1024**3
(r/'reports/parallel_data_ready.json').write_text(json.dumps(result));print(json.dumps(result))
"""%r
    result=json.loads(command(b,"/root/miniconda3/envs/hrdecoder_q_20261003/bin/python - <<'PY'\n"+code+"\nPY",300));write_json(ROOT/'reports/parallel_node2_data_ready.json',result);print(result,flush=True)
t.join();assert not errors,errors
