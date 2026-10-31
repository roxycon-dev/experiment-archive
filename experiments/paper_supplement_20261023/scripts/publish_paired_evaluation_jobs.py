"""Publish immutable20k endpoints as they finish; never select among checkpoints."""
from pathlib import Path
import hashlib
import json
import os
import shlex
import subprocess
import time
import torch

ROOT=Path(__file__).resolve().parents[1];EXP=ROOT.parent
DEST='/home/studentuser001/paper_supplement_20261023/evaluation_workspace'
LANROOT=DEST+'/paper_supplement_20261023'
SSH=['ssh','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-i','/root/.ssh/codex_supp_to59_20261023','-p','40560','studentuser001@127.0.0.1']
SCP=['scp','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-i','/root/.ssh/codex_supp_to59_20261023','-P','40560']
parent='68d7b5821fab1210b3336cf2a63b9de893f24fc32876f5255d7d662084b5e728'
torch.set_num_threads(1)

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()

def write(p,value):
    p.parent.mkdir(parents=True,exist_ok=True);temp=p.with_suffix('.tmp')
    temp.write_text(json.dumps(value,indent=2));temp.replace(p)

def remote(code):return subprocess.check_output(SSH+['/usr/bin/python3 -c '+shlex.quote(code)],text=True)

def publish(job):
    name=job['arm']+'_seed'+str(job['seed'])
    job['runtime_protocol_sha256']=sha(ROOT/'plan/paired_evaluation_runtime.md')
    dst=LANROOT+'/eval_jobs/'+name+'.json'
    code="from pathlib import Path;import json;p=Path("+repr(dst)+");p.parent.mkdir(parents=True,exist_ok=True);v="+repr(json.dumps(job,indent=2))+";assert not p.exists() or json.loads(p.read_text())==json.loads(v);t=p.with_suffix('.tmp');t.write_text(v);t.replace(p)"
    remote(code)
    write(ROOT/'evidence/paired_jobs_published'/(name+'.json'),job)
    print(json.dumps({'published':name,'completed':20000,'EMA_completed':20000}),flush=True)

lineage=json.loads((ROOT/'evidence/matched_lineage_audit.json').read_text());assert lineage['parent_sha256']==parent and all(lineage['matched_checks'].values())
c_receipt=json.loads((ROOT/'evidence/C20_export_receipt.json').read_text())
for arm in ('C','S'):
    weight=EXP/'HRDecoder_ArchitectureDesign_20261008/cache_execution/inference_weights'/(arm+'20_ema_inference.pth')
    expected=lineage['S_compact_sha256'] if arm=='S' else c_receipt['compact_sha256']
    assert sha(weight)==expected
    publish({'arm':arm,'seed':42,'completed':20000,'EMA_completed':20000,'parent_full_sha256':parent,
             'weight_path':DEST+'/HRDecoder_ArchitectureDesign_20261008/cache_execution/inference_weights/'+weight.name,
             'weight_sha256':expected,'source_checkpoint_sha256':lineage['S_full_sha256'] if arm=='S' else lineage['C_sha256'],
             'source':'Historical seed42 bitwise EMA export and audited matched lineage'})

done=set();exports=ROOT/'export_weights';exports.mkdir(exist_ok=True)
while len(done)<4:
    for seed in (2024,3407):
        for arm in ('C','S'):
            name=arm+'_seed'+str(seed)
            if name in done:continue
            run=ROOT/'training_v1/runs'/name
            complete=run/'completion.json'
            if not complete.exists():continue
            completion=json.loads(complete.read_text())
            assert completion['status']=='COMPLETE' and completion['completed']==completion['EMA_completed']==20000
            assert completion['schedule_horizon']==32000
            source=run/'last.pth';source_sha=sha(source);assert source_sha==completion['endpoint_checkpoint_sha256']
            state=torch.load(source,map_location='cpu',weights_only=False,mmap=True)
            assert state['completed']==state['ema_completed']==state['scheduler']['completed']==20000
            assert state['scheduler']['horizon']==32000
            identity=state['identity'];assert identity['arm']==arm and identity['seed']==seed and identity['parent_full_sha256']==parent
            target=exports/(name+'_EMA20k.pth');receipt_path=exports/(name+'_export.json')
            if not target.exists():
                temp=target.with_suffix('.tmp')
                torch.save({'ema':state['ema'],'completed':20000,'ema_completed':20000,'training_identity':identity,
                            'source_checkpoint_sha256':source_sha,'source_state_key':'ema'},temp);temp.replace(target)
            check=torch.load(target,map_location='cpu',weights_only=False,mmap=True)
            assert set(check['ema'])==set(state['ema']) and all(torch.equal(check['ema'][k],v) for k,v in state['ema'].items())
            digest=sha(target)
            receipt={'status':'PASS','arm':arm,'seed':seed,'completed':20000,'EMA_completed':20000,'parent_full_sha256':parent,
                     'source_checkpoint_sha256':source_sha,'weight_sha256':digest,'EMA_tensor_bytes_bitwise_equal':True,
                     'tensor_count':len(state['ema']),'training_identity':identity}
            write(receipt_path,receipt);del state,check
            dst=DEST+'/assets/paired_weights/'+target.name
            existing=json.loads(remote("from pathlib import Path;import json,hashlib;p=Path("+repr(dst)+");p.parent.mkdir(parents=True,exist_ok=True);print(json.dumps({'exists':p.exists(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None}))"))
            if existing['exists']:assert existing['sha256']==digest
            else:
                subprocess.run(SCP+[str(target),'studentuser001@127.0.0.1:'+dst+'.part'],check=True)
                remote("from pathlib import Path;import hashlib;p=Path("+repr(dst+'.part')+");assert hashlib.sha256(p.read_bytes()).hexdigest()=="+repr(digest)+";p.replace("+repr(dst)+")")
            publish({'arm':arm,'seed':seed,'completed':20000,'EMA_completed':20000,'parent_full_sha256':parent,
                     'weight_path':dst,'weight_sha256':digest,'source_checkpoint_sha256':source_sha,
                     'source':'New fixed20k endpoint, exact EMA export','export_receipt':receipt})
            done.add(name)
    write(ROOT/'paired_endpoint_publisher_status.json',{'stage':'ALL_SIX_ENDPOINT_JOBS_PUBLISHED' if len(done)==4 else 'WAITING_FIXED20K_ENDPOINTS','new_endpoints_published':sorted(done),'historic_seed42_jobs_published':2,'updated_unix':time.time()})
    if len(done)<4:time.sleep(30)
