"""Archived F12/seed2024 reproduction, fixed12k; old campaign remains closed."""
import time,json,shutil
import torch
from qlib.common import read_json,write_json,sha,utc_now,assert_source
from qlib import checkpoint
from qlib.geometry import center_logits
from qlib.loss import masked_loss
from experiment import ROOT,load_parent,rows_for,val_dataset
from inputs import loader_for
from validation import evaluate
from storage import publish_parent_probabilities,wait_for_space
import parent_adaptation as adaptation

def run(runtime_path,shared,resume=False):
    runtime=read_json(runtime_path);source=assert_source(ROOT);cached,cache_receipt,val,p0,p0_receipt=shared
    gate=read_json(ROOT/'reports/gpu_readiness.json');assert gate['status']=='READY' and gate['source_sha256']==source['sha256']
    runpath=ROOT/'runs/PARENT_F12_seed2024';runpath.mkdir(parents=True,exist_ok=True)
    model,loading=load_parent(runtime,2024,q_parent=True);adaptation.configure(model,'F12');model=model.cuda().train()
    optimizer,groups=adaptation.optimizer_for(model);scheduler=adaptation.Schedule(optimizer)
    rows=rows_for('PARENT',2024)
    identity={'arm':'PARENT_F12','seed':2024,'endpoint':12000,'source_sha256':source['sha256'],
        'runtime_sha256':sha(runtime_path),'initialization':loading,'historical_F_stream_sha256':sha(ROOT/'reports/stream_PARENT_seed2024.jsonl'),
        'procedure':'Exact archived F12: no EMA, no forward RNG reseeding, fresh archived AdamW/schedule, micro2x4, BF16; IO-only packed cache'}
    completed=checkpoint.load(runpath/'last.pth',model,optimizer,scheduler,None,identity,rows) if resume else 0
    if not resume:assert not (runpath/'last.pth').exists(),'Explicit parent resume required'
    write_json(runpath/'identity.json',identity);write_json(runpath/'parameter_groups.json',groups)
    audits=read_json(runpath/f'valid_audit_u{completed}.json')['sources'] if completed else {}
    loader=loader_for(cached,rows,'DDR',completed,12000,runtime['resources'],2024)
    try:
        for u in range(completed+1,12001):
            optimizer.zero_grad(set_to_none=True);losses=[];started=time.perf_counter()
            for micro in range(4):
                x,g,v,legal,positive=next(loader)
                audit=audits.setdefault('DDR',{'windows':0,'valid_pixels':[0]*4,'positive_pixels':[0]*4});audit['windows']+=len(x)
                for key,tensor in (('valid_pixels',legal),('positive_pixels',positive)):
                    audit[key]=[a+int(b) for a,b in zip(audit[key],tensor.sum(0).tolist())]
                x,g,v=(t.cuda(non_blocking=True) for t in (x,g,v))
                with torch.autocast('cuda',dtype=torch.bfloat16):z=center_logits(model(x))
                loss=masked_loss(z,g,v)/4;loss.backward();losses.append(loss.detach())
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            report=torch.stack((torch.stack(losses).sum(),norm)).cpu().tolist()
            assert all(__import__('math').isfinite(x) for x in report),'Nonfinite parent update'
            optimizer.step();scheduler.step();completed=u
            if u%20==0:
                stat={'update':u,'loss':report[0],'grad_norm':report[1],'seconds':time.perf_counter()-started,'time':utc_now()}
                with (runpath/'train.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(stat)+'\n')
                write_json(runpath/'status.json',{'stage':'TRAINING','completed':u,'endpoint':12000,'recent':stat,'updated_at':utc_now()})
            if u%200==0:
                wait_for_space(8*1024**3,runpath,u)
                checkpoint.save(runpath/'last.pth',model,optimizer,scheduler,None,u,identity,rows[u*8:u*8+8])
                write_json(runpath/f'valid_audit_u{u}.json',{'completed':u,'sources':audits})
            if u in (2000,4000,8000,12000):
                evaluate(model,val,runtime,runpath/f'val_u{u}',u,keep_probability=u==12000,p0_ram=p0)
                checkpoint.save(runpath/'last.pth',model,optimizer,scheduler,None,u,identity,rows[u*8:u*8+8])
        path=runpath/'checkpoint_u12000.pth';shutil.copy2(runpath/'last.pth',path)
        metrics=read_json(runpath/'val_u12000/metrics.json');write_json(ROOT/'reports/parent_validation_seed2024.json',metrics)
        write_json(ROOT/'reports/parent_seed2024.json',{'status':'READY','completed':12000,'sha256':sha(path),'path':str(path),'identity':identity,'time':utc_now()})
        write_json(runpath/'status.json',{'stage':'COMPLETE','completed':12000,'endpoint':12000,'updated_at':utc_now()})
        publish_parent_probabilities(runpath)
    except BaseException as e:
        write_json(runpath/'status.json',{'stage':'TECHNICAL_PAUSE','completed_in_memory':completed,'reason':str(e),'automatic_restart':False,'updated_at':utc_now()});raise
