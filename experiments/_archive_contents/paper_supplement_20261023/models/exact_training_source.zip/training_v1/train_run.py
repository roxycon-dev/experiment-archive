"""Finite 32k registered run; incomplete collapse/technical pause never promoted."""
import argparse,shutil,time,json
from pathlib import Path
import torch
from qlib.common import read_json,write_json,sha,utc_now,assert_source
from qlib.data import DDR
from experiment import ROOT,initialize,rows_for,next_rows,val_dataset,identity_for
from inputs import TJDR,cache_sources,loader_for
from schedule import optimizer_for,Schedule
from state import EMA,save,load
from engine import update
from validation import evaluate,cache_p0
from storage import wait_for_space,publish_endpoint
NODES=(4000,8000,12000,16000,20000,24000,28000,32000)

def review_trigger(nodes,parent_score,reference_score):
    ordered=sorted(nodes)
    if len(ordered)<2:return False
    return all(nodes[u]['single']['miou_pct']<=parent_score-3. and
               nodes[u]['fixed_fusion']['miou_pct']<=reference_score-1.5 for u in ordered[-2:])

def run(runtime_path,arm,seed,resume=False,shared=None):
    runtime=read_json(runtime_path);gate=read_json(ROOT/'reports/gpu_readiness.json');source=assert_source(ROOT)
    assert gate['status']=='READY' and gate['source_sha256']==source['sha256'] and gate['runtime_sha256']==sha(runtime_path)
    runpath=ROOT/f'runs/{arm}_seed{seed}';runpath.mkdir(parents=True,exist_ok=True)
    completed=0
    try:
        if arm in ('S','SD'):
            from kernel import load_kernel
            load_kernel()
        model,loading=initialize(runtime,arm,seed);model.train()
        optimizer,groups=optimizer_for(model);scheduler=Schedule(optimizer);ema=EMA(model)
        streams={'DDR':rows_for('DDR',seed),'TJDR':rows_for('TJDR',seed)}
        identity=identity_for(runtime_path,arm,seed,loading)
        if resume:completed=load(runpath/'last.pth',model,optimizer,scheduler,ema,identity,lambda u:next_rows(streams,u,arm))
        else:assert not (runpath/'last.pth').exists(),'Existing checkpoint requires explicit resume'
        write_json(runpath/'identity.json',identity);write_json(runpath/'parameter_groups.json',groups)
        audits=read_json(runpath/f'valid_audit_u{completed}.json')['sources'] if completed else {}
        if shared is None:
            datasets={'DDR':DDR(runtime['ddr_root'],'train',read_json(ROOT/'reports/data_identity.json')['splits']['train']),
                      'TJDR':TJDR(runtime['TJDR_root'],read_json(ROOT/'reports/TJDR_manifest.json'))}
            cached,cache_receipt=cache_sources(datasets,runtime['resources']['shared_cache_limit_bytes'])
            val=val_dataset(runtime);p0,p0_receipt=cache_p0(val,runtime)
        else:cached,cache_receipt,val,p0,p0_receipt=shared
        write_json(runpath/'cache.json',{'training':cache_receipt,'P0':p0_receipt})
        loaders={'DDR':loader_for(cached,streams['DDR'],'DDR',completed,32000,runtime['resources'],seed)}
        if arm in ('D','SD') and completed<20000:loaders['TJDR']=loader_for(cached,streams['TJDR'],'TJDR',completed,20000,runtime['resources'],seed)
        nodes={int(p.parent.name[5:]):read_json(p) for p in runpath.glob('val_u*/metrics.json')}
        parent_score=read_json(ROOT/f'reports/parent_validation_seed{seed}.json')['single']['miou_pct']
        reference_score=read_json(ROOT/'configs/protocol.json')['evaluation']['R_val_by_seed'][str(seed)]
        def checkpoint(u,endpoint=False):
            wait_for_space(8*1024**3,runpath,u)
            save(runpath/'last.pth',model,optimizer,scheduler,ema,u,identity,next_rows(streams,u,arm))
            write_json(runpath/f'valid_audit_u{u}.json',{'completed':u,'classes':['EX','HE','SE','MA'],'sources':audits})
            if endpoint:shutil.copy2(runpath/'last.pth',runpath/f'checkpoint_u{u}.pth')
        def node(u):
            write_json(runpath/'status.json',{'stage':'VALIDATING','completed':u,'endpoint':32000,'updated_at':utc_now()})
            nodes[u]=evaluate(ema.model,val,runtime,runpath/f'val_u{u}',u,u in (20000,32000),p0_ram=p0)
            checkpoint(u,u in (20000,32000))
            if u in (20000,32000):publish_endpoint(runpath,u)
            if review_trigger(nodes,parent_score,reference_score):
                write_json(runpath/'status.json',{'stage':'CURVE_REVIEW_REQUIRED','completed':u,'nodes':sorted(nodes)[-2:],
                    'reason':'Two consecutive registered nodes meet collapse review criterion','automatic_cancel':False,'updated_at':utc_now()})
                return False
            return True
        if completed in NODES and completed not in nodes:
            if not node(completed):return 'CURVE_REVIEW_REQUIRED'
        for u in range(completed+1,32001):
            stat=update(model,optimizer,scheduler,ema,loaders,seed,u-1,audits);completed=u
            if u==20000 and 'TJDR' in loaders:del loaders['TJDR']
            if u%20==0:
                torch.cuda.synchronize();stat['peak_gpu_bytes']=torch.cuda.max_memory_allocated()
                with (runpath/'train.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps({'update':u,'time':utc_now(),**stat})+'\n')
                write_json(runpath/'status.json',{'stage':'TRAINING','completed':u,'endpoint':32000,'updated_at':utc_now(),'recent':stat})
                print(arm,seed,u,stat,flush=True)
            if u%200==0:checkpoint(u)
            if u in NODES and not node(u):return 'CURVE_REVIEW_REQUIRED'
        write_json(runpath/'status.json',{'stage':'COMPLETE','completed':completed,'endpoint':32000,'updated_at':utc_now()})
        return 'COMPLETE'
    except BaseException as e:
        write_json(runpath/'status.json',{'stage':'TECHNICAL_PAUSE','completed_in_memory':completed,'reason':f'{type(e).__name__}: {e}',
                  'automatic_restart':False,'updated_at':utc_now()});raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--runtime',default='configs/runtime.server.json');p.add_argument('--arm',choices=['C','S','D','SD'],required=True)
    p.add_argument('--seed',type=int,choices=[42,2024],required=True);p.add_argument('--resume',action='store_true');a=p.parse_args()
    run(a.runtime,a.arm,a.seed,a.resume)
