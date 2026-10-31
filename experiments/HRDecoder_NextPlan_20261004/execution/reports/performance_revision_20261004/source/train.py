"""Bounded registered run. Technical failures pause; scientific stops advance the queue."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time
import torch
from torch.utils.data import DataLoader
import adaptation
from decision import ENDPOINT,NODES,catastrophic
from qlib.common import assert_source,digest,read_json,seed_all,sha,utc_now,write_json
from qlib.data import DDR,External,WindowDataset
from qlib.model import build_model
from qlib.optim import make_optimizer,Schedule as OriginalSchedule
from qlib.loss import masked_loss
from qlib.geometry import center_logits
from qlib.evaluate import evaluate
from qlib import checkpoint

ROOT = Path(__file__).resolve().parent


def fingerprint(model):
    h = hashlib.sha256()
    for name,tensor in sorted(model.state_dict().items()):
        value=tensor.detach().cpu().contiguous()
        h.update(name.encode()); h.update(str(value.dtype).encode()); h.update(str(tuple(value.shape)).encode())
        h.update(value.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def initialize(runtime,arm,seed):
    seed_all(seed)
    pretraining=read_json(ROOT/'reports/pretraining_identity.json')
    assert pretraining['status']=='READY'
    model,loading=build_model(runtime['dinov3_repo'],runtime['pretrained_checkpoint'],pretraining['sha256'])
    loading['initial_Q_state_sha256']=fingerprint(model)
    if arm!='E_NEG':
        initial=read_json(ROOT/'configs/initial_weights.json')[str(seed)]
        path=initial['server'] if str(ROOT).startswith('/root/') else initial['local']
        if sha(path)!=initial['sha256']:
            raise ValueError('Q16k model checkpoint SHA differs')
        state=torch.load(path,map_location='cpu',weights_only=False)
        assert state['completed']==16000 and state['identity']==initial['identity']
        assert state['identity_sha256']==digest(initial['identity'])
        model.load_state_dict(state['model'],strict=True)
        loading.update(start_weights_sha256=initial['sha256'],loaded_model_state_sha256=fingerprint(model),
                       checkpoint_completed=16000,old_optimizer_loaded=False)
        del state
        adaptation.configure(model,arm)
    return model,loading


def datasets_for(runtime,arm):
    data=read_json(ROOT/'reports/data_identity.json')
    datasets={'DDR':DDR(runtime['ddr_root'],'train',data['splits']['train'])}
    if arm=='E_NEG':
        manifest=read_json(runtime['external_manifest'])
        for source in ('IDRiD','MAPLES'):
            datasets[source]=External(source,manifest)
    return datasets,DDR(runtime['ddr_root'],'val',data['splits']['val'])


def rows_for(arm,seed):
    path=ROOT/f'reports/stream_{"E_NEG" if arm=="E_NEG" else "F"}_seed{seed}.jsonl'
    receipt=read_json(ROOT/'reports/stream_identity.json')
    assert sha(path)==receipt[path.name]
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()],sha(path)


def loader_for(datasets,rows,start,end,resources,seed):
    kwargs={'batch_size':resources['micro_batch'],'shuffle':False,'num_workers':resources['workers'],
            'pin_memory':True,'generator':torch.Generator().manual_seed(seed+333)}
    if resources['workers']:
        kwargs.update(prefetch_factor=resources['prefetch_factor'],persistent_workers=True)
    return iter(DataLoader(WindowDataset(datasets,rows,start*8,end*8),**kwargs))


def update(model,optimizer,scheduler,loader,resources,completed,valid_audit=None,stream_rows=None):
    model.set_update(completed)
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize(); started=time.perf_counter()
    waiting=0.;loss_value=0.
    for micro_index in range(resources['accumulation']):
        before=time.perf_counter();x,g,v=next(loader);waiting+=time.perf_counter()-before
        if valid_audit is not None:
            legal=v.sum(dim=(2,3)).tolist()
            foreground=(g*v).sum(dim=(2,3)).tolist()
            for index,(vv,gg) in enumerate(zip(legal,foreground)):
                slot=micro_index*resources['micro_batch']+index
                source=stream_rows[completed*8+slot]['source']
                audit=valid_audit.setdefault(source,{'valid_pixels':[0]*4,'positive_pixels':[0]*4,'windows':0})
                audit['windows']+=1
                audit['valid_pixels']=[a+b for a,b in zip(audit['valid_pixels'],vv)]
                audit['positive_pixels']=[a+b for a,b in zip(audit['positive_pixels'],gg)]
        x,g,v=(t.to('cuda',non_blocking=True) for t in (x,g,v))
        with torch.autocast('cuda',dtype=torch.bfloat16):
            z=center_logits(model(x))
        loss=masked_loss(z,g,v)/resources['accumulation']
        loss_value+=float(loss.detach())
        loss.backward()
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
    if not torch.isfinite(norm):
        raise FloatingPointError('Nonfinite gradient; no skipped update')
    optimizer.step();scheduler.step();torch.cuda.synchronize()
    return {'loss':loss_value,'seconds_per_update':time.perf_counter()-started,
            'data_wait_seconds':waiting,'grad_norm':float(norm)}


def disk_check():
    if shutil.disk_usage(ROOT).free < 2*1024**3:
        raise OSError('Insufficient checkpoint reserve: less than2GiB; pause for cleanup')


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--runtime',default='configs/runtime.server.json')
    ap.add_argument('--arm',choices=['E_NEG','F4','F12'],required=True)
    ap.add_argument('--seed',type=int,choices=[42,2024],required=True)
    ap.add_argument('--resume',action='store_true')
    args=ap.parse_args();runtime=read_json(args.runtime)
    gate=read_json(ROOT/'reports/gpu_readiness.json')
    source=assert_source(ROOT)
    assert gate['status']=='READY' and gate['runtime_sha256']==sha(args.runtime)
    assert gate['source_sha256']==source['sha256']
    if args.seed==2024:
        campaign=read_json(ROOT/'reports/campaign.json')
        assert args.arm in campaign['authorized_seed2024_arms'], 'Second seed has not passed promotion gate'
    resources=runtime['resources'] if args.arm=='E_NEG' else runtime['f_resources']
    assert resources['dtype']=='bfloat16' and resources['micro_batch']*resources['accumulation']==8
    if args.arm=='E_NEG':
        assert (resources['micro_batch'],resources['accumulation'])==(2,4)
        assert gate['E_NEG_paired_acceptance']['status']=='PASS'
    endpoint=ENDPOINT[args.arm]
    run=ROOT/f'runs/{args.arm}_seed{args.seed}'
    run.mkdir(parents=True,exist_ok=True)
    try:
        model,loading=initialize(runtime,args.arm,args.seed)
        if args.arm=='E_NEG':
            assert loading['initial_Q_state_sha256']==gate['E_NEG_paired_acceptance']['initialization_sha256_by_seed'][str(args.seed)]
        model=model.to('cuda').train()
        if args.arm=='E_NEG':
            optimizer,scheduler=make_optimizer(model),None
            scheduler=OriginalSchedule(optimizer);parameter_groups=None
        else:
            optimizer,parameter_groups=adaptation.optimizer_for(model)
            scheduler=adaptation.Schedule(optimizer)
        datasets,val=datasets_for(runtime,args.arm)
        rows,stream_sha=rows_for(args.arm,args.seed)
        identity={'arm':args.arm,'seed':args.seed,'endpoint':endpoint,'source_sha256':source['sha256'],
                  'protocol_sha256':sha(ROOT/'configs/protocol.json'),'runtime_sha256':sha(args.runtime),
                  'resources':resources,'stream_sha256':stream_sha,'data_sha256':read_json(ROOT/'reports/data_identity.json')['sha256'],
                  'initialization':loading,'external_manifest_sha256':sha(runtime['external_manifest']) if args.arm=='E_NEG' else None,
                  'p0_identity_sha256':sha(ROOT/'configs/p0_identity.json')}
        write_json(run/'initialization.json',loading)
        if parameter_groups is not None:
            write_json(run/'trainable_parameter_groups.json',parameter_groups)
        completed=0
        if args.resume:
            completed=checkpoint.load(run/'last.pth',model,optimizer,scheduler,None,identity,rows)
        elif (run/'last.pth').exists():
            raise ValueError('Existing run requires explicit identity-checked resume')
        write_json(run/'identity.json',identity)
        valid_audit={}
        if args.resume and (run/f'valid_audit_u{completed}.json').exists():
            valid_audit=read_json(run/f'valid_audit_u{completed}.json')['sources']
        elif args.resume and completed:
            raise ValueError('Missing matching source-valid audit at checkpoint cursor')
        nodes={int(p.parent.name.removeprefix('val_u')):read_json(p) for p in run.glob('val_u*/metrics.json')}
        references=read_json(ROOT/'configs/references.json')
        if completed in NODES[args.arm] and completed not in nodes:
            nodes[completed]=evaluate(model,val,runtime,run/f'val_u{completed}',completed,completed==endpoint)
        if catastrophic(args.arm,nodes,args.seed,references):
            write_json(run/'status.json',{'stage':'CATASTROPHIC_STOP','completed':completed,'nodes':[2000,4000],
                                          'paired_12k_conclusion_allowed':False,'updated_at':utc_now()})
            return
        loader=loader_for(datasets,rows,completed,endpoint,resources,args.seed)
        for u in range(completed+1,endpoint+1):
            statistics=update(model,optimizer,scheduler,loader,resources,u-1,valid_audit,rows)
            completed=u
            if u%20==0:
                with (run/'train.jsonl').open('a',encoding='utf-8') as f:
                    f.write(json.dumps({'update':u,**statistics,'time':utc_now()})+'\n')
                write_json(run/'status.json',{'stage':'TRAINING','completed':u,'endpoint':endpoint,'updated_at':utc_now()})
                print(args.arm,args.seed,u,statistics,flush=True)
            if u%200==0 or u in NODES[args.arm]:
                disk_check()
                checkpoint.save(run/'last.pth',model,optimizer,scheduler,None,u,identity,rows[u*8:u*8+8])
                write_json(run/f'valid_audit_u{u}.json',{'completed':u,'sources':valid_audit,'classes':['EX','HE','SE','MA']})
            if u in NODES[args.arm]:
                nodes[u]=evaluate(model,val,runtime,run/f'val_u{u}',u,keep_probability=u==endpoint)
                # Preserve registered late-node weights, optimizer and RNG. Never choose best step.
                if u in ((20000,24000) if args.arm=='E_NEG' else (8000,12000)):
                    disk_check()
                    checkpoint.save(run/f'checkpoint_u{u}.pth',model,optimizer,scheduler,None,u,identity,rows[u*8:u*8+8])
                checkpoint.save(run/'last.pth',model,optimizer,scheduler,None,u,identity,rows[u*8:u*8+8])
                if u==4000 and catastrophic(args.arm,nodes,args.seed,references):
                    write_json(run/'status.json',{'stage':'CATASTROPHIC_STOP','completed':u,'nodes':[2000,4000],
                                                  'paired_12k_conclusion_allowed':False,'updated_at':utc_now()})
                    return
        write_json(run/'status.json',{'stage':'COMPLETE','completed':completed,'endpoint':endpoint,
                                      'endpoint_metrics':nodes[endpoint],'updated_at':utc_now()})
    except BaseException as error:
        write_json(run/'status.json',{'stage':'TECHNICAL_PAUSE','reason':f'{type(error).__name__}: {error}',
                                      'completed_in_memory':locals().get('completed'),
                                      'automatic_restart':False,'updated_at':utc_now()})
        raise


if __name__=='__main__':
    main()
