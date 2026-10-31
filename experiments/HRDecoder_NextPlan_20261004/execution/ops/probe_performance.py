"""Bounded equivalence and throughput check, never a candidate or validation node."""
import gc
import hashlib
from pathlib import Path
import sys
import time
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import train
import performance
import adaptation
from qlib import checkpoint
from qlib.common import read_json,sha,utc_now,write_json
from qlib.optim import make_optimizer,Schedule


def tensor_hash(named):
    h=hashlib.sha256()
    for name,a in sorted(named):
        if a is None:continue
        b=a.detach().cpu().contiguous()
        h.update(name.encode()); h.update(str(b.dtype).encode());h.update(str(tuple(b.shape)).encode())
        h.update(b.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def snapshot(model,optimizer,scheduler,audit,statistics):
    opt=[]; metadata=[]
    for key,state in optimizer.state_dict()['state'].items():
        for name,value in state.items():
            if torch.is_tensor(value):opt.append((str(key)+'/'+name,value))
            else:metadata.append((key,name,value))
    return {'model':train.fingerprint(model),'gradients':tensor_hash([(n,p.grad) for n,p in model.named_parameters()]),
            'optimizer_tensors':tensor_hash(opt),'optimizer_metadata':metadata,'scheduler':scheduler.state_dict(),
            'rng_torch':tensor_hash([('cpu',torch.get_rng_state()),*[(str(i),a) for i,a in enumerate(torch.cuda.get_rng_state_all())]]),
            'audit':audit,'loss_grad_rows':[(x['loss'],x['grad_norm']) for x in statistics]}


def close(loader):
    if loader is not None:loader._shutdown_workers()


def main():
    runtime=read_json(ROOT/'configs/runtime.server.json');config=read_json(ROOT/'configs/performance.json')
    datasets,_=train.datasets_for(runtime,'E_NEG')
    print('building shared decoded cache',flush=True)
    cached,cache_receipt=performance.cache_sources(datasets,config,lambda x:print(x,flush=True))
    results={}
    for arm in ('E_NEG','F4','F12'):
        # Formal training keeps its historical nondeterministic CUDA setting.
        # For the comparison only, eliminate CUDA reduction-order noise so a
        # changed tensor/gradient cannot hide inside ordinary repeat variation.
        torch.use_deterministic_algorithms(True)
        print('equivalence',arm,flush=True)
        model,_=train.initialize(runtime,arm,42);model=model.cuda().train()
        if arm=='E_NEG':
            optimizer=make_optimizer(model);scheduler=Schedule(optimizer)
            identity=read_json(ROOT/'runs/E_NEG_seed42/identity.json')
            rows,_=train.rows_for(arm,42)
            completed=checkpoint.load(ROOT/'runs/E_NEG_seed42/last.pth',model,optimizer,scheduler,None,identity,rows)
            initial_rng=checkpoint.rng_state(); state_file=ROOT/'runs/E_NEG_seed42/last.pth'
        else:
            optimizer,_=adaptation.optimizer_for(model);scheduler=adaptation.Schedule(optimizer)
            rows,_=train.rows_for(arm,42);completed=0;initial_rng=checkpoint.rng_state()
            state_file=ROOT/f'reports/performance_probe_{arm}.pth'
            checkpoint.save(state_file,model,optimizer,scheduler,None,0,{},rows[:8]);identity={}
        resources=runtime['resources'] if arm=='E_NEG' else runtime['f_resources']
        subset=cached if arm=='E_NEG' else {'DDR':cached['DDR']}
        n=20 if arm=='E_NEG' else 5
        baseline_datasets,_=train.datasets_for(runtime,arm)
        loader=train.loader_for(baseline_datasets,rows,completed,completed+n,resources,42)
        checkpoint.restore_rng(initial_rng);audit={};timing=[]
        for u in range(completed,completed+n):timing.append(train.update(model,optimizer,scheduler,loader,resources,u,audit,rows))
        baseline=snapshot(model,optimizer,scheduler,audit,timing);close(loader);del loader,baseline_datasets;gc.collect()
        checkpoint.load(state_file,model,optimizer,scheduler,None,identity,rows)
        loader=performance.loader_for(subset,rows,completed,completed+n+60,resources,42,config)
        checkpoint.restore_rng(initial_rng);audit={};fast=[]
        for u in range(completed,completed+n):fast.append(performance.update(model,optimizer,scheduler,loader,resources,u,audit,rows))
        accelerated=snapshot(model,optimizer,scheduler,audit,fast)
        equal={key:baseline[key]==accelerated[key] for key in baseline}
        if not all(equal.values()):
            write_json(ROOT/'reports/performance_equivalence_failure.json',{'arm':arm,'equal':equal,'baseline':baseline,'accelerated':accelerated})
            raise ValueError('Acceleration changed training state: '+str(equal))
        torch.use_deterministic_algorithms(False)
        steady=[];torch.cuda.reset_peak_memory_stats()
        for u in range(completed+n,completed+n+60):
            steady.append(performance.update(model,optimizer,scheduler,loader,resources,u,{},rows))
        results[arm]={'equivalence_updates':n,'exact_checks':equal,
            'baseline_update_mean_seconds':float(np.mean([x['seconds_per_update'] for x in timing])),
            'baseline_data_wait_mean_seconds':float(np.mean([x['data_wait_seconds'] for x in timing])),
            'optimized_steady_updates':60,'optimized_update_mean_seconds':float(np.mean([x['seconds_per_update'] for x in steady])),
            'optimized_update_p90_seconds':float(np.quantile([x['seconds_per_update'] for x in steady],.9)),
            'optimized_data_wait_mean_seconds':float(np.mean([x['data_wait_seconds'] for x in steady])),
            'peak_allocated_bytes':torch.cuda.max_memory_allocated()}
        print(results[arm],flush=True);close(loader);del loader,model,optimizer,scheduler;gc.collect();torch.cuda.empty_cache()
        if arm!='E_NEG':state_file.unlink()
    receipt={'status':'PASS','time':utc_now(),'cache':cache_receipt,'config':config,
             'performance_sha256':sha(ROOT/'performance.py'),'config_sha256':sha(ROOT/'configs/performance.json'),
             'arms':results,'formal_candidate':False,'val_or_test_read':False,'formal_checkpoint_modified':False,
             'equivalence_comparison_only_deterministic_CUDA':True,
             'throughput_and_formal_training_historical_CUDA_setting':True}
    write_json(ROOT/'reports/performance_acceptance.json',receipt)
    print('PERFORMANCE_PASS',flush=True)


if __name__=='__main__':main()
