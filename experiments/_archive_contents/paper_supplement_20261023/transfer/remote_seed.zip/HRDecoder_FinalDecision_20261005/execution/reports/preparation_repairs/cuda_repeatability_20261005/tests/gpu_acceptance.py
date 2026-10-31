"""Mandatory native geometry/kernel/gradients/resume/100-step+fullval gates.

Acceptance uses comparison-only deterministic CUDA; formal runs retain the
historical CUDA settings. Benchmarks separately use formal settings, reset u0
and NEVER feed learned acceptance state into scientific training.
"""
import sys,time,copy,gc,tempfile,shutil
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from qlib.common import read_json,write_json,sha,utc_now,assert_source,digest
from qlib.loss import masked_loss
from qlib.geometry import center_logits
from experiment import ROOT,initialize,load_parent,rows_for,next_rows,val_dataset,identity_for
from inputs import cache_sources,loader_for,TJDR
from qlib.data import DDR
from schedule import optimizer_for,Schedule
from state import EMA,save,load
from engine import update
from spatial import CUDAAttention,reference_attention
from random_streams import forward_rng
from validation import evaluate,cache_p0
from kernel import load_kernel
from storage import check_storage_plan

def kernel_gate():
    load_kernel();shapes=[(3,4),(2,3),(1,2)]
    st=torch.tensor(shapes,device='cuda',dtype=torch.long);starts=torch.tensor([0,12,18],device='cuda',dtype=torch.long)
    v=torch.randn(2,20,3,8,device='cuda',requires_grad=True)
    locations=(torch.rand(2,7,3,3,4,2,device='cuda')*1.2-.1).requires_grad_()
    weights=torch.randn(2,7,3,3,4,device='cuda').flatten(-2).softmax(-1).reshape(2,7,3,3,4).requires_grad_()
    a=CUDAAttention.apply(v,st,starts,locations,weights);b=reference_attention(v,shapes,locations,weights)
    torch.testing.assert_close(a,b,atol=1e-5,rtol=1e-4)
    upstream=torch.randn_like(a)
    ga=torch.autograd.grad(a,(v,locations,weights),upstream,retain_graph=True);gb=torch.autograd.grad(b,(v,locations,weights),upstream)
    for x,y in zip(ga,gb):torch.testing.assert_close(x,y,atol=1e-4,rtol=1e-3)
    value=torch.arange(12,device='cuda',dtype=torch.float32).reshape(1,12,1,1)
    xy=torch.tensor([.125+.25,1/6],device='cuda').reshape(1,1,1,1,1,2)
    out=CUDAAttention.apply(value,torch.tensor([[3,4]],device='cuda'),torch.zeros(1,device='cuda',dtype=torch.long),xy,torch.ones(1,1,1,1,1,device='cuda'))
    torch.testing.assert_close(out,torch.ones_like(out),atol=1e-5,rtol=0)
    return {'status':'PASS','forward_atol':1e-5,'forward_rtol':1e-4,'gradient_atol':1e-4,'gradient_rtol':1e-3,'xy_one_value_pixel':'PASS'}

def snapshot(model,opt,sched,ema):
    # On-device exact comparisons avoid cross-platform tensor fingerprints.
    return {name:t.detach().clone() for name,t in model.state_dict().items()},copy.deepcopy(opt.state_dict()),copy.deepcopy(sched.state_dict()),{n:t.clone() for n,t in ema.model.state_dict().items()}

def compare_snapshot(a,b):
    def eq(x,y):
        if torch.is_tensor(x):assert torch.equal(x,y)
        elif isinstance(x,dict):assert x.keys()==y.keys();[eq(x[k],y[k]) for k in x]
        elif isinstance(x,(list,tuple)):assert len(x)==len(y);[eq(v,w) for v,w in zip(x,y)]
        else:assert x==y
    eq(a,b)

def small_model_gates(runtime,seed=42):
    parent,_=load_parent(runtime,seed);parent=parent.cuda().eval();image=torch.randn(1,3,1024,1024,device='cuda')
    with torch.no_grad(),torch.autocast('cuda',enabled=False):expected=parent(image)
    result={}
    for arm in ('C','S','D','SD'):
        model,_=initialize(runtime,arm,seed);model.eval()
        with torch.no_grad(),torch.autocast('cuda',enabled=False):actual=model(image)
        delta=float((expected-actual).abs().max());assert delta<=1e-5,(arm,delta)
        stable=expected.abs()>1e-5;assert torch.equal((actual>0)[stable],(expected>0)[stable])
        del actual
        # Full trainable graph on small geometry verifies strict full-model
        # resume and source routing without expensive native benchmarking twice.
        model.train();x=image[:,:,:64,:64].contiguous();g=torch.zeros(1,4,64,64,device='cuda');g[:,:,24:40,24:40]=1;v=torch.ones_like(g)
        opt,_=optimizer_for(model);sched=Schedule(opt);ema=EMA(model)
        def step(u):
            opt.zero_grad(set_to_none=True)
            with forward_rng(seed,'DDR',u,0,x.device),torch.autocast('cuda',dtype=torch.bfloat16):z=model(x)
            loss=masked_loss(z,g,v);loss.backward()
            if u==1 and model.has_spatial:
                for interaction in model.spatial.interactions:assert torch.isfinite(interaction.injector.output.weight.grad).all() and interaction.injector.output.weight.grad.abs().max()>0
                for proj in model.ddr.spatial_residual:assert torch.isfinite(proj.weight.grad).all() and proj.weight.grad.abs().max()>0
            if u==2 and model.has_spatial:
                names=[n for n,p in model.spatial.named_parameters() if p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().max()>0]
                assert any(n.startswith('pyramid.') for n in names) and any('extractor.' in n for n in names) and any('injector.value.' in n for n in names)
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);assert torch.isfinite(norm)
            opt.step();sched.step();ema.update(model)
        before=model.backbone.blocks[-1].attn.qkv.weight.clone();step(1);assert not torch.equal(before,model.backbone.blocks[-1].attn.qkv.weight)
        identity={'arm':arm,'acceptance_only':True};temporary=ROOT/'reports'/f'acceptance_resume_{arm}.pth';nextrows=lambda u:{'DDR':u,'TJDR':u if model.has_source else 0}
        save(temporary,model,opt,sched,ema,1,identity,nextrows(1));step(2);continuous=snapshot(model,opt,sched,ema)
        completed=load(temporary,model,opt,sched,ema,identity,nextrows);assert completed==1;step(2)
        compare_snapshot(continuous,snapshot(model,opt,sched,ema));temporary.unlink()
        if model.has_source:
            opt.zero_grad(set_to_none=True)
            with forward_rng(seed,'TJDR',3,0,x.device),torch.autocast('cuda',dtype=torch.bfloat16):z=model(x,'TJDR')
            masked_loss(z,g,v).backward()
            assert all(p.grad is None for p in model.ddr.parameters())
            assert any(p.grad is not None and p.grad.abs().max()>0 for p in model.tjdr.parameters())
            assert any(p.grad is not None and p.grad.abs().max()>0 for p in model.backbone.parameters())
        result[arm]={'native1024_zero_step_max_abs_fp32_logit':delta,'full_old_new_gradients':'PASS','full_model_EMA_optimizer_rng_resume':'PASS','source_route':'PASS' if model.has_source else 'not_applicable'}
        del model,opt,sched,ema,continuous;gc.collect();torch.cuda.empty_cache()
    del parent,image,expected;gc.collect();torch.cuda.empty_cache()
    # DDR gradients in C and D with source disabled: identical seeded backbone.
    models=[];grads=[]
    for arm in ('C','D'):
        model,_=initialize(runtime,arm,seed);model.train();x=torch.ones(1,3,64,64,device='cuda')*.1
        with forward_rng(seed,'DDR',1,0,x.device),torch.autocast('cuda',dtype=torch.bfloat16):z=model(x)
        masked_loss(z,torch.zeros_like(z),torch.ones_like(z)).backward()
        grads.append({n:p.grad.detach().cpu() for n,p in model.named_parameters() if p.grad is not None});del model;gc.collect();torch.cuda.empty_cache()
    compare_snapshot(grads[0],grads[1]);result['C_D_source_disabled_DDR_gradients']='EXACT_PASS'
    return result

def early():
    runtime_path=ROOT/'configs/runtime.server.json';runtime=read_json(runtime_path)
    file=ROOT/'reports/early_gpu_gates.json';assert not file.exists(),'Do not duplicate completed or active early gates'
    result={'status':'RUNNING','time':utc_now(),'source_sha256':assert_source(ROOT)['sha256'],'runtime_sha256':sha(runtime_path)}
    write_json(file,result)
    result['kernel']=kernel_gate()
    previous=(torch.backends.cudnn.deterministic,torch.backends.cudnn.benchmark,torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32)
    try:
        torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        result['model_gates']=small_model_gates(runtime)
    finally:
        torch.backends.cudnn.deterministic,torch.backends.cudnn.benchmark,torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32=previous
    result.update(status='PASS',completed_at=utc_now(),learned_states_discarded=True);write_json(file,result)

def main():
    runtime_path=ROOT/'configs/runtime.server.json';runtime=read_json(runtime_path)
    assert read_json(ROOT/'reports/TJDR_manifest.json')['status']=='READY'
    assert read_json(ROOT/'reports/cpu_acceptance.json')['status']=='PASS'
    assert read_json(ROOT/'reports/window_acceptance.json')['status']=='PASS'
    check_storage_plan()
    result={'status':'RUNNING','time':utc_now(),'source_sha256':assert_source(ROOT)['sha256'],'runtime_sha256':sha(runtime_path)}
    write_json(ROOT/'reports/gpu_readiness.json',result)
    from ops.verify_system import main as verify_system
    result['system']=verify_system(runtime)
    file=ROOT/'reports/early_gpu_gates.json'
    if not file.exists():early()
    gates=read_json(file);assert gates['status']=='PASS' and gates['source_sha256']==result['source_sha256'] and gates['runtime_sha256']==result['runtime_sha256']
    result.update(kernel=gates['kernel'],model_gates=gates['model_gates'],early_gate_reused_without_repeating=True)
    datasets={'DDR':DDR(runtime['ddr_root'],'train',read_json(ROOT/'reports/data_identity.json')['splits']['train']),
              'TJDR':TJDR(runtime['TJDR_root'],read_json(ROOT/'reports/TJDR_manifest.json'))}
    cached,cache_receipt=cache_sources(datasets,runtime['resources']['shared_cache_limit_bytes'],lambda p:print('cache',p,flush=True))
    val=val_dataset(runtime);p0,p0_receipt=cache_p0(val,runtime);result['caches']={'training':cache_receipt,'P0':p0_receipt}
    parent,_=load_parent(runtime,42);parent=parent.cuda();metrics=evaluate(parent,val,runtime,ROOT/'reports/parent_restore_val42',12000,p0_ram=p0)
    historical=read_json(Path(runtime['parent_f12_execution'])/'runs/F12_seed42/val_u12000/metrics.json')
    assert metrics['single']['counts']==historical['single']['counts'],'Restored historical F12 counts differ'
    write_json(ROOT/'reports/parent_validation_seed42.json',metrics);del parent;gc.collect();torch.cuda.empty_cache()
    result['benchmarks']={}
    for arm in ('C','S','D','SD'):
        model,_=initialize(runtime,arm,42);model.train();opt,_=optimizer_for(model);sched=Schedule(opt);ema=EMA(model)
        loaders={'DDR':loader_for(cached,rows_for('DDR',42),'DDR',0,100,runtime['resources'],42)}
        if model.has_source:loaders['TJDR']=loader_for(cached,rows_for('TJDR',42),'TJDR',0,100,runtime['resources'],42)
        torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter();wait=0
        for u in range(1,101):stat=update(model,opt,sched,ema,loaders,42,u-1);wait+=stat['data_wait_seconds']
        torch.cuda.synchronize();elapsed=time.perf_counter()-start
        timing=evaluate(ema.model,val,runtime,ROOT/f'reports/timing_{arm}_val',100,p0_ram=p0)
        identity={'arm':arm,'acceptance_only':True};path=ROOT/f'reports/timing_{arm}_checkpoint.pth';before=time.perf_counter()
        save(path,model,opt,sched,ema,100,identity,{});save_seconds=time.perf_counter()-before;checkpoint_bytes=path.stat().st_size;path.unlink()
        result['benchmarks'][arm]={'complete_updates':100,'seconds_per_update':elapsed/100,'data_wait_seconds_per_update':wait/100,
            'full_val_seconds':timing['seconds'],'checkpoint_seconds':save_seconds,'checkpoint_bytes':checkpoint_bytes,'peak_gpu_bytes':torch.cuda.max_memory_allocated()}
        write_json(ROOT/'reports/gpu_readiness.json',result);print(arm,result['benchmarks'][arm],flush=True)
        del model,opt,sched,ema,loaders;gc.collect();torch.cuda.empty_cache()
    result.update(status='READY',completed_at=utc_now(),benchmarks_used_for_scientific_initialization=False,formal_CUDA_settings_restored=True)
    write_json(ROOT/'reports/gpu_readiness.json',result)

if __name__=='__main__':main()
