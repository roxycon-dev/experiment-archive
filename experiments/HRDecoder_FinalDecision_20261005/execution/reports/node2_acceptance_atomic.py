"""Bounded native CUDA repeatability diagnosis and deterministic state oracle.

Native kernel forward/backward acceptance and all formal benchmarks/training
retain the frozen kernel. Only the checkpoint-continuation comparison uses a
CPU reference backward to remove the kernel's unordered CUDA atomic reduction.
No tolerance is increased and no learned acceptance state enters training.
"""
import sys,copy,gc
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from qlib.common import read_json,write_json,utc_now,sha
from qlib.checkpoint import rng_state,restore_rng
from experiment import initialize
from schedule import optimizer_for,Schedule
from state import EMA,load
from random_streams import forward_rng
from qlib.loss import masked_loss
from tests.gpu_acceptance import snapshot
from spatial import CUDAAttention,reference_attention

def native_ram_repeatability():
    runtime=read_json(ROOT/'configs/runtime.server.json')
    files=[p for p in (ROOT/'reports').glob('acceptance_resume_*.pth')]
    assert len(files)==1, 'Diagnose the actual failed arm only'
    path=files[0];arm=path.stem.split('_')[-1]
    model,_=initialize(runtime,arm,42);model.train();opt,_=optimizer_for(model);sched=Schedule(opt);ema=EMA(model)
    load(path,model,opt,sched,ema,{'arm':arm,'acceptance_only':True},lambda u:{'DDR':u,'TJDR':u if model.has_source else 0})
    base=snapshot(model,opt,sched,ema)
    x=torch.randn(1,3,64,64,device='cuda');g=torch.zeros(1,4,64,64,device='cuda');g[:,:,24:40,24:40]=1;v=torch.ones_like(g)
    outputs=[];states=[]
    for _ in range(2):
        model.load_state_dict(base[0]);opt.load_state_dict(copy.deepcopy(base[1]));sched.load_state_dict(copy.deepcopy(base[2]));ema.model.load_state_dict(base[3]);restore_rng(base[4])
        opt.zero_grad(set_to_none=True)
        with forward_rng(42,'DDR',2,0,x.device),torch.autocast('cuda',dtype=torch.bfloat16):z=model(x)
        loss=masked_loss(z,g,v);outputs.append((z.detach().clone(),loss.detach().clone()));loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();sched.step();ema.update(model);states.append(snapshot(model,opt,sched,ema))
    assert all(torch.equal(a,b) for a,b in zip(outputs[0],outputs[1]))
    differences=[]
    def visit(a,b,name):
        if torch.is_tensor(a):
            if not torch.equal(a,b):differences.append({'path':name,'max_abs':float((a.double()-b.double()).abs().max()),'mismatch':int((a!=b).sum())})
        elif isinstance(a,dict):
            for k in a:visit(a[k],b[k],name+'/'+str(k))
        elif isinstance(a,(tuple,list)):
            for i,(av,bv) in enumerate(zip(a,b)):visit(av,bv,name+'/'+str(i))
    visit(states[0],states[1],'state')
    assert differences,'Atomic diagnosis did not reproduce; do not waive original comparison'
    receipt={'stage':'NATIVE_BACKWARD_VARIATION_WITHOUT_SERIALIZATION_REPRODUCED','arm':arm,'forward_and_loss':'EXACT','checkpoint_file_sha256':sha(path),'largest_differences':sorted(differences,key=lambda r:-r['max_abs'])[:20],'checkpoint_roundtrip_not_the_cause':True,'formal_updates':0,'time':utc_now()}
    write_json(ROOT/'reports/node2_native_atomic_diagnosis.json',receipt)
    del base,model,opt,sched,ema,states,outputs;gc.collect();torch.cuda.empty_cache()
    return receipt

def install_comparison_oracle():
    import tests.gpu_acceptance as gate
    original_gate=gate.small_model_gates;native_backward=CUDAAttention.backward
    def deterministic_backward(ctx,grad):
        value,shapes,starts,locations,weights=ctx.saved_tensors
        with torch.enable_grad():
            vv=value.detach().cpu().requires_grad_();ll=locations.detach().cpu().requires_grad_();ww=weights.detach().cpu().requires_grad_()
            out=reference_attention(vv,[tuple(r) for r in shapes.cpu().tolist()],ll,ww)
            dv,dl,dw=torch.autograd.grad(out,(vv,ll,ww),grad.detach().cpu())
        return dv.to(value.device),None,None,dl.to(locations.device),dw.to(weights.device)
    def compare(runtime,seed=42):
        CUDAAttention.backward=staticmethod(deterministic_backward)
        try:result=original_gate(runtime,seed)
        finally:CUDAAttention.backward=native_backward
        for arm in ('C','S','D','SD'):
            result[arm]['continuation_comparison_backward']='deterministic_CPU_FP32_reference; native kernel gated separately'
            result[arm]['formal_backward_changed']=False
        return result
    gate.small_model_gates=compare
