"""Bounded acceptance-only Flash/MATH repeatability diagnosis; no scientific run."""
import os,sys,copy,gc
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import torch
from torch.nn.attention import sdpa_kernel,SDPBackend
from qlib.common import read_json,write_json,utc_now
from experiment import initialize,load_parent
from schedule import optimizer_for,Schedule
from state import EMA,load
from random_streams import forward_rng
from qlib.loss import masked_loss
from tests.gpu_acceptance import snapshot

def differences(a,b):
    rows=[]
    def visit(x,y,path):
        if torch.is_tensor(x):
            if not torch.equal(x,y):
                d=(x.double()-y.double()).abs()
                rows.append({'path':path,'shape':list(x.shape),'dtype':str(x.dtype),'max_abs':float(d.max()),'mismatch_values':int((d>0).sum()),'max_reference_abs':float(x.abs().max())})
        elif isinstance(x,dict):
            assert x.keys()==y.keys()
            for k in x:visit(x[k],y[k],path+'/'+str(k))
        elif isinstance(x,(list,tuple)):
            for k,(v,w) in enumerate(zip(x,y)):visit(v,w,path+'/'+str(k))
        else:assert x==y
    visit(a,b,'snapshot');return rows

def main():
    rt=read_json(ROOT/'configs/runtime.server.json');result={}
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    parent,_=load_parent(rt,42);image=torch.randn(1,3,1024,1024,device='cuda');del parent
    x=image[:,:,:64,:64].contiguous();g=torch.zeros(1,4,64,64,device='cuda');g[:,:,24:40,24:40]=1;v=torch.ones_like(g)
    model,_=initialize(rt,'C',42);model.train();opt,_=optimizer_for(model);sched=Schedule(opt);ema=EMA(model)
    identity={'arm':'C','acceptance_only':True};file=ROOT/'reports/acceptance_resume_C.pth';nextrows=lambda u:{'DDR':u,'TJDR':0}
    for mode in (SDPBackend.FLASH_ATTENTION,SDPBackend.MATH):
        outcomes=[]
        for repeat in range(2):
            load(file,model,opt,sched,ema,identity,nextrows);opt.zero_grad(set_to_none=True)
            with sdpa_kernel(mode):
                with forward_rng(42,'DDR',2,0,x.device),torch.autocast('cuda',dtype=torch.bfloat16):z=model(x)
                masked_loss(z,g,v).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();sched.step();ema.update(model)
            outcomes.append(snapshot(model,opt,sched,ema))
        mismatch=differences(*outcomes);result[str(mode)]={'mismatch_tensors':len(mismatch),'examples':mismatch[:12],'global_max_abs':max((r['max_abs'] for r in mismatch),default=0)}
        print(mode,result[str(mode)],flush=True);del outcomes;gc.collect();torch.cuda.empty_cache()
    write_json(ROOT/'reports/cuda_resume_diagnosis.json',{'time':utc_now(),'comparison_only_no_scientific_updates':True,'results':result})

if __name__=='__main__':main()
