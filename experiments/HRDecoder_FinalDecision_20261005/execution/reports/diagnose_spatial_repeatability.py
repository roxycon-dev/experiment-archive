"""Bounded precision diagnosis outside frozen experiment source; synthetic only."""
import os,sys,gc
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import torch,numpy as np
from qlib.common import read_json,write_json,utc_now
from experiment import initialize,load_parent
from schedule import optimizer_for,Schedule
from state import EMA,load
from random_streams import forward_rng
from qlib.loss import masked_loss
from tests.gpu_acceptance import snapshot
from kernel import load_kernel

def differences(a,b):
    rows=[]
    def visit(x,y,path):
        if torch.is_tensor(x):
            if not torch.equal(x,y):
                d=(x.double()-y.double()).abs();rows.append({'path':path,'shape':list(x.shape),'max_abs':float(d.max()),'mismatch_values':int((d>0).sum()),'max_reference_abs':float(x.abs().max())})
        elif isinstance(x,np.ndarray):assert np.array_equal(x,y)
        elif isinstance(x,dict):
            assert x.keys()==y.keys()
            for k in x:visit(x[k],y[k],path+'/'+str(k))
        elif isinstance(x,(tuple,list)):
            for k,(v,w) in enumerate(zip(x,y)):visit(v,w,path+'/'+str(k))
        else:assert x==y
    visit(a,b,'snapshot');return rows

def main():
    rt=read_json(ROOT/'configs/runtime.server.json');result={};load_kernel()
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    parent,_=load_parent(rt,42);image=torch.randn(1,3,1024,1024,device='cuda');del parent
    x=image[:,:,:64,:64].contiguous();g=torch.zeros(1,4,64,64,device='cuda');g[:,:,24:40,24:40]=1;v=torch.ones_like(g)
    model,_=initialize(rt,'S',42);model.train();opt,_=optimizer_for(model);sched=Schedule(opt);ema=EMA(model)
    identity={'arm':'S','acceptance_only':True};file=ROOT/'reports/acceptance_resume_S.pth';nextrows=lambda u:{'DDR':u,'TJDR':0}
    assert file.exists(),'Locate failed arm instead of assuming one'
    for amp in (True,False):
        outcomes=[];logits=[]
        for repeat in range(2):
            load(file,model,opt,sched,ema,identity,nextrows);opt.zero_grad(set_to_none=True)
            with forward_rng(42,'DDR',2,0,x.device):
                with torch.autocast('cuda',dtype=torch.bfloat16,enabled=amp):z=model(x)
                logits.append(z.detach().clone());masked_loss(z,g,v).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();sched.step();ema.update(model)
            outcomes.append(snapshot(model,opt,sched,ema))
        mismatch=differences(*outcomes)
        result['BF16' if amp else 'FP32']={'forward_exact':torch.equal(*logits),'forward_max_abs':float((logits[0]-logits[1]).abs().max()),
            'mismatch_tensors':len(mismatch),'largest_differences':sorted(mismatch,key=lambda r:-r['max_abs'])[:10],
            'global_max_abs':max((r['max_abs'] for r in mismatch),default=0)}
        print('BF16' if amp else 'FP32',result['BF16' if amp else 'FP32'],flush=True);del outcomes,logits;gc.collect();torch.cuda.empty_cache()
    write_json(ROOT/'reports/spatial_repeatability_diagnosis.json',{'time':utc_now(),'synthetic_only_no_scientific_updates':True,'results':result})

if __name__=='__main__':main()
