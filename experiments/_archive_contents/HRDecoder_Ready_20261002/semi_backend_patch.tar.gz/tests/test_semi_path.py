"""Verify independent pseudo normalization, ignore classes and sequential gradients."""
import copy,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.build_pseudo_labels import consensus
from trainer.semi import SemiRunner
from trainer.engine import StageRunner
from trainer.loss import MaskedBinaryCEDice

def main():
    a=np.zeros((4,40,40),np.float32);b=a.copy();a[0,:2,:2]=.9;b[0,:2,:2]=.9
    a[1,0,0]=.9;b[1,0,0]=.2
    t,v=consensus(a,b,[True,True,False,False],np.random.RandomState(42))
    assert t[0].sum()==4 and (v[0]-t[0]).sum()==512
    assert t[1].sum()==0 and not v[2:].any() and np.all(t<=v)
    torch.manual_seed(42);model=torch.nn.Conv2d(3,4,1);ref=copy.deepcopy(model)
    loss_fn=MaskedBinaryCEDice()
    def batch(pseudo=False):
        target=torch.randint(0,2,(1,4,4,4)).float();valid=torch.ones_like(target)
        if pseudo:valid[:,3]=0
        return dict(image=torch.randn(1,3,4,4),target=target,valid=valid)
    true=[batch(),batch()];aux=[batch(True),batch(True)]
    opt=torch.optim.SGD(model.parameters(),lr=.1);other=torch.optim.SGD(ref.parameters(),lr=.1)
    cfg=dict(grad_accum=2,total_updates=3,seed=42,mixed_precision=False,warmup_updates=0,
             cosine_min_ratio=.01,grad_clip=None,log_every=0)
    runner=SemiRunner.__new__(SemiRunner)
    StageRunner.__init__(runner,model=model,dataset=None,loss_fn=loss_fn,optimizer=opt,cfg=cfg,device='cpu',loader=true)
    runner.pseudo_weight=.25;runner.pseudo_stream=iter(aux);runner.pseudo_crops=0
    runner.train_one_update_window(iter(true))
    for g,p in zip(true,aux):
        lg,_=loss_fn(ref(g['image']),g['target'],g['valid'])
        lp,_=loss_fn(ref(p['image']),p['target'],p['valid'])
        ((lg+.25*lp)/2).backward()
    other.step()
    assert runner.state.update==1 and runner.state.micro_step==2 and runner.pseudo_crops==2
    for x,y in zip(model.parameters(),ref.parameters()):torch.testing.assert_close(x,y,rtol=1e-6,atol=1e-7)
    print('Consensus, negative cap, class-ignore and independently normalized semi gradients passed.')

if __name__=='__main__':main()
