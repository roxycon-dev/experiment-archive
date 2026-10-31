"""Selection, stop boundaries and complete recoverable state, without GPU fees."""
import sys,copy,tempfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,torch
from qlib.common import write_json,utc_now
from qlib.checkpoint import rng_state
from state import EMA,save,load,cursor
from schedule import Schedule
from decision import choose
from train_run import review_trigger
ROOT=Path(__file__).resolve().parents[1]

def equal(a,b):
    if torch.is_tensor(a):assert torch.equal(a,b)
    elif isinstance(a,np.ndarray):assert np.array_equal(a,b)
    elif isinstance(a,dict):assert a.keys()==b.keys();[equal(a[k],b[k]) for k in a]
    elif isinstance(a,(list,tuple)):assert len(a)==len(b);[equal(x,y) for x,y in zip(a,b)]
    else:assert a==b

def main():
    refs={'42':40.,'2024':40.}
    bad={'arm':'SD','update':32000,'formula':'single','scores':{'42':49.,'2024':39.}}
    good={'arm':'S','update':20000,'formula':'fixed_fusion','scores':{'42':40.25,'2024':40.75}}
    assert choose([copy.deepcopy(bad),copy.deepcopy(good)],refs)['arm']=='S','Eligibility precedes mean ranking'
    assert choose([{'arm':'C','update':20000,'formula':'single','scores':{'42':40.,'2024':41.}}],refs) is None
    tied=[dict(good,arm='SD',update=32000,formula='half_fusion'),dict(good,arm='C',formula='single'),good]
    assert choose(copy.deepcopy(tied),refs)['formula']=='single'
    def node(single,fusion):return {'single':{'miou_pct':single},'fixed_fusion':{'miou_pct':fusion}}
    assert not review_trigger({4000:node(37,38.5)},40,40)
    assert review_trigger({4000:node(37,38.5),8000:node(37,38.5)},40,40)
    assert not review_trigger({4000:node(37,38.5),8000:node(37.001,38.5)},40,40)
    assert cursor(20001,'SD')=={'DDR':160008,'TJDR':80000}
    assert cursor(32000,'C')=={'DDR':256000,'TJDR':0}
    torch.manual_seed(51005);model=torch.nn.Sequential(torch.nn.Linear(5,8),torch.nn.GELU(),torch.nn.Linear(8,4))
    opt=torch.optim.AdamW([{'params':model.parameters(),'peak_lr':1e-4}],lr=1e-4);sched=Schedule(opt);ema=EMA(model)
    x=torch.randn(3,5);target=torch.randn(3,4)
    def step():
        opt.zero_grad(set_to_none=True);loss=(model(x)-target).square().mean();loss.backward();opt.step();sched.step();ema.update(model)
    step();identity={'arm':'SD','seed':42};nextrows=lambda u:{'DDR':[u]*8,'TJDR':[u]*4}
    with tempfile.TemporaryDirectory(dir=ROOT/'reports') as temp:
        path=Path(temp)/'state.pth';save(path,model,opt,sched,ema,1,identity,nextrows(1));saved_rng=rng_state()
        step();expected=(copy.deepcopy(model.state_dict()),copy.deepcopy(opt.state_dict()),copy.deepcopy(ema.model.state_dict()),rng_state())
        assert load(path,model,opt,sched,ema,identity,nextrows)==1;equal(saved_rng,rng_state());step()
        equal(expected,(model.state_dict(),opt.state_dict(),ema.model.state_dict(),rng_state()))
        wrong=dict(identity,seed=2024)
        try:load(path,model,opt,sched,ema,wrong,nextrows)
        except AssertionError:pass
        else:raise AssertionError('Resume accepted wrong identity')
    write_json(ROOT/'reports/protocol_acceptance.json',{'status':'PASS','time':utc_now(),
        'eligibility_before_rank':True,'exact_ties':True,'stop_two_nodes_inclusive_boundary':True,
        'source_exit_cursor':True,'optimizer_scheduler_EMA_rng_exact_roundtrip':True,'wrong_identity_rejected':True})
    print('PROTOCOL_ACCEPTANCE_PASS')

if __name__=='__main__':main()
