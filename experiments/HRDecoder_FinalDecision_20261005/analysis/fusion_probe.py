"""Retrospective VAL-only cross-fit diagnostic. Never a new test-selected winner."""
from pathlib import Path
import json, time, hashlib, sys
import numpy as np
from PIL import Image

EXPERT=sys.argv[1] if len(sys.argv)>1 else 'F4'
assert EXPERT in ['F4','F12']
OUT=Path(__file__).resolve().parent
if EXPERT!='F4':
    OUT=OUT/EXPERT
    OUT.mkdir(exist_ok=True)
OLD=Path(r'D:\dataset\experiment\HRDecoder_Next_20261003')
Q=Path(r'D:\dataset\experiment\HRDecoder_Q_20261003\server_results\runs\Q_DDR_seed42\val_u16000')
F=Path(r'D:\dataset\experiment\HRDecoder_NextPlan_20261004\execution\server_results\runs')/f'{EXPERT}_seed42/val_u12000'
LAB=Path(r'D:\HRDecoder\data\DDR_split_613\labels\val')
N=128

def read(p):
    with np.load(p,allow_pickle=False) as z: a=z['prob']
    assert a.dtype==np.float32 and np.isfinite(a).all()
    return a

def inputs(s):
    c=read(OLD/'server_results/work/G/champion'/f'{s}.npz')
    u=read(OLD/'server_results/work/G/U0_native_whole'/f'{s}.npz')
    p=.75*c+.25*u
    r=.75*p+.25*read(Q/f'{s}.npz')
    f=read(F/f'{s}.npz')
    y=np.asarray(Image.open(LAB/f'{s}.png'))
    assert r.shape==f.shape==(4,*y.shape)
    return r,f,y

def counts(b,g):
    return np.array([np.count_nonzero(b&g),np.count_nonzero(b&~g),np.count_nonzero(~b&g)],dtype=np.int64)

def metric(c):
    v=100*c[:,0]/np.maximum(c.sum(1),1)
    return dict(counts=c.tolist(),iou_pct=v.tolist(),miou_pct=float(v.mean()))

def main():
    t0=time.time()
    ids=json.loads((OLD/'configs/val_contract.json').read_text(encoding='utf-8-sig'))['stems']
    order=sorted(range(len(ids)),key=lambda i: hashlib.sha256(('fusion_probe_20261005:'+ids[i]).encode()).hexdigest())
    folds=np.zeros(len(ids),dtype=np.int64)
    for k,i in enumerate(order): folds[i]=k%4
    weights=[0,.125,.25,.5,.75,1]
    thresholds=[float(x) for x in np.round(np.arange(.1,.901,.025),3)]
    proto=dict(status='RETROSPECTIVE_DIAGNOSTIC_NOT_DEPLOYMENT',split='val',n=76,folds=4,
       base='R42=.75*(.75*C+.25*U0)+.25*Q_DDR42_16k',expert=f'{EXPERT}_42_12k',
       weights=weights,thresholds=thresholds,histogram_bins=N,
       purpose='Check whether low-complexity class-wise reweighting/calibration can plausibly close the gap; no new network or GPU',
       limits=['Models already selected on these val images; meta-level folds do not undo model selection bias.',
               'Image folds; patient identity unavailable.','No F12 subset used for full-val conclusions.'],
       final_scoring='Exact original FP32 arrays; binning used for fit only',test_read=False,
       fold_assignment={s:int(folds[i]) for i,s in enumerate(ids)})
    (OUT/'fusion_probe_protocol.json').write_text(json.dumps(proto,indent=2),encoding='utf-8')
    hist=np.zeros((76,4,2,N*N),dtype=np.int64)
    baseline=np.zeros((76,4,3),dtype=np.int64)
    for i,s in enumerate(ids):
        r,f,y=inputs(s)
        for c in range(4):
            g=y==c+1; baseline[i,c]=counts(r[c]>.5,g)
            a=np.minimum((r[c]*N).astype(np.int32),N-1)
            b=np.minimum((f[c]*N).astype(np.int32),N-1)
            k=(a*N+b).ravel(); truth=g.ravel()
            hist[i,c,1]=np.bincount(k[truth],minlength=N*N)
            hist[i,c,0]=np.bincount(k[~truth],minlength=N*N)
        if (i+1)%10==0: print('fit histogram',i+1,flush=True)
    np.savez_compressed(OUT/'fusion_probe_histograms.npz',hist=hist,baseline=baseline,folds=folds)
    x=(np.arange(N)+.5)/N
    # Tie order favors smaller expert weight and threshold closest to .5.
    params=[(w,t) for w in weights for t in sorted(thresholds,key=lambda z:(abs(z-.5),z))]
    rules=np.stack([((1-w)*x[:,None]+w*x[None,:]>t).ravel() for w,t in params]).astype(np.int64)
    chosen={}; oof_params=np.zeros((76,4,2))
    for fold in range(4):
        h=hist[folds!=fold].sum(0); selected=[]
        for c in range(4):
            tp=rules@h[c,1]; fp=rules@h[c,0]; score=tp/np.maximum(h[c,1].sum()+fp,1)
            j=int(np.argmax(score)); selected.append(params[j]); oof_params[folds==fold,c]=params[j]
        chosen[str(fold)]=selected
    # Small, exact second pass; the held-out image never selects its own parameters.
    oof=np.zeros_like(baseline)
    for i,s in enumerate(ids):
        r,f,y=inputs(s)
        for c in range(4):
            w,t=oof_params[i,c]; pred=((1-w)*r[c]+w*f[c])>t
            oof[i,c]=counts(pred,y==c+1)
        if (i+1)%10==0: print('exact holdout',i+1,flush=True)
    rng=np.random.default_rng(5102026); delta=[]
    for _ in range(2000):
        ii=rng.integers(0,76,76)
        delta.append(metric(oof[ii].sum(0))['miou_pct']-metric(baseline[ii].sum(0))['miou_pct'])
    res=dict(protocol=proto,base=metric(baseline.sum(0)),crossfit=metric(oof.sum(0)),
       delta_pp=metric(oof.sum(0))['miou_pct']-metric(baseline.sum(0))['miou_pct'],
       conditional_image_bootstrap95_pp=np.percentile(delta,[2.5,97.5]).tolist(),
       chosen_per_fold=chosen,per_image_base=baseline.tolist(),per_image_oof=oof.tolist(),seconds=time.time()-t0,
       no_model_winner_selected=True,bootstrap_caveat='Parameters fixed in bootstrap, excludes refitting, historical selection and seed variation.')
    (OUT/'fusion_probe_result.json').write_text(json.dumps(res,indent=2),encoding='utf-8')
    print(json.dumps({k:res[k] for k in ['base','crossfit','delta_pp','conditional_image_bootstrap95_pp','chosen_per_fold','seconds']}),flush=True)

if __name__=='__main__':main()
