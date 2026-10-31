"""Constrained fusion from fp32, original-coordinate val probability caches.

Up to two new model families. Per-class supplement alpha in {0,.125,.25,.5}.
Count every image once per option; selection and bootstrap are then cheap.
All OOF metrics aggregate TP/FP/FN. Ties within .0005 prefer less new weight.
Final vote uses only the three fold choices, never rescans full val weights.
"""
import argparse
import json
from collections import Counter
from pathlib import Path
import numpy as np
from PIL import Image

ALPHAS=(0.,.125,.25,.5)
CLASSES=('EX','HE','SE','MA')

def iou(counts):
    counts=np.asarray(counts,dtype=np.float64)
    den=counts.sum(axis=-1)
    return np.divide(counts[...,0],den,out=np.full_like(den,np.nan),where=den>0)

def choose(scores,options,tolerance=.0005):
    best=float(np.nanmax(scores))
    ids=[i for i,s in enumerate(scores) if np.isfinite(s) and s>=best-tolerance]
    return min(ids,key=lambda i:(options[i]['alpha'],options[i]['family'] or ''))

def crossfit(counts,options,folds):
    # counts: option x image x class x (TP,FP,FN)
    selected=[];global_selected=[]
    held=[];global_held=[];base=[]
    for k in range(3):
        fit=folds!=k;test=folds==k
        scores=iou(counts[:,fit].sum(axis=1))
        picks=[choose(scores[:,c],options) for c in range(4)]
        g=choose(np.nanmean(scores,axis=1),options)
        selected.append([options[i] for i in picks]);global_selected.append(options[g])
        held.append(np.stack([counts[i,test,c].sum(axis=0) for c,i in enumerate(picks)]))
        global_held.append(counts[g,test].sum(axis=0));base.append(counts[0,test].sum(axis=0))
    deploy=[]
    for c in range(4):
        votes=[row[c]['family'] for row in selected];winner,n=Counter(votes).most_common(1)[0]
        if winner is None or n<2:
            deploy.append(dict(family=None,alpha=0.));continue
        ws=sorted(row[c]['alpha'] for row in selected if row[c]['family']==winner)
        weight=float(ws[(len(ws)-1)//2])  # lower median preserves the allowed grid
        deploy.append(dict(family=winner,alpha=weight))
    class_counts=sum(held);global_counts=sum(global_held);base_counts=sum(base)
    # Instability: no majority for any selected supplemental family, or a
    # weight range > .25. Fall back to a majority-voted global blend.
    unstable=any(len(set(row[c]['family'] for row in selected))==3 or
                 np.ptp([row[c]['alpha'] for row in selected])>.25 for c in range(4))
    if unstable or np.nanmean(iou(class_counts))<=np.nanmean(iou(global_counts))+.0005:
        gv=[o['family'] for o in global_selected];family,n=Counter(gv).most_common(1)[0]
        ws=sorted(o['alpha'] for o in global_selected if o['family']==family)
        alpha=float(ws[(len(ws)-1)//2]) if n>=2 and family else 0.
        if alpha==0:family=None
        deploy=[dict(family=family,alpha=alpha) for _ in range(4)]
        selected_mode='global_fallback'
    else:selected_mode='per_class'
    return dict(fold_choices=selected,global_fold_choices=global_selected,
        oof_base_iou=iou(base_counts).tolist(),oof_global_iou=iou(global_counts).tolist(),
        oof_class_iou=iou(class_counts).tolist(),oof_base_miou=float(np.nanmean(iou(base_counts))),
        oof_global_miou=float(np.nanmean(iou(global_counts))),oof_class_miou=float(np.nanmean(iou(class_counts))),
        oof_class_counts=class_counts.tolist(),oof_global_counts=global_counts.tolist(),
        deploy=deploy,selected_mode=selected_mode,unstable=unstable,
        threshold=.5,strict_greater=True,full_val_rescan=False)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--folds',required=True,type=Path)
    ap.add_argument('--champion',required=True,type=Path);ap.add_argument('--candidate',action='append',default=[])
    ap.add_argument('--labels',required=True,type=Path);ap.add_argument('--out',required=True,type=Path)
    a=ap.parse_args()
    if len(a.candidate)>2:raise ValueError('At most two nominated new families')
    fold_info=json.loads(a.folds.read_text(encoding='utf-8'));ids=sorted(fold_info['folds'])
    if fold_info.get('split')!='val' or len(ids)!=76:raise ValueError('Require fixed 76-image val folds')
    options=[dict(family=None,alpha=0.)];paths={}
    for raw in a.candidate:
        name,path=raw.split('=',1)
        if name in paths:raise ValueError('Each new family may be nominated only once')
        paths[name]=Path(path)
        options += [dict(family=name,alpha=alpha) for alpha in ALPHAS[1:]]
    identities={}
    for name,path in [('champion',a.champion)]+list(paths.items()):
        idx=json.loads((path/'index.json').read_text(encoding='utf-8'))
        if (idx.get('split')!='val' or idx.get('dtype')!='float32' or
            idx.get('coordinates')!='original_image' or idx.get('channel_order')!=list(CLASSES) or
            sorted(s['sample_id'] for s in idx.get('samples',[]))!=ids):
            raise ValueError(f'Cache identity mismatch: {name}')
        if not idx.get('checkpoint_sha256'):raise ValueError(f'Missing checkpoint identity: {name}')
        identities[name]=idx
    counts=np.zeros((len(options),len(ids),4,3),dtype=np.int64)
    for i,sid in enumerate(ids):
        base=np.load(a.champion/f'{sid}.npy',mmap_mode='r')
        label=np.asarray(Image.open(a.labels/f'{sid}.png'))
        if base.dtype!=np.float32 or base.shape!=(4,*label.shape):raise ValueError('Champion cache requires original-coordinate fp32 4HW')
        others={name:np.load(path/f'{sid}.npy',mmap_mode='r') for name,path in paths.items()}
        for name,p in others.items():
            if p.shape!=base.shape or p.dtype!=np.float32:raise ValueError(f'Probability cache mismatch: {name}/{sid}')
        for j,opt in enumerate(options):
            for c in range(4):
                p=base[c] if opt['family'] is None else ((1-opt['alpha'])*base[c]+opt['alpha']*others[opt['family']][c])
                if not np.isfinite(p).all() or p.min()<0 or p.max()>1:raise ValueError('Invalid probability')
                gt=label==c+1;valid=label!=255;pred=(p>.5)&valid
                counts[j,i,c]=[np.count_nonzero(pred&gt),np.count_nonzero(pred&~gt),np.count_nonzero(~pred&gt&valid)]
    result=crossfit(counts,options,np.array([fold_info['folds'][s] for s in ids]))
    result.update(n_images=len(ids),nominated_families=list(paths),channel_order=CLASSES,
                  cache_identities=identities,
                  checkpoint_selection_limitation=fold_info['limitation'])
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,indent=2),encoding='utf-8')
    np.savez(a.out.with_suffix('.counts.npz'),counts=counts,options=json.dumps(options),ids=ids)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
