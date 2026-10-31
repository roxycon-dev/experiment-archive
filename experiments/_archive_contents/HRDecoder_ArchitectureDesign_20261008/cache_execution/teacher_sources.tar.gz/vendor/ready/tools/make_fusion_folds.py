"""Lock three balanced val folds using GT prevalence; no model outputs."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from PIL import Image

def create_folds(root, seed=42, patients=None):
    labels=sorted((Path(root)/'labels/val').glob('*.png'))
    if len(labels)!=76:raise ValueError('DDR val must contain 76 fused labels')
    features={}
    for p in labels:
        y=np.asarray(Image.open(p))
        area=np.array([(y==c).sum() for c in range(1,5)],dtype=np.float64)
        features[p.stem]=np.r_[area>0,np.log1p(area)]
    xs=np.stack(list(features.values()))
    xs[:,4:]/=np.maximum(xs[:,4:].mean(0),1)
    features=dict(zip(features,xs))
    groups={}
    for sid in features:groups.setdefault((patients or {}).get(sid,sid),[]).append(sid)
    freq=xs[:,:4].sum(0)
    rng=np.random.RandomState(seed)
    ordered=list(groups.items());rng.shuffle(ordered)
    ordered.sort(key=lambda kv:sum((features[s][:4]/np.maximum(freq,1)).sum() for s in kv[1]),reverse=True)
    target=xs.sum(0)/3;running=np.zeros((3,8));size=np.zeros(3)
    folds={}
    for patient,sids in ordered:
        f=sum(features[s] for s in sids)
        costs=[]
        for k in range(3):
            r=running.copy();r[k]+=f
            n=size.copy();n[k]+=len(sids)
            cost=np.square((r-target)/np.maximum(target,1)).sum()+2*np.square((n-len(labels)/3)/(len(labels)/3)).sum()
            costs.append(cost)
        k=int(np.argmin(costs));running[k]+=f;size[k]+=len(sids)
        for sid in sids:folds[sid]=k
    return dict(seed=seed,split='val',folds=folds,sizes=size.astype(int).tolist(),
        positives_per_fold=running[:,:4].astype(int).tolist(),
        patient_grouping=patients is not None,selection='GT stratification only; no model predictions',
        limitation='checkpoint already selected on val; fusion cross-fit is not fully independent outer validation')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--ddr-root',required=True);ap.add_argument('--out',required=True)
    ap.add_argument('--patient-csv',type=Path);a=ap.parse_args()
    patients=None
    if a.patient_csv:
        with a.patient_csv.open(encoding='utf-8',newline='') as f:
            patients={r['sample_id']:r['patient_id'] for r in csv.DictReader(f)}
    result=create_folds(a.ddr_root,patients=patients)
    Path(a.out).write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='folds'},indent=2))

if __name__=='__main__':main()
