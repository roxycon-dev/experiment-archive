"""Frozen native-coordinate manifests and cache I/O for the three paired seeds."""
from pathlib import Path
import hashlib
import json
import os
import time
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1];EXP=ROOT.parent
DATA=EXP/'data/DDR_split_613'
READY=EXP/'HRDecoder_Ready_20261002'

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()

def write(p,value):
    p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_suffix(p.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8');os.replace(str(temp),str(p))

def save_probability(path,q):
    assert q.dtype==np.float32 and q.ndim==3 and q.shape[0]==4 and np.isfinite(q).all()
    assert q.min()>=0 and q.max()<=1
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.npy.tmp')
    with temp.open('wb') as f:np.save(f,q,allow_pickle=False)
    os.replace(str(temp),str(path))
    return sha(path)

def samples(cohort):
    if cohort in ('ddr','canary'):
        source=json.loads((ROOT/'evidence/local_asset_audit.json').read_text(encoding='utf-8'))
        split='test' if cohort=='ddr' else 'val'
        rows=source['data'][split]['samples']
        if cohort=='canary':rows=[r for r in rows if r['sid']=='007-1774-100']
        result=[]
        for r in rows:
            result.append(dict(r,cohort=cohort,image_path=str(DATA/'images'/split/(r['sid']+'.jpg')),
                               label_path=str(DATA/'labels'/split/(r['sid']+'.png'))))
        assert len(result)==(227 if cohort=='ddr' else 1)
        return result
    assert cohort=='idrid'
    source=json.loads((ROOT/'evidence/external_holdout_manifest.json').read_text(encoding='utf-8'))
    for r in source['samples']:
        r['cohort']=cohort
        r['image_path']=str(EXP/'data/idrid/images/testing'/(r['sid']+'.jpg'))
        for m in r['masks'].values():
            normalized=m['path'].replace('\\','/')
            m['path']=str(EXP/normalized[normalized.index('HRDecoder_Q_20261003/'):])
    assert len(source['samples'])==27
    return source['samples']

def image_rgb(row):
    assert sha(row['image_path'])==row['image_sha256']
    rgb=np.asarray(Image.open(row['image_path']).convert('RGB'))
    if 'hw' in row:assert list(rgb.shape[:2])==row['hw']
    return rgb

def truth(row):
    if row['cohort']!='idrid':
        assert sha(row['label_path'])==row['label_sha256']
        label=np.asarray(Image.open(row['label_path']))
        return np.stack([label==c for c in range(1,5)]),label!=255
    result=[]
    for name in ['EX','HE','SE','MA']:
        m=row['masks'][name];assert sha(m['path'])==m['sha256']
        result.append(np.asarray(Image.open(m['path']))>0)
    return np.stack(result),np.ones(result[0].shape,dtype=bool)

def counts(mask,row):
    gt,valid=truth(row)
    assert mask.dtype==bool and mask.shape==gt.shape
    return np.stack([(mask&gt&valid).sum((1,2)),(mask&~gt&valid).sum((1,2)),(~mask&gt&valid).sum((1,2))],axis=1).astype(np.int64).tolist()

def cache_dir(cohort,branch):
    return ROOT/'results/common_cache59'/cohort/branch

def load_cached(cohort,branch,sid,wait_seconds=0):
    p=cache_dir(cohort,branch)/(sid+'.npy');record=p.with_suffix('.json')
    started=time.monotonic()
    while not record.exists():
        if time.monotonic()-started>=wait_seconds:raise FileNotFoundError(record)
        time.sleep(2)
    r=json.loads(record.read_text(encoding='utf-8'));assert sha(p)==r['probability_sha256']
    q=np.load(str(p),allow_pickle=False)
    assert q.dtype==np.float32 and np.isfinite(q).all() and q.shape[0]==4
    return q,r
