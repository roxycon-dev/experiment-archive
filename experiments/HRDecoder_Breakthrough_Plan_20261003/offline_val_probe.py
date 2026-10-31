"""Planning evidence only: saved validation probabilities; no model inference or test."""
from pathlib import Path
import json, time, sys, hashlib
import numpy as np
from PIL import Image

ROOT = Path(r'D:\dataset\experiment\HRDecoder_Next_20261003')
OUT = Path(__file__).resolve().parent
WORK = ROOT / 'server_results/work'
LABELS = Path(r'D:\HRDecoder\data\DDR_split_613\labels\val')
CLASSES = ('EX','HE','SE','MA')
CACHES = {'C':WORK/'G/champion', 'U':WORK/'G/U0_native_whole',
          'M0':WORK/'M_seed42/M0/val_u1000', 'M1':WORK/'M_seed42/M1/val_u1000',
          'H1':WORK/'G/H1_square2048_whole_ordinary'}

def counts(pred, label):
    result=[]
    for c in range(4):
        gt=label==c+1; p=pred[c]; tp=int(np.count_nonzero(gt&p))
        result.append([tp,int(np.count_nonzero(p&~gt)),int(np.count_nonzero(gt&~p))])
    return np.asarray(result,dtype=np.int64)

def summary(ct):
    ct=np.asarray(ct,dtype=np.int64)
    iou=ct[:,0]/np.maximum(ct.sum(1),1)
    return dict(counts=ct.tolist(),miou_pct=float(iou.mean()*100),
                class_iou_pct=dict(zip(CLASSES,(iou*100).tolist())))

def load(cache,sid):
    with np.load(cache/(sid+'.npz'),allow_pickle=False) as z:p=z['prob']
    if p.dtype!=np.float32 or not np.isfinite(p).all():raise ValueError('Bad probability dtype/value')
    return p

def main():
    ids=json.loads((ROOT/'configs/val_contract.json').read_text(encoding='utf-8-sig'))['stems']
    assert len(ids)==76 and len(set(ids))==76
    # Fixed before reading any new candidate result: two common-best M models,
    # one deployment weight .25; H1 used for diagnostic agreement only.
    protocol=dict(split='val',n_images=76,threshold=.5,strict_greater=True,
        candidate_formulas={'M0_25':'0.75*C+0.25*M0@1000','M1_25':'0.75*C+0.25*M1@1000'},
        scope='Retrospective planning diagnostic, not a new frozen test winner; no alpha search',
        oracle='GT chooses the correct thresholded binary expert at each pixel if available; NOT deployable and NOT a cap on a new model',
        experts_for_oracle=['P0','C','M0','M1','H1'],cuda=False,test_read=False)
    (OUT/'offline_probe_protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding='utf-8')
    total={}; rows=[]; started=time.time(); sha_labels={}
    for ix,sid in enumerate(ids):
        lp=LABELS/(sid+'.png'); label=np.asarray(Image.open(lp))
        if not np.isin(label,[0,1,2,3,4]).all():raise ValueError('Uncontracted label')
        sha_labels[sid]=hashlib.sha256(lp.read_bytes()).hexdigest()
        c=load(CACHES['C'],sid); u=load(CACHES['U'],sid)
        assert c.shape==(4,*label.shape) and u.shape==c.shape
        p0=(.75*c+.25*u).astype(np.float32); del u
        pred={'C':c>.5,'P0':p0>.5}; del p0
        row=dict(sid=sid,counts={})
        for name in ['M0','M1','H1']:
            p=load(CACHES[name],sid); assert p.shape==c.shape
            pred[name]=p>.5
            if name.startswith('M'):
                mix=.75*c+.25*p
                ct=counts(mix>.5,label); row['counts'][name+'_25']=ct.tolist();del mix
            del p
        del c
        for name in ['P0','C','M0','M1','H1']:
            row['counts'][name]=counts(pred[name],label).tolist()
        any_pos=pred['P0'].copy();all_pos=pred['P0'].copy()
        for name in ['C','M0','M1','H1']:
            any_pos|=pred[name];all_pos&=pred[name]
        oracle=[];shared=[]
        for k in range(4):
            gt=label==k+1
            tp=int(np.count_nonzero(gt&any_pos[k]));fp=int(np.count_nonzero(~gt&all_pos[k]));fn=int(np.count_nonzero(gt&~any_pos[k]))
            oracle.append([tp,fp,fn])
            shared.append(dict(shared_fn=fn,shared_fp=fp,
                old_fn=int(np.count_nonzero(gt&~pred['P0'][k])),
                old_fp=int(np.count_nonzero(~gt&pred['P0'][k]))))
        row['counts']['pixel_oracle']=oracle;row['shared_errors']=shared
        for name,ct in row['counts'].items():total[name]=total.get(name,np.zeros((4,3),np.int64))+np.asarray(ct)
        rows.append(row)
        if (ix+1)%10==0:print(f'{ix+1}/76 validation images; {time.time()-started:.1f}s',flush=True)
    original={
        'P0':WORK/'G/current_deployment/metrics.json',
        'C':WORK/'G/champion_diagnostics/metrics.json',
        'M0':CACHES['M0']/'metrics.json','M1':CACHES['M1']/'metrics.json','H1':CACHES['H1']/'metrics.json'}
    checks={}
    for name,mp in original.items():
        d=json.loads(mp.read_text(encoding='utf-8-sig'))
        checks[name]=bool(np.array_equal(total[name],np.asarray(d['counts'])))
        if not checks[name]:raise ValueError('Archived prediction/label metric mismatch: '+name)
    stats={name:summary(ct) for name,ct in total.items()}
    old=np.asarray([r['counts']['P0'] for r in rows])
    rng=np.random.default_rng(100303);indices=rng.integers(0,76,(2000,76))
    old_sample=old[indices].sum(1);old_score=(old_sample[:,:,0]/np.maximum(old_sample.sum(2),1)).mean(1)
    for name in ['M0_25','M1_25']:
        new=np.asarray([r['counts'][name] for r in rows]);ct=new[indices].sum(1)
        ds=((ct[:,:,0]/np.maximum(ct.sum(2),1)).mean(1)-old_score)*100
        stats[name]['delta_vs_P0_pp']=stats[name]['miou_pct']-stats['P0']['miou_pct']
        stats[name]['paired_image_percentile95_pp']=np.quantile(ds,[.025,.975]).tolist()
    shared=[]
    for k,name in enumerate(CLASSES):
        sums={key:sum(r['shared_errors'][k][key] for r in rows) for key in ['shared_fn','shared_fp','old_fn','old_fp']}
        sums.update(lesion_class=name,shared_fraction_of_old_fn=sums['shared_fn']/max(sums['old_fn'],1),shared_fraction_of_old_fp=sums['shared_fp']/max(sums['old_fp'],1));shared.append(sums)
    result=dict(protocol=protocol,seconds=time.time()-started,source_count_checks=checks,
        summary=stats,shared_errors=shared,per_image=rows,label_sha256=sha_labels,
        limits=['post-hoc val only; no correction for historical model selection','image bootstrap is not seed variability','perfect pixel chooser uses GT and is not deployable','existing archive backups used; no fresh full-probability SHA pass'])
    (OUT/'offline_val_probe_result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ['per_image','label_sha256']},ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
