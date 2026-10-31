"""Read-only val diagnostics: 25% oracle, fold-out calibration, paired image uncertainty."""
import argparse
import hashlib
from pathlib import Path
import numpy as np

from tools.probability_cache import load_probability
from nextlib.common import ROOT, read_json, write_json, CLASSES, digest
from nextlib.data import DDR
from nextlib.evaluate import counts, metrics


def image_bootstrap(candidate, baseline, repetitions=2000, seed=7303):
    candidate,baseline=np.asarray(candidate,np.int64),np.asarray(baseline,np.int64)
    if candidate.shape != baseline.shape or candidate.ndim != 3:
        raise ValueError('Matched per-image 4x3 counts required')
    rng=np.random.default_rng(seed)
    diff=[]
    n=len(candidate)
    for _ in range(repetitions):
        ix=rng.integers(n,size=n)
        diff.append(metrics(candidate[ix].sum(0))['mIoU_lesion4']-
                    metrics(baseline[ix].sum(0))['mIoU_lesion4'])
    return dict(delta_mean=float(np.mean(diff)), percentile95=np.percentile(diff,[2.5,97.5]).tolist(),
                repetitions=repetitions,seed=seed,
                note='Resamples paired images; this does not estimate training-seed variability')


def calibration_thresholds(images):
    pos=np.asarray([r['probability_histogram']['pos'] for r in images],np.int64).sum(0)
    neg=np.asarray([r['probability_histogram']['neg'] for r in images],np.int64).sum(0)
    tp=np.cumsum(pos[:,::-1],axis=1)[:,::-1]
    fp=np.cumsum(neg[:,::-1],axis=1)[:,::-1]
    score=tp/np.maximum(pos.sum(1)[:,None]+fp,1)
    # Restrict to a simple constant per-class probability offset; avoid tails.
    bins=score.shape[1]
    allowed=np.arange(round(.1*bins),round(.9*bins)+1)
    selected=[]
    for c in range(4):
        best=allowed[score[c,allowed]==score[c,allowed].max()]
        ix=int(best[np.argmin(abs(best/bins-.5))])
        selected.append(ix/bins)
    return selected


def analyze(dataset,cache,baseline,out,champion_cache=None):
    report=read_json(Path(cache)/'metrics.json')
    images=report['per_image']
    if set(r['sid'] for r in images)!=set(dataset.stems): raise ValueError('Incomplete val cache')
    folds={sid:int(hashlib.sha256(('strict-calibration-v1:'+sid).encode()).hexdigest(),16)%3
           for sid in dataset.stems}
    thresholds={}
    for fold in range(3):
        thresholds[fold]=calibration_thresholds([r for r in images if folds[r['sid']]!=fold])
    fitted=calibration_thresholds(images)
    calibrated,original,bc=[],[],[]
    oracle=np.zeros((4,3),np.int64)
    for sid in dataset.stems:
        _,label=dataset.load(sid)
        p=load_probability(cache,sid)
        champion=load_probability(champion_cache,sid) if champion_cache is not None else p
        q=load_probability(baseline,sid)
        original.append(counts(p,label));bc.append(counts(q,label))
        row=[]
        for c,t in enumerate(thresholds[folds[sid]]):
            gt=label==c+1
            pred=p[c]>t
            row.append([int((gt&pred).sum()),int((~gt&pred).sum()),int((gt&~pred).sum())])
            # Perfect hypothetical 25% expert, as a diagnostic capacity upper bound.
            # This bound always refers to the supplied champion probabilities.
            tp=int((gt&(champion[c]>1/3)).sum())
            fp=int((~gt&(champion[c]>2/3)).sum())
            oracle[c]+=[tp,fp,int(gt.sum())-tp]
        calibrated.append(row)
    result=dict(cache=str(Path(cache).resolve()),baseline=str(Path(baseline).resolve()),
        fold_rule='sha256(strict-calibration-v1:sid) mod 3',folds=folds,
        fold_fitted_thresholds=thresholds,full_val_fitted_thresholds=fitted,
        threshold_std=np.std(list(thresholds.values()),axis=0).tolist(),
        fixed_05=metrics(np.asarray(original).sum(0)),
        calibration_OOF=metrics(np.asarray(calibrated).sum(0)),
        baseline_fixed_05=metrics(np.asarray(bc).sum(0)),
        image_uncertainty=image_bootstrap(calibrated,bc),
        fixed25_GT_oracle=metrics(oracle),
        oracle_input_cache=str(champion_cache or cache),
        oracle_note='GT-assisted capability diagnostic, never a deployable prediction',
        fit_note='OOF thresholds fitted on other val folds; full-val fitted score is not reported as evidence')
    write_json(out,result)
    return result


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--cache',required=True);ap.add_argument('--baseline',required=True)
    ap.add_argument('--out',required=True)
    ap.add_argument('--champion-cache',default=str(ROOT/'work/G/champion'))
    a=ap.parse_args()
    s=read_json(ROOT/'configs/settings.json')
    d=DDR(s['ddr_root'],'val',read_json(s['val_contract'])['stems'])
    print(analyze(d,a.cache,a.baseline,a.out,a.champion_cache)['calibration_OOF'])
