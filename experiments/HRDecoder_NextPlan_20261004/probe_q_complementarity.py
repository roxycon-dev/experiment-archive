"""Retrospective validation diagnostics only; no model inference/test/model selection."""
from pathlib import Path
import json, time, hashlib
import numpy as np
from PIL import Image

OUT=Path(__file__).resolve().parent
QROOT=Path(r'D:\dataset\experiment\HRDecoder_Q_20261003')
OLD=Path(r'D:\dataset\experiment\HRDecoder_Next_20261003')
LABEL=Path(r'D:\HRDecoder\data\DDR_split_613\labels\val')
CLASSES=['EX','HE','SE','MA']

def read(p):
    with np.load(p,allow_pickle=False) as a: x=a['prob']
    assert x.dtype==np.float32 and np.isfinite(x).all() and x.min()>=0 and x.max()<=1
    return x

def score(x):
    a=np.asarray(x,dtype=np.int64)
    i=100*a[:,0]/a.sum(1)
    return {'counts':a.tolist(),'iou_pct':dict(zip(CLASSES,i.tolist())),'miou_pct':float(i.mean())}

def main():
    ids=json.loads((OLD/'configs/val_contract.json').read_text(encoding='utf-8-sig'))['stems']
    assert len(ids)==len(set(ids))==76
    protocol={'split':'val','n_images':76,'seeds':[42,2024],'checkpoint':16000,
              'current_formula':'.75*P0+.25*Q','P0_formula':'.75*C+.25*U0',
              'diagnostic_only_q_thresholds':[.25,.5,.75],
              'oracle':'GT chooses P0 or Q binary expert, not deployable',
              'purpose':'error partition and fixed sensitivity probes, no winner selection',
              'test_read':False,'gpu':False,'server':False}
    (OUT/'probe_q_protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    source=json.loads((QROOT/'reports/paired_validation_all_nodes_24k.json').read_text(encoding='utf-8-sig'))['nodes']
    totals={s:{k:np.zeros((4,3),dtype=np.int64) for k in ['p0','single','fixed_fusion','oracle','q_t025','q_t075']} for s in [42,2024]}
    error_names=['p0_fn','fusion_fn','shared_fn','q_correct_but_fusion_misses','p0_correct_but_fusion_misses',
                 'p0_fn_q_recognizes','p0_fn_rescued_by_fusion','p0_low_confident_fn','p0_low_fn_q_recognizes',
                 'fusion_fp','shared_fp','p0_fp_removed_by_fusion','p0_tn_harmed_by_fusion']
    errors={s:{k:np.zeros(4,dtype=np.int64) for k in error_names} for s in [42,2024]}
    rows=[]; hashes={}; started=time.time()
    for ix,sid in enumerate(ids):
        lp=LABEL/(sid+'.png'); lab=np.asarray(Image.open(lp)); hashes[sid]=hashlib.sha256(lp.read_bytes()).hexdigest()
        assert np.isin(lab,[0,1,2,3,4]).all()
        pc=read(OLD/'server_results/work/G/champion'/(sid+'.npz'))
        pu=read(OLD/'server_results/work/G/U0_native_whole'/(sid+'.npz'))
        p0=.75*pc+.25*pu; del pc,pu
        assert p0.shape==(4,*lab.shape)
        for seed in [42,2024]:
            q=read(QROOT/f'server_results/runs/Q_DDR_seed{seed}/val_u16000'/(sid+'.npz'))
            assert q.shape==p0.shape
            row={'sid':sid,'seed':seed,'counts':{},'errors':{k:[] for k in error_names}}
            for c in range(4):
                g=lab==(c+1); neg=~g; p=p0[c]>.5; a=q[c]>.5; f=(.75*p0[c]+.25*q[c])>.5
                predictions={'p0':p,'single':a,'fixed_fusion':f,'oracle':np.where(g,p|a,p&a),
                             'q_t025':q[c]>.25,'q_t075':q[c]>.75}
                for name,b in predictions.items():
                    ct=np.array([np.count_nonzero(g&b),np.count_nonzero(neg&b),np.count_nonzero(g&~b)],dtype=np.int64)
                    totals[seed][name][c]+=ct; row['counts'].setdefault(name,[]).append(ct.tolist())
                predicates=[g&~p,g&~f,g&~p&~a,g&a&~f,g&p&~f,
                            g&~p&a,g&~p&f,g&(p0[c]<=1/3),g&(p0[c]<=1/3)&a,
                            neg&f,neg&p&a,neg&p&~f,neg&~p&f]
                for name,b in zip(error_names,predicates):
                    n=int(np.count_nonzero(b)); errors[seed][name][c]+=n; row['errors'][name].append(n)
            rows.append(row); del q
        del p0
        if (ix+1)%10==0: print(f'val {ix+1}/76',flush=True)
    checks={}
    for seed in [42,2024]:
        checks[seed]={k:bool(np.array_equal(totals[seed][k],source[f'Q_DDR_seed{seed}']['16000'][k]['counts'])) for k in ['p0','single','fixed_fusion']}
        assert all(checks[seed].values())
    result={'protocol':protocol,'source_count_checks':checks,'seconds':time.time()-started,
            'metrics':{s:{k:score(v) for k,v in a.items()} for s,a in totals.items()},
            'errors':{s:{k:v.tolist() for k,v in a.items()} for s,a in errors.items()},
            'per_image':rows,'label_sha256':hashes,
            'limits':['Post-hoc val diagnostic, no new frozen winner.','Oracle uses labels and is not achievable evidence.',
                      'Three Q thresholds are diagnostic only, not a new deployment recommendation.',
                      'Archive identity receipts reused; no independent rehash of all probability files.']}
    (OUT/'probe_q_result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:result[k] for k in ['source_count_checks','seconds','metrics','errors']},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
