"""Independent local native-mask recount and conditional three-seed paired statistics."""
import csv
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from lesion_metrics import match_lesions,grouped_counts

ROOT=Path(__file__).resolve().parents[1];DATA=Path('D:/HRDecoder/data/DDR_split_613')
cv2.setNumThreads(2)

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()

def write(p,value):
    p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.json.tmp')
    t.write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8');t.replace(p)

def frozen_rows(cohort):
    if cohort in ('ddr','val'):
        split='val' if cohort=='val' else 'test'
        source=json.loads((ROOT/'evidence/local_asset_audit.json').read_text(encoding='utf-8'))['data'][split]['samples']
        return [dict(r,image_path=str(DATA/'images'/split/(r['sid']+'.jpg')),label_path=str(DATA/'labels'/split/(r['sid']+'.png'))) for r in source]
    return json.loads((ROOT/'evidence/external_holdout_manifest.json').read_text(encoding='utf-8'))['samples']

def truth(row,cohort):
    if cohort in ('ddr','val'):
        assert sha(row['label_path'])==row['label_sha256'];label=np.asarray(Image.open(row['label_path']))
        return np.stack([label==c for c in range(1,5)]),label!=255
    masks=[]
    for c in ('EX','HE','SE','MA'):
        p=row['masks'][c];assert sha(p['path'])==p['sha256'];masks.append(np.asarray(Image.open(p['path']))>0)
    return np.stack(masks),np.ones(masks[0].shape,dtype=bool)

def recount(b,g,v):
    return np.stack((np.count_nonzero(b&g&v,axis=(1,2)),np.count_nonzero(b&~g&v,axis=(1,2)),np.count_nonzero(~b&g&v,axis=(1,2))),axis=1).astype(np.int64)

def score(a):
    total=np.asarray(a,np.int64).sum(0);tp,fp,fn=total.T
    values=tp/np.maximum(tp+fp+fn,1)*100
    return {'mIoU_pct':float(values.mean()),**dict(zip(('EX_IoU_pct','HE_IoU_pct','SE_IoU_pct','MA_IoU_pct'),values.tolist())),
            'MA_pixel_precision_pct':float(tp[3]/max(tp[3]+fp[3],1)*100),'MA_pixel_recall_pct':float(tp[3]/max(tp[3]+fn[3],1)*100),'global_counts':total.tolist()}

def analyze(name):
    original=name=='baseline_original_seed42'
    baseline=name=='baseline_seed42' or original
    run=ROOT/('results/baseline_original_local' if original else 'results/baseline_retrained59') if baseline else ROOT/'results/paired_seed59'/name
    receipt=json.loads((ROOT/'evidence/recovered_evaluations'/(name+'.json')).read_text())
    out=run/'analysis';summary_file=out/'verified_summary.json'
    if summary_file.exists():
        saved=json.loads(summary_file.read_text(encoding='utf-8'));assert saved['source_archive_sha256']==receipt['archive_sha256'];return saved
    assert json.loads((run/'completion.json').read_text())['status']=='COMPLETE' and not (run/'failure.json').exists()
    manifest=json.loads((run/'manifest.json').read_text());methods=('baseline',) if baseline else ('single','fixed_fusion')
    if original:
        identity=json.loads((ROOT/'evidence/recovered_original_baseline_identity.json').read_text(encoding='utf-8'))
        assert manifest['job']['config_sha256']==identity['inference_config_sha256']
        assert manifest['job']['checkpoint_sha256']==identity['source_checkpoint_sha256']
        assert manifest['job']['weight_sha256']==identity['compact_sha256']
    elif baseline:
        prepared=json.loads((ROOT/'evidence/baseline_reproduction_preparation.json').read_text(encoding='utf-8'))
        assert manifest['job']['config_sha256']==prepared['reproduction_config_sha256']
    bins=json.loads((ROOT/'evidence/ma_size_bins.json').read_text())['boundaries_pixels']
    summaries={};arrays={};lesion_rows={};p0_hashes={}
    for cohort in (('val','ddr','idrid') if original else ('ddr','idrid')):
        rows=frozen_rows(cohort);n=76 if cohort=='val' else (227 if cohort=='ddr' else 27);assert len(rows)==n
        assert json.loads((run/cohort/'completion.json').read_text())['images']==n
        arrays[cohort]={k:[] for k in methods};p0_hashes[cohort]={};lesion_rows[cohort]=[]
        for i,row in enumerate(rows):
            sid=row['sid'];record=json.loads((run/cohort/'per_image'/(sid+'.json')).read_text())
            pred=run/cohort/'predictions'/(sid+'.npz')
            assert record['sid']==sid and record['cohort']==cohort
            assert sha(pred)==record['mask_sha256']
            assert sha(row['image_path'])==row['image_sha256']==record['image_sha256']
            if cohort in ('ddr','val'):assert record['label_sha256']==row['label_sha256']
            else:assert record['external_GT_sha256']=={c:m['sha256'] for c,m in row['masks'].items()}
            gt,valid=truth(row,cohort)
            with np.load(pred,allow_pickle=False) as z:
                assert set(z.files)==set(methods)
                for method in methods:
                    b=z[method];assert b.dtype==bool and b.shape==gt.shape
                    integer=recount(b,gt,valid);assert integer.tolist()==record['counts'][method],(sid,method)
                    arrays[cohort][method].append(integer)
                    if baseline:
                        lr=match_lesions(gt[3],b[3],valid);lr.update(sid=sid,size_groups=grouped_counts(lr,bins));lesion_rows[cohort].append(lr)
            if not baseline:p0_hashes[cohort][sid]=record['P0_sha256']
            if (i+1)%50==0:print(json.dumps({'verified_run':name,'cohort':cohort,'images':i+1}),flush=True)
        summaries[cohort]={k:score(v) for k,v in arrays[cohort].items()}
    out.mkdir(exist_ok=True)
    np.savez_compressed(out/'pixel_counts.npz',**{cohort+'_'+method:np.asarray(value,np.int64) for cohort,methods_ in arrays.items() for method,value in methods_.items()})
    if baseline:
        for cohort,rows in lesion_rows.items():
            totals=np.sum([[r['tp'],r['fp'],r['fn']] for r in rows],axis=0);tp,fp,fn=map(int,totals)
            groups={key:{field:sum(r['size_groups'][key][field] for r in rows) for field in ('gt','detected')} for key in ('small','medium','large')}
            for group in groups.values():group['detection_pct']=group['detected']/max(group['gt'],1)*100
            summaries[cohort]['MA_lesion']={'TP':tp,'FP':fp,'FN':fn,'detection_pct':tp/max(tp+fn,1)*100,
                'false_positives_per_image':fp/len(rows),'lesion_precision_pct':tp/max(tp+fp,1)*100,'size_groups':groups,
                'size_boundaries_pixels':bins,'matching':'8-connectivity;IoU>=.10;maximum cardinality then IoU;one-to-one'}
            write(out/('MA_lesions_'+cohort+'.json'),rows)
    result={'status':'PASS_LOCAL_MASK_AND_GT_SHA_RECOUNT','name':name,'source_archive_sha256':receipt['archive_sha256'],
            'model_checkpoint_SHA_from_run':manifest['job'].get('weight_sha256',manifest['job'].get('checkpoint_sha256')),
            'cohorts':summaries,'P0_per_image_SHA':p0_hashes if not baseline else None,
            'model_bytes_local_backup':'SEPARATE_PENDING','raw_probabilities_local_backup':False}
    write(summary_file,result);return result

def aggregate(summaries):
    results={s['name']:s for s in summaries};pairs=[];group=[]
    available=[s for s in (42,2024,3407) if all(a+'_seed'+str(s) in results for a in ('C','S'))]
    runtime_reference=None;p0_reference=None
    for seed in available:
        c=results['C_seed'+str(seed)];s=results['S_seed'+str(seed)]
        cm=json.loads((ROOT/'results/paired_seed59'/c['name']/'manifest.json').read_text());sm=json.loads((ROOT/'results/paired_seed59'/s['name']/'manifest.json').read_text())
        for field in ('torch','CUDA','GPU','precision','helper_sha256'):assert cm[field]==sm[field]
        runtime_identity={field:cm[field] for field in ('torch','CUDA','GPU','precision','helper_sha256')}
        if runtime_reference is None:runtime_reference=runtime_identity
        else:assert runtime_identity==runtime_reference,'Do not mix evaluation runtimes across seeds'
        assert cm['job']['parent_full_sha256']==sm['job']['parent_full_sha256']=='68d7b5821fab1210b3336cf2a63b9de893f24fc32876f5255d7d662084b5e728'
        assert cm['job']['runtime_protocol_sha256']==sm['job']['runtime_protocol_sha256']==sha(ROOT/'plan/paired_evaluation_runtime.md')
        assert c['P0_per_image_SHA']==s['P0_per_image_SHA']
        if p0_reference is None:p0_reference=c['P0_per_image_SHA']
        else:assert c['P0_per_image_SHA']==p0_reference,'All seeds must share identical per-image P0'
        ca=np.load(ROOT/'results/paired_seed59'/c['name']/'analysis/pixel_counts.npz');sa=np.load(ROOT/'results/paired_seed59'/s['name']/'analysis/pixel_counts.npz')
        for cohort in ('ddr','idrid'):
            n=227 if cohort=='ddr' else 27;indices=np.random.default_rng(20261023).integers(0,n,(2000,n))
            for mode in ('single','fixed_fusion'):
                cc=ca[cohort+'_'+mode];sc=sa[cohort+'_'+mode]
                def bootstrap(x):
                    sums=x[indices].sum(axis=1);return (sums[:,:,0]/np.maximum(sums.sum(axis=2),1)).mean(axis=1)*100
                ci=np.quantile(bootstrap(sc)-bootstrap(cc),[.025,.975]).tolist()
                cr=c['cohorts'][cohort][mode];sr=s['cohorts'][cohort][mode]
                pairs.append({'seed':seed,'cohort':cohort,'mode':mode,'C_mIoU_pct':cr['mIoU_pct'],'S_mIoU_pct':sr['mIoU_pct'],
                              'S_minus_C_pp':sr['mIoU_pct']-cr['mIoU_pct'],'paired_image_bootstrap_CI95_pp':ci,
                              'C_MA_IoU_pct':cr['MA_IoU_pct'],'S_MA_IoU_pct':sr['MA_IoU_pct']})
        ca.close();sa.close()
    for cohort in ('ddr','idrid'):
        for mode in ('single','fixed_fusion'):
            rows=[r for r in pairs if r['cohort']==cohort and r['mode']==mode]
            if not rows:continue
            row={'cohort':cohort,'mode':mode,'paired_seeds':len(rows)}
            for key in ('C_mIoU_pct','S_mIoU_pct','S_minus_C_pp'):
                a=np.array([r[key] for r in rows]);row[key+'_mean']=float(a.mean());row[key+'_sample_SD']=float(a.std(ddof=1)) if len(a)>1 else None
            group.append(row)
    out=ROOT/'results/paired_seed_summary';out.mkdir(exist_ok=True)
    value={'status':'COMPLETE_THREE_PAIRED_SEEDS' if len(available)==3 else 'PARTIAL_DO_NOT_CLAIM_THREE_SEED_COMPLETION',
           'paired_seeds_available':available,'required':[42,2024,3407],'per_seed':pairs,'mean_and_sample_SD':group,
           'interpretation':'Adaptation variability conditional on one common F12_seed42 parent, not independent end-to-end training',
           'CI_note':'Image bootstrap intervals do not measure seed variation; no patient identities available; no tuning after evaluation'}
    write(out/'summary.json',value)
    if pairs:
        with (out/'paired_seed_results.csv').open('w',newline='',encoding='utf-8-sig') as f:
            w=csv.DictWriter(f,fieldnames=list(pairs[0]));w.writeheader();w.writerows(pairs)
    print(json.dumps({'paired_seed_status':value['status'],'seeds_available':available,'completed_local_runs':list(results)}),flush=True)

def main():
    summaries=[]
    for receipt in sorted((ROOT/'evidence/recovered_evaluations').glob('*.json')):
        summaries.append(analyze(receipt.stem))
    aggregate(summaries)

if __name__=='__main__':main()
