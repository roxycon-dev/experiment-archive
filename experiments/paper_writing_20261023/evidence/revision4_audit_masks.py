"""Read-only independent saved binary-mask/GT recount. Never loads a model."""
from pathlib import Path
import json,csv,hashlib,os,zipfile,time
from collections import Counter
import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment
import cv2
cv2.setNumThreads(2)
OUT=Path(__file__).resolve().parents[1];ROOT=OUT.parent/'paper_supplement_20261023';E=OUT/'evidence'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.file_digest(open(p,'rb'),'sha256').hexdigest()
def write(p,a):p.write_text(json.dumps(a,ensure_ascii=False,indent=2),encoding='utf-8')
files=[];archives=[]
for folder,dirs,names in os.walk(ROOT):
    for name in names:
        p=Path(folder)/name;s=p.stat();r={'path':str(p.relative_to(ROOT)),'bytes':s.st_size,'suffix':p.suffix,'modified_unix':s.st_mtime};files.append(r)
        if p.suffix.lower()=='.zip':
            try:
                with zipfile.ZipFile(p) as z:archives.append({'path':r['path'],'bytes':s.st_size,'n_members':len(z.infolist()),'uncompressed_bytes':sum(i.file_size for i in z.infolist()),'listing':'central directory; not extracted'})
            except Exception as ex:archives.append({'path':r['path'],'bytes':s.st_size,'error':str(ex)})
with open(E/'revision4_file_inventory.csv','w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=list(files[0]));w.writeheader();w.writerows(files)
write(E/'revision4_inventory_summary.json',{'root':str(ROOT),'files':len(files),'bytes':sum(r['bytes'] for r in files),'extensions':dict(Counter(r['suffix'] for r in files)),'archives':archives,'read_boundary':'All file metadata; archive directory metadata; selected complete reports/protocols/saved per-image records; no binary weight deserialization or training/inference.'})
external=read(ROOT/'evidence/external_holdout_manifest.json');ext={r['sid']:r for r in external['samples']}
targets=[('stage_ddr',ROOT/'results/ddr_test_local_v1','ddr'),('stage_idrid',ROOT/'results/idrid_holdout_v1','idrid')]
for cohort in ['val','ddr','idrid']:targets.append(('baseline_'+cohort,ROOT/'results/baseline_original_local'/cohort,cohort))
for seed in [42,2024,3407]:
    for arm in ['C','S']:
        for cohort in ['ddr','idrid']:targets.append((f'{arm}_{seed}_{cohort}',ROOT/'results/paired_seed59'/f'{arm}_seed{seed}'/cohort,cohort))
checks=[];summary=[];start=time.time();lesion_checks=[]
def assert_check(name,actual,expected):
    ok=bool(np.array_equal(actual,expected));checks.append({'check':name,'pass':ok})
    if not ok:raise AssertionError(name)
def lesion(gt,pred):
    ng,gl,gs,_=cv2.connectedComponentsWithStats(gt.astype(np.uint8),8)
    np_,pl,ps,_=cv2.connectedComponentsWithStats(pred.astype(np.uint8),8)
    ga=gs[1:,4];pa=ps[1:,4];mask=(gl>0)&(pl>0);matches=[]
    if mask.any():
        pairs,inter=np.unique(gl[mask].astype(np.int64)*np_+pl[mask],return_counts=True);gi=pairs//np_-1;pi=pairs%np_-1
        iou=inter/(ga[gi]+pa[pi]-inter);keep=iou>=.1;gkeys=np.unique(gi[keep]);pkeys=np.unique(pi[keep])
        if len(gkeys) and len(pkeys):
            weights=np.zeros((len(gkeys),len(pkeys)));bonus=min(len(ga),len(pa))+1
            weights[np.searchsorted(gkeys,gi[keep]),np.searchsorted(pkeys,pi[keep])]=bonus+iou[keep]
            aa,bb=linear_sum_assignment(weights,maximize=True)
            matches=[int(gkeys[a]+1) for a,b in zip(aa,bb) if weights[a,b]>0]
    return [len(matches),len(pa)-len(matches),len(ga)-len(matches)],ga.tolist()
for label,folder,cohort in targets:
    records=[read(p) for p in sorted((folder/'per_image').glob('*.json'))];totals={};n=0
    lesion_records=None
    if label in ['stage_ddr','stage_idrid']:
        lesion_records=read(folder/'analysis/lesion_metrics.json')
        lesion_records={m:{r['sid']:r for r in rows} for m,rows in lesion_records.items()}
    elif label in ['baseline_ddr','baseline_idrid']:
        a=read(ROOT/'results/baseline_original_local/analysis'/f'MA_lesions_{cohort}.json');lesion_records={'baseline':{r['sid']:r for r in a}}
    for record in records:
        sid=record['sid'];maskpath=folder/'predictions'/(sid+'.npz')
        assert_check(label+sid+' binary mask SHA',sha(maskpath),record['mask_sha256'])
        if cohort in ['ddr','val']:
            split='val' if cohort=='val' else 'test';img=Path('D:/HRDecoder/data/DDR_split_613/images')/split/(sid+'.jpg');p=Path('D:/HRDecoder/data/DDR_split_613/labels')/split/(sid+'.png')
            assert_check(label+sid+' GT SHA',sha(p),record['label_sha256']);gtindex=np.asarray(Image.open(p));valid=gtindex!=255;gt=np.stack([gtindex==i for i in [1,2,3,4]])
        else:
            er=ext[sid];img=Path(er['image_path']);gg=[]
            for c in ['EX','HE','SE','MA']:
                p=Path(er['masks'][c]['path']);assert_check(label+sid+c+' GT SHA',sha(p),er['masks'][c]['sha256']);gg.append(np.asarray(Image.open(p))>0)
            gt=np.stack(gg);valid=np.ones(gt.shape[1:],bool)
        assert_check(label+sid+' image SHA',sha(img),record['image_sha256'])
        with np.load(maskpath,allow_pickle=False) as z:
            assert set(z.files)==set(record['counts'])
            for mode in z.files:
                pred=z[mode];assert pred.dtype==bool and pred.shape==gt.shape
                tp=np.count_nonzero(pred&gt&valid,axis=(1,2));fp=np.count_nonzero(pred&~gt&valid,axis=(1,2));fn=np.count_nonzero(~pred&gt&valid,axis=(1,2));ct=np.stack([tp,fp,fn],axis=1)
                assert_check(label+sid+mode+' mask GT counts',ct,record['counts'][mode]);totals[mode]=totals.get(mode,np.zeros((4,3),np.int64))+ct
                if lesion_records is not None and mode in ['baseline','full','no_h4']:
                    measures,areas=lesion(gt[3]&valid,pred[3]&valid);rr=lesion_records[mode][sid]
                    assert_check(label+sid+mode+' lesion TPFPFN',measures,[rr[k] for k in ['tp','fp','fn']]);assert_check(label+sid+mode+' GT areas',areas,rr['gt_areas'])
        n+=1
        if n%50==0:print(json.dumps({'run':label,'images':n,'seconds':round(time.time()-start)}),flush=True)
    summary.append({'run':label,'n_images':n,'totals':{k:v.tolist() for k,v in totals.items()}})
    write(E/'revision4_mask_GT_audit_progress.json',{'completed':summary,'all_completed_checks_pass':True,'checks_completed':len(checks),'elapsed_seconds':time.time()-start})
write(E/'revision4_mask_GT_audit.json',{'status':'PASS_ALL_SAVED_MASK_GT_RECOUNTS','runs':summary,'checks':checks,'all_pass':True,'elapsed_seconds':time.time()-start,'scope':'Every selected stage1, recovered original baseline val/test/external, and six FP32 paired runs binary mask/hash/GT/image; independent native pixel count; MA component TPFPFN and GT areas for baseline/full/no correction; no model execution'})
print(json.dumps({'completed_runs':len(summary),'checks':len(checks),'all_pass':True,'seconds':time.time()-start}))
