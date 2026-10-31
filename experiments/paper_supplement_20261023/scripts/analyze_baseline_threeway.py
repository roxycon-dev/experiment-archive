"""Join the new baseline with the frozen phase1 full/no-H4 predictions; no new inference or selection."""
import csv
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from analyze_recovered_training_results import ROOT,DATA,sha,score,write,frozen_rows
from lesion_metrics import grouped_counts,components

def ci(x):return np.quantile(x,[.025,.975]).tolist()

def image_bootstrap(a,index):
    sums=a[index].sum(1);return (sums[:,:,0]/np.maximum(sums.sum(2),1)).mean(1)*100

def draw_examples(base,out):
    phase=ROOT/'results/ddr_test_local_v1'
    selected=json.loads((phase/'analysis/example_selection.json').read_text(encoding='utf-8'))
    figures=out/'figures';figures.mkdir(exist_ok=True)
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',16)
    receipts=[]
    for example in selected['examples']:
        sid=example['sid'];rgb=np.asarray(Image.open(DATA/'images/test'/(sid+'.jpg')).convert('RGB'))
        assert sha(DATA/'images/test'/(sid+'.jpg'))==example['image_sha256']
        label=np.asarray(Image.open(DATA/'labels/test'/(sid+'.png')));valid=label!=255
        full_path=phase/'predictions'/(sid+'.npz');baseline_path=base/'ddr/predictions'/(sid+'.npz')
        assert sha(full_path)==example['mask_archive_sha256']
        record=json.loads((base/'ddr/per_image'/(sid+'.json')).read_text());assert sha(baseline_path)==record['mask_sha256']
        with np.load(full_path) as z:full=z['full'][3]&valid;no=z['no_h4'][3]&valid
        with np.load(baseline_path) as z:original=z['baseline'][3]&valid
        gt=(label==4)&valid;source=full if example['category']=='new_fp' else gt
        _,_,stats,_=components(source);x,y,w,h,area=map(int,stats[example['component_id']]);assert area==example['area_pixels']
        x0,y0,x1,y1=example['crop_xyxy'];raw=rgb[y0:y1,x0:x1]
        canvas=Image.new('RGB',(1250,330),'white');draw=ImageDraw.Draw(canvas)
        draw.text((8,6),example['category']+' | '+sid+' | fixed phase1 selection',font=font,fill='black')
        for j,(mask,title) in enumerate(zip([None,gt,original,no,full],['Image','MA ground truth','HRDecoder rerun','Without H4','Full system'])):
            a=raw.copy()
            if mask is not None:
                crop=mask[y0:y1,x0:x1];color=np.array([0,158,115]) if j==1 else np.array([213,94,0])
                a[crop]=(.45*a[crop]+.55*color).astype(np.uint8)
                edge=cv2.morphologyEx(crop.astype(np.uint8),cv2.MORPH_GRADIENT,np.ones((3,3),np.uint8)).astype(bool);a[edge]=color
            panel=Image.fromarray(a);scale=min(242/panel.width,242/panel.height)
            panel=panel.resize((round(panel.width*scale),round(panel.height*scale)),Image.Resampling.NEAREST)
            box=ImageDraw.Draw(panel);box.rectangle((max(0,(x-x0-3)*scale),max(0,(y-y0-3)*scale),min(panel.width-1,(x+w-x0+3)*scale),min(panel.height-1,(y+h-y0+3)*scale)),outline='white',width=1)
            canvas.paste(panel,(j*250+(250-panel.width)//2,62+(242-panel.height)//2));draw.text((j*250+8,34),title,font=font,fill='black')
        target=figures/(example['category']+'_'+sid+'_'+str(example['component_id'])+'.png');canvas.save(target,dpi=(300,300))
        receipts.append(dict(example,baseline_mask_sha256=sha(baseline_path),figure=str(target),selection_repeated=False))
    write(out/'example_provenance.json',{'rule':selected['rule'],'baseline_not_used_for_example_selection':True,'examples':receipts})

def main():
    base=ROOT/'results/baseline_retrained59';verified=json.loads((base/'analysis/verified_summary.json').read_text(encoding='utf-8'))
    assert verified['status']=='PASS_LOCAL_MASK_AND_GT_SHA_RECOUNT'
    arrays=np.load(base/'analysis/pixel_counts.npz');bins=json.loads((ROOT/'evidence/ma_size_bins.json').read_text())['boundaries_pixels']
    out=ROOT/'results/baseline_threeway';out.mkdir(exist_ok=True);pixels=[];lesions=[];comparisons={}
    for cohort,oldname in [('ddr','ddr_test_local_v1'),('idrid','idrid_holdout_v1')]:
        old=ROOT/'results'/oldname;assert json.loads((old/'completion.json').read_text())['status']=='COMPLETE'
        rows=frozen_rows(cohort);n=len(rows);indices=np.random.default_rng(20261023).integers(0,n,(2000,n))
        count={'baseline':arrays[cohort+'_baseline']}
        for name in ('no_h4','full'):
            count[name]=np.asarray([json.loads((old/'per_image'/(r['sid']+'.json')).read_text())['counts'][name] for r in rows],np.int64)
        for name,integer in count.items():pixels.append({'cohort':cohort,'method':name,**score(integer)})
        scores={name:image_bootstrap(integer,indices) for name,integer in count.items()}
        comparisons[cohort]={name:{'full_minus_row_mIoU_pp':score(count['full'])['mIoU_pct']-score(count[name])['mIoU_pct'],
                                  'paired_image_CI95_pp':ci(scores['full']-scores[name])} for name in ('baseline','no_h4')}
        old_lesions=json.loads((old/'analysis/lesion_metrics.json').read_text(encoding='utf-8'))
        records={'baseline':json.loads((base/'analysis'/('MA_lesions_'+cohort+'.json')).read_text(encoding='utf-8')),
                 'no_h4':old_lesions['no_h4'],'full':old_lesions['full']}
        boot={}
        for name,lr in records.items():
            lookup={r['sid']:r for r in lr};lr=[lookup[r['sid']] for r in rows]
            matrix=np.asarray([[r['tp'],r['fp'],r['fn']] for r in lr],np.int64);sums=matrix[indices].sum(1);total=matrix.sum(0);tp,fp,fn=map(int,total)
            groups=[grouped_counts(r,bins) for r in lr]
            for size in ('all','small','medium','large'):
                if size=='all':denom=matrix[:,0]+matrix[:,2];det=matrix[:,0]
                else:denom=np.array([r[size]['gt'] for r in groups]);det=np.array([r[size]['detected'] for r in groups])
                sampling=det[indices].sum(1)/np.maximum(denom[indices].sum(1),1)*100
                lesions.append({'cohort':cohort,'method':name,'size':size,'GT':int(denom.sum()),'detected':int(det.sum()),
                    'detection_pct':float(det.sum()/max(denom.sum(),1)*100),'detection_image_CI95_pct':ci(sampling),
                    'false_positives_per_image':fp/n if size=='all' else None,'lesion_precision_pct':tp/max(tp+fp,1)*100 if size=='all' else None})
            boot[name]={'detection':sums[:,0]/np.maximum(sums[:,0]+sums[:,2],1)*100,'FP_per_image':sums[:,1]/n}
        comparisons[cohort]['MA_full_minus_no_h4']={metric:ci(boot['full'][metric]-boot['no_h4'][metric]) for metric in ('detection','FP_per_image')}
    draw_examples(base,out)
    result={'status':'COMPLETE_THREEWAY_BASELINE_MA_COMPARISON','pixel_rows':pixels,'MA_lesion_rows':lesions,'comparisons':comparisons,
            'pixel_size_boundaries':bins,'runtime_note':'Baseline newly trained legacyFP32; full/noH4 are the original frozen phase1 BF16 system outputs. Same images,GT,coordinates,threshold and metric; no claim of bitwise cross-runtime replay.',
            'scope':'Frozen-system replacement/dependency and observed MA recovery/false-positive tradeoff; not a proof of independent training causality',
            'patient_bootstrap':False,'new_training_or_tuning':False}
    write(out/'summary.json',result)
    for name,rows in [('pixel_table',pixels),('MA_lesion_table',lesions)]:
        with (out/(name+'.csv')).open('w',newline='',encoding='utf-8-sig') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

if __name__=='__main__':main()
