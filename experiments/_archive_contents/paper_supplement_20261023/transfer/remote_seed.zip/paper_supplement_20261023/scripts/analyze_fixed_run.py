"""Hash-verified pixel/lesion evaluation and deterministic real-image figures."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from lesion_metrics import match_lesions, grouped_counts, components

ROOT = Path(__file__).resolve().parents[1]
DATA = Path('D:/HRDecoder/data/DDR_split_613')
NAMES = {'full':'Full system', 'no_h4':'Without H4', 'no_convnext':'Without ConvNeXt', 'matched_c20':'Replace S20 with C20'}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write_json(p, value):
    p.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def summary(a):
    a = np.asarray(a,dtype=np.int64).sum(axis=0)
    tp,fp,fn = a.T
    iou = tp/np.maximum(tp+fp+fn,1)
    return {'miou_pct':float(iou.mean()*100), **dict(zip(['EX_IoU_pct','HE_IoU_pct','SE_IoU_pct','MA_IoU_pct'],(iou*100).tolist())),
            'MA_precision_pct':float(tp[3]/max(tp[3]+fp[3],1)*100),
            'MA_recall_pct':float(tp[3]/max(tp[3]+fn[3],1)*100),'counts':a.tolist()}


def recompute_binary(b,label):
    gt = np.stack([label==c for c in range(1,5)])
    v = label!=255
    return np.stack([(b&gt&v).sum((1,2)),(b&~gt&v).sum((1,2)),(~b&gt&v).sum((1,2))],axis=1).astype(np.int64)


def bootstrap_scores(a,indices):
    sums = a[indices].sum(axis=1)
    return (sums[:,:,0]/np.maximum(sums.sum(axis=2),1)).mean(axis=1)*100


def ci(values):
    return np.quantile(values,[.025,.975]).tolist()


def draw_example(example, run, split, output):
    sid = example['sid']
    rgb = np.asarray(Image.open(DATA/'images'/split/(sid+'.jpg')).convert('RGB'))
    label = np.asarray(Image.open(DATA/'labels'/split/(sid+'.png')))
    gt = label==4
    valid = label!=255
    with np.load(run/'predictions'/(sid+'.npz'),allow_pickle=False) as z:
        no,full = z['no_h4'][3]&valid,z['full'][3]&valid
    mask = full if example['category']=='new_fp' else gt
    _,labels,stats,_ = components(mask)
    k=example['component_id']
    x,y,w,h,area=map(int,stats[k])
    x0,y0=max(0,x-64),max(0,y-64)
    x1,y1=min(rgb.shape[1],x+w+64),min(rgb.shape[0],y+h+64)
    raw=rgb[y0:y1,x0:x1]
    masks=[None,gt[y0:y1,x0:x1],no[y0:y1,x0:x1],full[y0:y1,x0:x1]]
    font_path=Path('C:/Windows/Fonts/arial.ttf')
    font=ImageFont.truetype(str(font_path),15) if font_path.exists() else ImageFont.load_default()
    title_font=ImageFont.truetype(str(font_path),16) if font_path.exists() else font
    panel_size=250
    canvas=Image.new('RGB',(4*panel_size, panel_size+74),'white')
    draw=ImageDraw.Draw(canvas)
    title=f"{example['category']} | {sid} | component {k}, area {area} px"
    draw.text((8,6),title,fill='black',font=title_font)
    for j,(mask,title) in enumerate(zip(masks,['Image','MA ground truth','Without H4','Full system'])):
        a=raw.copy()
        if mask is not None:
            color=np.array([0,158,115]) if j==1 else np.array([213,94,0])
            a[mask]=(.45*a[mask]+.55*color).astype(np.uint8)
            contour=cv2.morphologyEx(mask.astype(np.uint8),cv2.MORPH_GRADIENT,np.ones((3,3),np.uint8)).astype(bool)
            a[contour]=color
        image=Image.fromarray(a)
        image.thumbnail((panel_size-8,panel_size-8),Image.Resampling.NEAREST)
        canvas.paste(image,(j*panel_size+(panel_size-image.width)//2,54+(panel_size-image.height)//2))
        draw.text((j*panel_size+8,32),title,fill='black',font=font)
    path=output/(example['category']+'_'+sid+'_'+str(k)+'.png')
    canvas.save(path,dpi=(300,300))
    example.update(crop_xyxy=[x0,y0,x1,y1],area_pixels=area,figure=str(path),image_sha256=sha(DATA/'images'/split/(sid+'.jpg')),
                   mask_archive_sha256=sha(run/'predictions'/(sid+'.npz')))
    return canvas


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('run',type=Path)
    parser.add_argument('--allow-partial',action='store_true')
    args=parser.parse_args()
    run=args.run.resolve()
    if not run.is_relative_to((ROOT/'results').resolve()):
        raise ValueError('Run outside supplement directory')
    manifest=json.loads((run/'manifest.json').read_text(encoding='utf-8'))
    complete=(run/'completion.json').exists() and manifest['scope']!='compatibility_smoke_only'
    if not complete and not args.allow_partial:
        raise RuntimeError('Incomplete/smoke run cannot be published as completed evidence')
    out=run/('analysis' if complete else 'analysis_preview')
    out.mkdir(exist_ok=True)
    figs=out/'figures';figs.mkdir(exist_ok=True)
    bins=json.loads((ROOT/'evidence/ma_size_bins.json').read_text(encoding='utf-8'))['boundaries_pixels']
    rows=[];lesions={'full':[],'no_h4':[]};examples=[];picked={k:0 for k in ['recovered','still_missed','new_fp']}
    ma_positive_images=0
    cv2.setNumThreads(2)
    for sid in manifest['sample_ids']:
        record=run/'per_image'/(sid+'.json')
        if not record.exists():
            if complete:raise FileNotFoundError(record)
            continue
        r=json.loads(record.read_text(encoding='utf-8'))
        pred=run/'predictions'/(sid+'.npz')
        assert sha(pred)==r['mask_sha256']
        label_path=DATA/'labels'/manifest['split']/(sid+'.png')
        assert sha(label_path)==r['label_sha256']
        label=np.asarray(Image.open(label_path))
        ma_positive_images += int((label==4).any())
        with np.load(pred,allow_pickle=False) as z:
            masks={k:z[k] for k in NAMES}
        assert all(b.dtype==bool and b.shape==(4,*label.shape) for b in masks.values())
        assert np.array_equal(masks['full'][:3],masks['no_h4'][:3])
        for k,b in masks.items():
            assert recompute_binary(b,label).tolist()==r['counts'][k],sid+' '+k
        rows.append(r)
        lr={}
        for method in lesions:
            result=match_lesions(label==4,masks[method][3],label!=255)
            result['size_groups']=grouped_counts(result,bins)
            result['sid']=sid
            lesions[method].append(result)
            lr[method]=result
        mh={x['gt_id'] for x in lr['full']['matches']}
        nh={x['gt_id'] for x in lr['no_h4']['matches']}
        candidates={'recovered':sorted(mh-nh),'still_missed':sorted(set(lr['full']['missed_gt_ids'])&set(lr['no_h4']['missed_gt_ids']))}
        _,fp_labels,_,_=components(masks['full'][3]&(label!=255))
        touched=set(np.unique(fp_labels[masks['no_h4'][3]]).tolist())
        candidates['new_fp']=[i for i in lr['full']['false_pred_ids'] if i not in touched]
        for category,ids in candidates.items():
            for k in ids[:3-picked[category]]:
                examples.append({'category':category,'sid':sid,'component_id':k})
                picked[category]+=1
        if len(rows)%25==0:print(json.dumps({'analyzed_images':len(rows)}),flush=True)
    if not rows:raise ValueError('No verified images')
    if complete:assert len(rows)==manifest['expected_images']
    n=len(rows);rng=np.random.default_rng(20261023);indices=rng.integers(0,n,(2000,n))
    arrays={k:np.asarray([r['counts'][k] for r in rows],dtype=np.int64) for k in NAMES}
    scores={k:bootstrap_scores(a,indices) for k,a in arrays.items()}
    pixels=[]
    for k in NAMES:
        s=summary(arrays[k]);s.pop('counts')
        pixels.append({'method':k,'description':NAMES[k],'n_images':n,**s,
                       'miou_ci_low_pct':ci(scores[k])[0],'miou_ci_high_pct':ci(scores[k])[1],
                       'full_minus_row_pp':float(scores['full'].mean()-scores[k].mean())})
    for row in pixels:
        row['full_minus_row_pp']=pixels[0]['miou_pct']-row['miou_pct']
    with (out/'component_table.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(pixels[0]));w.writeheader();w.writerows(pixels)
    comparisons={k:{'full_minus_row_pp':pixels[0]['miou_pct']-next(r['miou_pct'] for r in pixels if r['method']==k),
                    'paired_image_ci95_pp':ci(scores['full']-scores[k])} for k in NAMES if k!='full'}
    lesion_rows=[]
    for k,items in lesions.items():
        a=np.asarray([[r['tp'],r['fp'],r['fn'],r['gt_count'],r['pred_count']] for r in items],dtype=np.int64)
        totals=a.sum(axis=0);resampled=a[indices].sum(axis=1)
        row={'method':k,'n_images':n,'MA_positive_images':ma_positive_images,'gt_lesions':int(totals[3]),'pred_lesions':int(totals[4]),'detected_lesions':int(totals[0]),
             'false_lesions':int(totals[1]),'missed_lesions':int(totals[2]),
             'detection_pct':float(totals[0]/max(totals[3],1)*100),'FP_per_image':float(totals[1]/n),
             'lesion_precision_pct':float(totals[0]/max(totals[4],1)*100),
             'detection_ci95_pct':ci(resampled[:,0]/np.maximum(resampled[:,3],1)*100),
             'FP_per_image_ci95':ci(resampled[:,1]/n),'size_groups':{}}
        for group in ['small','medium','large']:
            ga=np.asarray([[r['size_groups'][group]['gt'],r['size_groups'][group]['detected']] for r in items])
            g=ga.sum(axis=0);b=ga[indices].sum(axis=1)
            row['size_groups'][group]={'gt':int(g[0]),'detected':int(g[1]),'detection_pct':float(g[1]/g[0]*100) if g[0] else None,
                                       'ci95_pct':ci(b[:,1]/np.maximum(b[:,0],1)*100)}
        lesion_rows.append(row)
    flat=[]
    for r in lesion_rows:
        for group,g in r['size_groups'].items():
            flat.append({'method':r['method'],'area_group':group,'gt':g['gt'],'detected':g['detected'],
                         'detection_pct':g['detection_pct'],'ci_low_pct':g['ci95_pct'][0],'ci_high_pct':g['ci95_pct'][1],
                         'FP_per_image_all_sizes':r['FP_per_image']})
    with (out/'ma_size_table.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    lesion_pairs={}
    for k,items in lesions.items():
        a=np.asarray([[r['tp'],r['fp'],r['gt_count']] for r in items],dtype=np.int64)
        b=a[indices].sum(axis=1)
        lesion_pairs[k]={'detection_pct':b[:,0]/np.maximum(b[:,2],1)*100,'FP_per_image':b[:,1]/n}
        for group in ['small','medium','large']:
            g=np.asarray([[r['size_groups'][group]['detected'],r['size_groups'][group]['gt']] for r in items])
            b=g[indices].sum(axis=1)
            lesion_pairs[k][group+'_detection_pct']=b[:,0]/np.maximum(b[:,1],1)*100
    lesion_differences={metric:{'full_minus_no_h4_ci95':ci(lesion_pairs['full'][metric]-lesion_pairs['no_h4'][metric])}
                        for metric in lesion_pairs['full']}
    report={'status':'COMPLETE' if complete else 'PREVIEW_NOT_FINAL','split':manifest['split'],'n_images':n,
            'pixel_table':pixels,'pixel_comparisons':comparisons,'lesion_table':lesion_rows,
            'MA_full_minus_no_h4_paired_intervals':lesion_differences,
            'bins_pixels':bins,'matching':'8-connected, IoU>=0.10, maximum-cardinality then maximum-IoU, one-to-one',
            'bootstrap':{'unit':'image','draws':2000,'seed':20261023,'training_seed_variance_included':False},
            'baseline':'UNAVAILABLE: historical split613 seed42 u32000 checkpoint not recovered; no new baseline training',
            'independent_generalization':'NOT_ESTABLISHED','runtime':'new local replay; see manifest and runtime amendment'}
    write_json(out/'summary.json',report)
    write_json(out/'lesion_metrics.json',lesions)
    first={};panels=[]
    for example in examples:
        panel=draw_example(example,run,manifest['split'],figs)
        if example['category'] not in first:first[example['category']]=panel
    if first:
        contact=Image.new('RGB',(1000,324*len(first)),'white')
        for i,category in enumerate(['recovered','still_missed','new_fp']):
            if category in first:panels.append(first[category])
        for i,panel in enumerate(panels):contact.paste(panel,(0,324*i))
        contact.save(figs/'ma_examples_contact.png',dpi=(300,300))
    write_json(out/'example_selection.json',{'rule':'first three by sid and component ID per preregistered outcome','examples':examples,'category_counts_selected':picked})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':8,'axes.labelsize':8,'legend.fontsize':7,'pdf.fonttype':42,'savefig.bbox':'tight'})
    fig,ax=plt.subplots(1,2,figsize=(6.9,2.8),gridspec_kw={'width_ratios':[2,1]})
    x=np.arange(3);width=.34
    for j,row in enumerate(lesion_rows):
        groups=[row['size_groups'][g] for g in ['small','medium','large']]
        y=np.array([g['detection_pct'] if g['detection_pct'] is not None else 0 for g in groups])
        err=np.array([[v-g['ci95_pct'][0],g['ci95_pct'][1]-v] for g,v in zip(groups,y)]).T
        color=['#0072B2','#D55E00'][j]
        ax[0].bar(x+(j-.5)*width,y,width,label=NAMES[row['method']],color=color,yerr=np.maximum(err,0),capsize=2)
        fp=row['FP_per_image'];lo,hi=row['FP_per_image_ci95']
        ax[1].bar(j,fp,color=color,yerr=[[max(0,fp-lo)],[max(0,hi-fp)]],capsize=3)
    ax[0].set_xticks(x,['Small\n<=32 px','Medium\n33-74 px','Large\n>74 px'])
    ax[0].set_ylabel('MA lesion detection (%)');ax[0].set_ylim(0,100);ax[0].legend(frameon=False)
    ax[1].set_xticks([0,1],['Full','No H4']);ax[1].set_ylabel('False positive lesions / image')
    for a in ax:
        a.spines[['top','right']].set_visible(False);a.grid(axis='y',alpha=.15);a.set_axisbelow(True)
    if not complete:fig.suptitle('PREVIEW — incomplete or compatibility run',color='#A33')
    fig.tight_layout()
    for suffix in ['pdf','svg','png']:fig.savefig(figs/('ma_size_and_fp.'+suffix),dpi=300)
    plt.close(fig)
    print(json.dumps({'status':report['status'],'n_images':n,'pixel_table':pixels,'output':str(out)},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
