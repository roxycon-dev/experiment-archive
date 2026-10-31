"""Independently aggregate saved supplement counts; author bilingual tabular fragments."""
from pathlib import Path
from collections import Counter
import csv,hashlib,json,os,zipfile
import numpy as np

OUT=Path(__file__).resolve().parents[1];ROOT=OUT.parent/'paper_supplement_20261023';E=OUT/'evidence'
EN=OUT/'paper/tables_r4';ZH=OUT/'paper_zh/tables_r4'
EN.mkdir(exist_ok=True,parents=True);ZH.mkdir(exist_ok=True,parents=True)
sources={};checks=[];counts_packet={};names=['EX','HE','SE','MA']
def sha(p):return hashlib.file_digest(open(p,'rb'),'sha256').hexdigest()
def read(p):
    p=Path(p);sources[str(p.relative_to(ROOT))]={'bytes':p.stat().st_size,'sha256':sha(p)}
    return json.loads(p.read_text(encoding='utf-8-sig'))
def write(p,a):p.write_text(json.dumps(a,ensure_ascii=False,indent=2),encoding='utf-8')
def check(name,a,b,tol=1e-9):checks.append({'name':name,'pass':bool(np.allclose(a,b,atol=tol,rtol=0))})
def score(a):
    c=np.asarray(a,dtype=np.int64).sum(0);tp,fp,fn=c.T;i=tp/np.maximum(tp+fp+fn,1)*100
    return {'mIoU_pct':float(i.mean()),**{k+'_IoU_pct':float(i[j]) for j,k in enumerate(names)},'MA_P_pct':float(tp[3]/max(tp[3]+fp[3],1)*100),'MA_R_pct':float(tp[3]/max(tp[3]+fn[3],1)*100),'global_counts':c.tolist()}
def raw_counts(folder,expected):
    records=[read(p) for p in sorted((folder/'per_image').glob('*.json'))];assert len(records)==expected,(folder,len(records))
    assert len({r['sid'] for r in records})==expected
    modes=list(records[0]['counts']);data={m:np.asarray([r['counts'][m] for r in records],np.int64) for m in modes}
    counts_packet[str(folder.relative_to(ROOT))]={'sids':[r['sid'] for r in records],'counts':{m:a.tolist() for m,a in data.items()},'record_identity_SHA':[{k:r[k] for k in ['sid','image_sha256','label_sha256','mask_sha256','P0_sha256'] if k in r} for r in records]}
    return data,records
def boot(a,b,seed=20261023):
    n=len(a);ix=np.random.default_rng(seed).integers(0,n,(2000,n))
    def s(x):
        z=x[ix].sum(1);return (z[:,:,0]/np.maximum(z.sum(2),1)).mean(1)*100
    return np.quantile(s(a)-s(b),[.025,.975]).tolist()
def csvwrite(p,rows):
    with open(p,'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

result={'evaluation_boundaries':{'main_components':'Phase1 BF16/reference systems plus recovered original FP32 HRDecoder; same original image/GT coordinates; no equal-compute assertion','paired':'All six C/S endpoints uniformly re-evaluated FP32/reference; one fixed F12_seed42 parent;20k EMA inside32k schedule','external':'IDRiD-derived27 frozen cohort, user-confirmed use history, transformed square geometry; patient IDs unavailable'},'main':{},'paired_details':[],'paired_summary':[],'MA':{},'efficiency':[]}
stage={};stagearr={};basearr={}
original=read(ROOT/'results/baseline_original_local/analysis/verified_summary.json')
base_run=ROOT/'results/baseline_original_local'
for cohort,n,stagepath in [('ddr',227,'ddr_test_local_v1'),('idrid',27,'idrid_holdout_v1')]:
    folder=ROOT/'results'/stagepath;data,records=raw_counts(folder,n);stagearr[cohort]=data
    summary=read(folder/'analysis/summary.json');stage[cohort]=summary
    bdata,_=raw_counts(base_run/cohort,n);basearr[cohort]=bdata['baseline']
    rows=[]
    for model in ['baseline','full','no_h4','no_convnext','matched_c20']:
        m=score(bdata['baseline'] if model=='baseline' else data[model]);m.update(cohort=cohort,model=model)
        if model=='baseline':check(cohort+' original baseline',m['mIoU_pct'],original['cohorts'][cohort]['baseline']['mIoU_pct'])
        else:
            sr=next(x for x in summary['pixel_table'] if x['method']==model)
            for key in ['mIoU_pct']+[c+'_IoU_pct' for c in names]:check(cohort+model+key,m[key],sr['miou_pct' if key=='mIoU_pct' else key])
        rows.append(m)
    result['main'][cohort]=rows
    result.setdefault('component_pair_CI',{})[cohort]={model:{'delta_pp':rows[1]['mIoU_pct']-next(r for r in rows if r['model']==model)['mIoU_pct'],'CI95_pp':boot(data['full'],bdata['baseline'] if model=='baseline' else data[model])} for model in ['baseline','no_h4','no_convnext','matched_c20']}
    csvwrite(E/f'revision4_main_{cohort}.csv',[{k:v for k,v in r.items() if k!='global_counts'} for r in rows])

paired_saved=read(ROOT/'results/paired_seed_summary/summary.json');pairedarrays={};pairedrecords={};manifest_reference=None;p0_reference={}
for seed in [42,2024,3407]:
    for arm in ['C','S']:
        run=ROOT/'results/paired_seed59'/f'{arm}_seed{seed}';man=read(run/'manifest.json')
        assert man['job']['parent_full_sha256']=='68d7b5821fab1210b3336cf2a63b9de893f24fc32876f5255d7d662084b5e728'
        assert man['job']['completed']==man['job']['EMA_completed']==20000
        identity={k:man[k] for k in ['torch','CUDA','GPU','precision','helper_sha256']}
        if manifest_reference is None:manifest_reference=identity
        else:assert identity==manifest_reference
        for cohort,n in [('ddr',227),('idrid',27)]:
            data,recs=raw_counts(run/cohort,n);pairedarrays[arm,seed,cohort]=data
            p0={r['sid']:r['P0_sha256'] for r in recs}
            if cohort in p0_reference:assert p0_reference[cohort]==p0
            else:p0_reference[cohort]=p0
    for cohort in ['ddr','idrid']:
        for mode in ['single','fixed_fusion']:
            c=pairedarrays['C',seed,cohort][mode];s=pairedarrays['S',seed,cohort][mode];cs,ss=score(c),score(s)
            r={'seed':seed,'cohort':cohort,'mode':mode,'C_mIoU_pct':cs['mIoU_pct'],'S_mIoU_pct':ss['mIoU_pct'],'delta_pp':ss['mIoU_pct']-cs['mIoU_pct'],'CI95_pp':boot(s,c),'C_MA_IoU_pct':cs['MA_IoU_pct'],'S_MA_IoU_pct':ss['MA_IoU_pct']}
            old=next(z for z in paired_saved['per_seed'] if z['seed']==seed and z['cohort']==cohort and z['mode']==mode)
            check(str((seed,cohort,mode))+' delta',r['delta_pp'],old['S_minus_C_pp']);check(str((seed,cohort,mode))+' bootstrap',r['CI95_pp'],old['paired_image_bootstrap_CI95_pp'])
            result['paired_details'].append(r)
for cohort in ['ddr','idrid']:
    for mode in ['single','fixed_fusion']:
        r={'cohort':cohort,'mode':mode,'n_seeds':3};rows=[z for z in result['paired_details'] if z['cohort']==cohort and z['mode']==mode]
        for key in ['C_mIoU_pct','S_mIoU_pct','delta_pp']:
            a=np.asarray([z[key] for z in rows]);r[key+'_mean']=float(a.mean());r[key+'_sample_SD']=float(a.std(ddof=1))
        result['paired_summary'].append(r)
result['paired_runtime']=manifest_reference
result['cohort_sizes']={'DDR_test':len(stagearr['ddr']['full']),'IDRiD_derived_holdout':len(stagearr['idrid']['full'])}
result['frozen_evaluation']={'bootstrap_draws':2000,'bootstrap_seed':20261023,'threshold':.5,'adaptation_seeds':[42,2024,3407],'selected_EMA_update':man['job']['completed'],'learning_rate_schedule_horizon':32000}
csvwrite(E/'revision4_paired_details.csv',[{**r,'CI95_pp':json.dumps(r['CI95_pp'])} for r in result['paired_details']]);csvwrite(E/'revision4_paired_summary.csv',result['paired_summary'])

threeway=read(ROOT/'results/baseline_original_threeway/summary.json')
result['MA_comparisons']={}
for cohort,n,path in [('ddr',227,'ddr_test_local_v1'),('idrid',27,'idrid_holdout_v1')]:
    lesions=read(ROOT/'results'/path/'analysis/lesion_metrics.json');base=read(base_run/'analysis'/f'MA_lesions_{cohort}.json');rows=[]
    for model,rr in [('baseline',base),('no_h4',lesions['no_h4']),('full',lesions['full'])]:
        assert len(rr)==n
        for row in rr:
            assert row['tp']==len(row['matches']);assert row['tp']+row['fn']==row['gt_count'];assert row['tp']+row['fp']==row['pred_count']
            assert len({m['gt_id'] for m in row['matches']})==len({m['pred_id'] for m in row['matches']})==row['tp']
            assert all(m['iou']>=.1-1e-12 for m in row['matches'])
            groups={k:{'gt':0,'detected':0} for k in ['small','medium','large']}
            detected={m['gt_id'] for m in row['matches']}
            for index,area in enumerate(row['gt_areas'],1):
                group='small' if area<=32 else ('medium' if area<=74 else 'large')
                groups[group]['gt']+=1;groups[group]['detected']+=int(index in detected)
            if 'size_groups' in row:
                for key in groups:
                    for field in groups[key]:assert groups[key][field]==row['size_groups'][key][field]
            row['size_groups']=groups
        for size in ['all','small','medium','large']:
            tp=sum(r['tp'] for r in rr) if size=='all' else sum(r['size_groups'][size]['detected'] for r in rr)
            gt=sum(r['gt_count'] for r in rr) if size=='all' else sum(r['size_groups'][size]['gt'] for r in rr)
            fp=sum(r['fp'] for r in rr)
            z={'cohort':cohort,'model':model,'size':size,'GT':gt,'detected':tp,'detection_pct':100*tp/max(gt,1),'FP_per_image':fp/n if size=='all' else None,'lesion_precision_pct':100*tp/max(tp+fp,1) if size=='all' else None}
            old=next(a for a in threeway['MA_lesion_rows'] if a['cohort']==cohort and a['method']==model and a['size']==size)
            check(str((cohort,model,size))+' detection',z['detection_pct'],old['detection_pct']);check(str((cohort,model,size))+' GT',gt,old['GT'])
            rows.append(z)
    result['MA'][cohort]=rows;csvwrite(E/f'revision4_ma_{cohort}.csv',rows)
    ix=np.random.default_rng(20261023).integers(0,n,(2000,n))
    def lesion_samples(method):
        rr=sorted(lesions[method],key=lambda z:z['sid']);tp=np.asarray([z['tp'] for z in rr]);gt=np.asarray([z['gt_count'] for z in rr]);fp=np.asarray([z['fp'] for z in rr])
        return tp[ix].sum(1)/np.maximum(gt[ix].sum(1),1)*100,fp[ix].mean(1)
    full_detection,full_fp=lesion_samples('full');no_detection,no_fp=lesion_samples('no_h4')
    fullrow=next(z for z in rows if z['model']=='full' and z['size']=='all');norow=next(z for z in rows if z['model']=='no_h4' and z['size']=='all')
    result['MA_comparisons'][cohort]={'full_minus_without_correction_detection_pp':fullrow['detection_pct']-norow['detection_pct'],'detection_CI95_pp':np.quantile(full_detection-no_detection,[.025,.975]).tolist(),'FP_per_image_delta':fullrow['FP_per_image']-norow['FP_per_image'],'FP_per_image_CI95':np.quantile(full_fp-no_fp,[.025,.975]).tolist(),'bootstrap_draws':2000,'bootstrap_seed':20261023}
    check(cohort+' MA detection paired bootstrap',result['MA_comparisons'][cohort]['detection_CI95_pp'],threeway['comparisons'][cohort]['MA_full_minus_no_h4']['detection'])
    check(cohort+' MA FP paired bootstrap',result['MA_comparisons'][cohort]['FP_per_image_CI95'],threeway['comparisons'][cohort]['MA_full_minus_no_h4']['FP_per_image'])
bins=read(ROOT/'evidence/ma_size_bins.json');areas=np.asarray([a for r in bins['training_samples'] for a in r['areas']])
check('MA training area count',len(areas),6793);check('MA train-derived boundaries',np.quantile(areas,[1/3,2/3],method='nearest'),[32,74])
result['MA_bins']={'training_images':454,'training_components':len(areas),'boundaries_pixels':[32,74],'matching_IoU':.1,'connectivity':8}

bp=read(ROOT/'results/baseline_original_profile_local/summary.json');fp=read(ROOT/'results/full_system_profile_v1/analysis/summary.json')
assert bp['hardware']==fp['hardware'] and bp['sample_ids']==fp['profile_samples']
for model,ss,parameters,weight,peak in [('baseline',bp['pipeline_to_native_binary_mask_seconds'],bp['registered_parameters'],bp['inference_weight_artifact_bytes'],bp['max_single_process_allocated_peak_bytes']),('full',fp['pipeline_to_fused_mask_seconds'],fp['registered_parameters']['full_system_total_registered_parameters'],fp['registered_parameters']['weight_artifact_bytes_full'],fp['max_single_process_allocated_peak_bytes'])]:
    check(model+' timing mean',np.mean(ss['per_image']),ss['mean']);result['efficiency'].append({'model':model,'n_images':3,'mean_seconds':float(np.mean(ss['per_image'])),'parameters':parameters,'parameters_M':parameters/1e6,'weights_GiB':weight/1024**3,'single_process_allocated_GiB':peak/1024**3})
csvwrite(E/'revision4_efficiency.csv',result['efficiency'])

# Pure tabular fragments; manuscript author controls captions, labels and floating layout.
def tabular(headers,rows,align=None):
    return '\n'.join([r'\begin{tabular}{'+(align or ('l'+'r'*(len(headers)-1)))+'}',r'\hline',' & '.join(headers)+r'\\\hline']+[' & '.join(map(str,r))+r'\\' for r in rows]+[r'\hline',r'\end{tabular}'])+'\n'
def fmt(x,d=2):return f'{x:.{d}f}'
labels={'baseline':('Original HRDecoder','原HRDecoder'),'full':('Complete system','完整系统'),'no_h4':('Without MA correction','去除MA定向修正'),'no_convnext':('Without ConvNeXt','去除ConvNeXt'),'matched_c20':('Matched continuation','匹配续训替换')}
for language,folder in [(0,EN),(1,ZH)]:
    for cohort in ['ddr','idrid']:
        rows=[[labels[r['model']][language]]+[fmt(r[c+'_IoU_pct']) for c in names]+[fmt(r['mIoU_pct'])] for r in result['main'][cohort]]
        (folder/f'main_{cohort}.tex').write_text(tabular(['Method' if not language else '模型','EX','HE','SE','MA','mIoU'],rows),encoding='utf-8')
    rows=[]
    for r in result['paired_summary']:
        dat='DDR' if r['cohort']=='ddr' else 'IDRiD-derived';mode=('Single branch' if r['mode']=='single' else 'Fixed fusion') if not language else ('单分支' if r['mode']=='single' else '固定融合')
        rows.append([dat,mode]+[f"${fmt(r[k+'_mean'],4)}\\pm{fmt(r[k+'_sample_SD'],4)}$" for k in ['C_mIoU_pct','S_mIoU_pct','delta_pp']])
    (folder/'paired_summary.tex').write_text(tabular(['Dataset','Output','Continuation','Spatial adaptation',r'$\Delta$ (pp)'] if not language else ['数据','输出','续训','空间适配',r'$\Delta$（百分点）'],rows,'llrrr'),encoding='utf-8')
    rows=[]
    for r in result['paired_details']:
        rows.append(['DDR' if r['cohort']=='ddr' else 'IDRiD-derived',('Single' if r['mode']=='single' else 'Fusion') if not language else ('单支' if r['mode']=='single' else '融合'),str(r['seed']),fmt(r['C_mIoU_pct'],4),fmt(r['S_mIoU_pct'],4),fmt(r['delta_pp'],4),f"[{fmt(r['CI95_pp'][0],4)}, {fmt(r['CI95_pp'][1],4)}]"])
    (folder/'paired_details.tex').write_text(tabular(['Dataset','Output','Seed','Continuation','Spatial',r'$\Delta$ (pp)','Image 95\\% CI'] if not language else ['数据','输出','种子','续训','空间适配',r'$\Delta$','图像95\\%区间'],rows,'lllrrrr'),encoding='utf-8')
    for cohort in ['ddr','idrid']:
        rows=[]
        for r in result['MA'][cohort]:
            if cohort=='idrid' and r['size']!='all':continue
            size={'all':('All','全部'),'small':('Small','小'),'medium':('Medium','中'),'large':('Large','大')}[r['size']][language]
            rows.append([labels[r['model']][language],size,str(r['GT']),str(r['detected']),fmt(r['detection_pct']),fmt(r['FP_per_image']) if r['FP_per_image'] is not None else '--',fmt(r['lesion_precision_pct']) if r['lesion_precision_pct'] is not None else '--'])
        (folder/f'ma_{cohort}.tex').write_text(tabular(['Method','Size','GT','Detected','Detection\\%','FP/image','Precision\\%'] if not language else ['模型','大小','GT','检出','检出率\\%','FP/图','病灶P\\%'],rows,'llrrrrr'),encoding='utf-8')
    rows=[]
    for model in ['baseline','no_h4','full']:
        rr={r['size']:r for r in result['MA']['ddr'] if r['model']==model}
        rows.append([labels[model][language]]+[fmt(rr[s]['detection_pct']) for s in ['all','small','medium','large']]+[fmt(rr['all']['FP_per_image']),fmt(rr['all']['lesion_precision_pct'])])
    (folder/'ma_ddr_compact.tex').write_text(tabular(['Method','All\\%','Small\\%','Medium\\%','Large\\%','FP/image','Precision\\%'] if not language else ['模型','全部\\%','小\\%','中\\%','大\\%','FP/图','病灶P\\%'],rows),encoding='utf-8')
    rows=[[labels[r['model']][language],fmt(r['parameters_M']),fmt(r['mean_seconds'],3),fmt(r['weights_GiB'],3),fmt(r['single_process_allocated_GiB'],3)] for r in result['efficiency']]
    (folder/'efficiency.tex').write_text(tabular(['Method','Parameters (M)','Seconds/image','Weights (GiB)','Allocated (GiB)'] if not language else ['模型','注册参数(M)','秒/图','权重(GiB)','allocated(GiB)'],rows),encoding='utf-8')

write(E/'revision4_results.json',result);write(E/'revision4_input_counts.json',counts_packet);write(E/'revision4_source_manifest.json',{'root':str(ROOT),'sources':sources});write(E/'revision4_recomputation_audit.json',{'checks':checks,'all_pass':all(r['pass'] for r in checks),'scope':'All main/paired saved per-image JSON integer totals;12 exact paired image bootstraps;sample SD;per-image lesion matching-record invariants and aggregation;training-area quantiles;latency means. Mask/GT independent audit is recorded separately.'})
patch=[]
def numeric(a,pointer=''):
    if isinstance(a,dict):
        for k,v in a.items():numeric(v,pointer+'/'+k)
    elif isinstance(a,list):
        for i,v in enumerate(a):numeric(v,pointer+'/'+str(i))
    elif isinstance(a,(float,int)) and not isinstance(a,bool):patch.append({'value':float(a),'source':'evidence/revision4_results.json','selector':{'kind':'json-pointer','pointer':pointer},'representations':['raw'],'tolerance':.0051/max(abs(a),1),'note':'Revision4 recomputed from saved primary per-image counts; runtime/cohort identities must remain separate. Two-decimal table presentation.'})
numeric(result);write(E/'revision4_numeric_registry_patch.json',{'schema_version':'ai-research-writing/numeric-evidence-v2','entries':patch})
registry=json.loads((OUT/'numeric_evidence.json').read_text(encoding='utf-8'));registry['entries']=[r for r in registry['entries'] if r['source']!='evidence/revision4_results.json']+patch;write(OUT/'numeric_evidence.json',registry)
assert all(r['pass'] for r in checks)
print(json.dumps({'checks':len(checks),'all_pass':True,'patch_entries':len(patch),'tables':[p.name for p in EN.glob('*.tex')],'paired_summary':result['paired_summary']},ensure_ascii=False))
