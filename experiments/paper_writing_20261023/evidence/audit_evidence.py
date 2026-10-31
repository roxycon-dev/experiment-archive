"""Offline evidence indexing and integer-count recomputation; never imports model code."""
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime, timezone
import csv, hashlib, json, os, shutil, tarfile, zipfile
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parents[1]
E=OUT/'evidence'; E.mkdir(exist_ok=True)
FD=ROOT/'HRDecoder_FinalDecision_20261005'
OLD=next(FD.glob('*/evidence/final_locked_test_evidence.json')).parent
SR=FD/'execution/server_results'
def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p,v): Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
def sha(p): return hashlib.file_digest(open(p,'rb'),'sha256').hexdigest()
def csvwrite(p, rows, fields=None):
    with open(p,'w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=fields or list(rows[0]));w.writeheader();w.writerows(rows)
def text(p,s): Path(p).write_text(s,encoding='utf-8')

# Metadata covers all preexisting workspace files. Generated paper project is recorded separately.
files=[]; errors=[]; agents=[]
for base, dirs, names in os.walk(ROOT,onerror=lambda x: errors.append(str(x)),followlinks=False):
    dirs.sort();names.sort()
    for name in names:
        p=Path(base)/name
        if OUT in p.parents: continue
        try:
            st=p.stat(); rel=p.relative_to(ROOT).as_posix(); suf=p.suffix.lower()
            role=('weight' if suf in {'.pth','.pt','.ckpt','.safetensors'} else 'archive' if suf in {'.zip','.gz','.tgz','.tar','.7z','.rar','.xz'} else 'image' if suf in {'.png','.jpg','.jpeg','.tif','.tiff','.bmp','.svg'} else 'result/config' if suf in {'.json','.jsonl','.csv','.tsv','.yaml','.yml'} else 'code' if suf in {'.py','.sh','.ps1','.cpp','.h','.cu'} else 'report' if suf in {'.md','.pdf','.docx','.tex','.bib'} else 'other')
            files.append({'path':rel,'bytes':st.st_size,'modified_utc':datetime.fromtimestamp(st.st_mtime,timezone.utc).isoformat(),'extension':suf,'role':role})
            if name=='AGENTS.md': agents.append(rel)
        except OSError as ex: errors.append(str(ex))
csvwrite(E/'workspace_file_metadata.csv',files)
groups=defaultdict(lambda: {'files':0,'bytes':0,'roles':Counter()})
for r in files:
    a=groups[r['path'].split('/')[0]];a['files']+=1;a['bytes']+=r['bytes'];a['roles'][r['role']]+=1
summary={'created_utc':datetime.now(timezone.utc).isoformat(),'workspace':str(ROOT),'excludes_generated_project':str(OUT),'n_files':len(files),'bytes':sum(x['bytes'] for x in files),'roles':Counter(x['role'] for x in files),'top_level':groups,'errors':errors,'AGENTS_files':agents}
save(E/'workspace_inventory_summary.json',summary)

# Header-only archive listing. Never extracts; large compressed tar and unsupported archives remain explicit.
archive_rows=[];members=[]
for r in files:
    if r['role']!='archive':continue
    p=ROOT/r['path'];status='';n=0
    try:
        if zipfile.is_zipfile(p):
            with zipfile.ZipFile(p) as z:
                for item in z.infolist():
                    if n>=200000:status='member-limit';break
                    members.append({'archive':r['path'],'member':item.filename,'bytes':item.file_size});n+=1
            status=status or 'listed-zip-central-directory'
        elif r['bytes']<=256*1024**2 and (p.suffix.lower() in {'.tar','.tgz'} or p.name.endswith(('.tar.gz','.tar.xz'))):
            with tarfile.open(p,'r:*') as z:
                for item in z:
                    if n>=200000:status='member-limit';break
                    members.append({'archive':r['path'],'member':item.name,'bytes':item.size});n+=1
            status=status or 'listed-tar-headers'
        else:status='metadata-only: unsupported format or compressed tar >256MiB; no extraction'
    except Exception as ex:status='list-error: '+str(ex)
    archive_rows.append({**r,'members_listed':n,'listing_status':status})
csvwrite(E/'archive_inventory.csv',archive_rows, list(files[0])+['members_listed','listing_status'])
csvwrite(E/'archive_members.csv',members,['archive','member','bytes'])

# Portable snapshots with hashes of exactly the evidence bytes read.
sources=[]
def snapshot(p,name):
    p=Path(p);dest=E/name;shutil.copyfile(p,dest)
    sources.append({'original':str(p),'snapshot':dest.relative_to(OUT).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p),'read_scope':'JSON fully parsed or text inspected'})
    return read(dest) if dest.suffix=='.json' else dest
test=snapshot(OLD/'final_locked_test_evidence.json','locked_test_source.json')
val=snapshot(OLD/'four_arm_saved_validation.json','validation_source.json')
base=snapshot(OLD/'original_HRDecoder_baseline_evidence.json','historical_baseline_source.json')
ledger=snapshot(OLD/'historical_experiment_ledger.json','historical_experiment_ledger.json')
snapshot(OLD/'historical_experiment_ledger.csv','historical_experiment_ledger.csv')
snapshot(SR/'reports/final_selection_lock.json','selection_lock.json')
snapshot(FD/'execution/reports/single_seed_parallel_amendment.json','single_seed_amendment.json')
snapshot(SR/'configs/protocol.json','original_protocol.json')
snapshot(SR/'configs/p0_identity.json','p0_identity.json')
snapshot(SR/'configs/parent_Q16k.json','Q_parent_identity.json')
snapshot(SR/'reports/final_test.json','original_final_test.json')
hpath=next(ROOT.glob('HRDecoder_*20260916/audit_runs/audit_20260916_A/experiments/h7_test.json'))
h=snapshot(hpath,'ma_correction_source.json')
p0=snapshot(ROOT/'HRDecoder_Ready_20261002/best_pipeline/final_test.json','frozen_partner_source.json')

checks=[]
def check(name, condition, detail=''):
    checks.append({'check':name,'pass':bool(condition),'detail':detail})
def metrics(ct):
    ct=np.asarray(ct,dtype=np.int64);tp,fp,fn=ct.T
    return {'iou_pct':tp/np.maximum(tp+fp+fn,1)*100,'dice_pct':2*tp/np.maximum(2*tp+fp+fn,1)*100,'precision_pct':tp/np.maximum(tp+fp,1)*100,'recall_pct':tp/np.maximum(tp+fn,1)*100}
classes=['EX','HE','SE','MA'];result_rows=[]
for label,key,reportkey in [('locked','new_counts','test_per_class'),('previous','R_counts','R_test_per_class')]:
    arr=np.asarray([x[key] for x in test['per_image']],dtype=np.int64)
    totals=arr.sum(axis=0);m=metrics(totals)
    for i,c in enumerate(classes):
        check(label+' '+c+' integer total',list(totals[i])==[test[reportkey][c][k] for k in ['tp','fp','fn']])
        for metric in m:check(label+' '+c+' '+metric,abs(float(m[metric][i])-test[reportkey][c][metric])<1e-10)
        result_rows.append({'system':label,'split':'test','class':c,'tp':int(totals[i,0]),'fp':int(totals[i,1]),'fn':int(totals[i,2]),**{k:float(v[i]) for k,v in m.items()}})
    check(label+' macro IoU',abs(float(m['iou_pct'].mean())-test['locked_test_miou_pct' if label=='locked' else 'R_test_miou_pct'])<1e-10)
csvwrite(E/'test_class_metrics.csv',result_rows)
per_image=[]
for x in test['per_image']:
    for j,c in enumerate(classes):
        per_image.append({'sid':x['sid'],'class':c,**{label+'_'+k:x[key][j][i] for label,key in [('locked','new_counts'),('previous','R_counts')] for i,k in enumerate(['tp','fp','fn'])}})
csvwrite(E/'test_per_image_counts.csv',per_image)
check('227 unique image identities',len({x['sid'] for x in test['per_image']})==len(test['per_image'])==227)
old=np.asarray([x['R_counts'] for x in test['per_image']]);new=np.asarray([x['new_counts'] for x in test['per_image']])
rng=np.random.default_rng(100303);ix=rng.integers(0,len(old),(2000,len(old)))
def bscore(a):
    z=a[ix].sum(axis=1);return (z[:,:,0]/np.maximum(z.sum(axis=2),1)).mean(axis=1)*100
delta=bscore(new)-bscore(old);ci=np.quantile(delta,[.025,.975])
check('paired image bootstrap exact reproduction',np.allclose(ci,test['conditional_bootstrap_from_original_final_test']['paired_image_percentile95_pp'],rtol=0,atol=1e-12))
csvwrite(E/'bootstrap_draws.csv',[{'draw':i,'delta_pp':float(v)} for i,v in enumerate(delta)])
ma_rows=[]
for label in ['fused','final']:
    ct=[[h[label]['counts'][c][k] for k in ['TP','FP','FN']] for c in classes];m=metrics(ct)
    for i,c in enumerate(classes):
        check('MA correction '+label+' '+c,abs(float(m['iou_pct'][i])/100-h[label]['per_class'][c])<1e-12)
        ma_rows.append({'stage':label,'class':c,**{k:float(v[i]) for k,v in m.items()},**{k.lower():h[label]['counts'][c][k] for k in ['TP','FP','FN']}})
check('MA correction preserves non-target integer counts',all(h['fused']['counts'][c]==h['final']['counts'][c] for c in classes[:3]))
csvwrite(E/'ma_correction_metrics.csv',ma_rows)
vrows=[]
for r in val['records']:
    p=Path(r['source_path'])
    if p.is_file():
        check('validation raw hash '+r['label']+' '+str(r['update']),sha(p)==r['source_sha256'])
        raw=read(p)
        check('validation original image identity count '+r['label']+str(r['update']),len({x['sid'] for x in raw['per_image']})==76)
    else:check('validation source available '+r['label'],False,str(p))
    for formula,o in r['outputs'].items():
        if p.is_file():
            raw_totals=np.asarray([z[formula] for z in raw['per_image']],dtype=np.int64).sum(axis=0)
            expected=np.asarray([[o['per_class'][c][k] for k in ['tp','fp','fn']] for c in classes])
            check('validation raw per-image integer sum '+r['label']+str(r['update'])+formula,np.array_equal(raw_totals,expected))
            check('validation raw per-image macro '+r['label']+str(r['update'])+formula,abs(float(metrics(raw_totals)['iou_pct'].mean())-o['miou_pct'])<1e-10)
        for c,cc in o['per_class'].items():
            check('val count metric '+r['label']+str(r['update'])+formula+c,abs(cc['tp']/max(cc['tp']+cc['fp']+cc['fn'],1)*100-cc['iou_pct'])<1e-10)
        vrows.append({'arm':r['arm'],'seed':r['seed'],'update':r['update'],'split':r['split'],'n_images':r['n_images'],'selection_node':r['selection_node'],'formula':formula,'miou_pct':o['miou_pct'],**{c+'_iou_pct':o['per_class'][c]['iou_pct'] for c in classes},'source_path':r['source_path'],'source_sha256':r['source_sha256']})
csvwrite(E/'validation_metrics.csv',vrows)
endpoint=[x for x in vrows if x['arm'] in ['C','S','D','SD'] and x['update'] in [20000,32000] and x['formula'] in ['single','fixed_fusion','half_fusion']]
csvwrite(E/'registered_endpoints.csv',endpoint)
derived={'ma_correction_miou_before_pct':h['fused']['mIoU']*100,'ma_correction_miou_after_pct':h['final']['mIoU']*100,'ma_correction_miou_gain_pp':(h['final']['mIoU']-h['fused']['mIoU'])*100,'ma_correction_MA_gain_pp':(h['final']['per_class']['MA']-h['fused']['per_class']['MA'])*100,'bootstrap95_lower_pp':float(ci[0]),'bootstrap95_upper_pp':float(ci[1]),'bootstrap_confidence_pct':(.975-.025)*100,'ma_precision_delta_vs_previous_pp':test['test_per_class']['MA']['precision_pct']-test['R_test_per_class']['MA']['precision_pct'],'ma_recall_delta_vs_previous_pp':test['test_per_class']['MA']['recall_pct']-test['R_test_per_class']['MA']['recall_pct'],'approx_ma_iou_gain_vs_historical_baseline_pp':test['test_per_class']['MA']['iou_pct']-base['historical_test']['per_class_iou_pct_reported']['MA']}
save(E/'derived_statistics.json',derived)
overview=[]
def overviewrow(label,miou,values,source,grade):
    overview.append({'system':label,'split':'test','miou_pct':miou,**dict(zip([c+'_iou_pct' for c in classes],values)),'source':source,'interpretation':grade})
overviewrow('Original HRDecoder',42.2,list(base['historical_test']['per_class_iou_pct_reported'].values()),'evidence/historical_baseline_source.json','rounded historical report; not matched reproduction')
for label in ['fused','final']:
    overviewrow('Four-member HRDecoder mixture '+label,h[label]['mIoU']*100,[h[label]['per_class'][c]*100 for c in classes],'evidence/ma_correction_source.json','paired correction comparison, historical test reused')
overviewrow('MA-corrected HRDecoder frozen re-export',p0['champion_miou']*100,[v*100 for v in p0['champion_per_class']],'evidence/frozen_partner_source.json','later export; distinct integer counts from correction audit')
overviewrow('Frozen HRDecoder-ConvNeXt partner',p0['selected_pipeline_miou']*100,[v*100 for v in p0['selected_pipeline_per_class']],'evidence/frozen_partner_source.json','system addition, not isolated architecture effect')
for label,section,miou in [('Pre-spatial system','R_test_per_class','R_test_miou_pct'),('Locked complete system','test_per_class','locked_test_miou_pct')]:
    overviewrow(label,test[miou],[test[section][c]['iou_pct'] for c in classes],'evidence/locked_test_source.json','single-seed system comparison; test historically reused')
csvwrite(E/'system_overview.csv',overview)
save(E/'recomputation_audit.json',{'mode':'offline arithmetic only; no model execution','checks':checks,'all_pass':all(x['pass'] for x in checks),'numpy_version':np.__version__,'formulas':{'IoU':'TP/(TP+FP+FN)','Dice':'2TP/(2TP+FP+FN)','precision':'TP/(TP+FP)','recall':'TP/(TP+FN)','macro':'arithmetic mean of four global per-class IoUs'},'bootstrap':'paired image resampling with replacement; default_rng(100303), 2000 draws, np.quantile [.025,.975]; fixed model pair and selection; not patient or seed uncertainty'})

# Every numeric entry has an exact portable selector; percentages already stored in percent use raw representation.
entries=[]
def num(value,source,pointer,note=''):
    entries.append({'value':float(value),'source':source,'selector':{'kind':'json-pointer','pointer':pointer},'aggregate':'identity','representations':['raw'],'tolerance':0.000051/max(abs(float(value)),1),'note':note})
for k in ['n_images','locked_test_miou_pct','R_test_miou_pct','gain_vs_R_pp','historical_baseline_test_pct_reported','approx_gain_vs_baseline_pp']:
    num(test[k],'evidence/locked_test_source.json','/'+k,k)
for section in ['test_per_class','R_test_per_class','gain_vs_R_per_class_pp']:
    for c,o in test[section].items():
        if isinstance(o,dict):
            for k,v in o.items():num(v,'evidence/locked_test_source.json',f'/{section}/{c}/{k}',f'{section} {c} {k}')
        else:num(o,'evidence/locked_test_source.json',f'/{section}/{c}',section+' '+c)
for k,v in base['reported_split_sizes'].items():num(v,'evidence/historical_baseline_source.json',f'/reported_split_sizes/{k}','custom split '+k)
num(base['best_validation']['miou_pct'],'evidence/historical_baseline_source.json','/best_validation/miou_pct','historical validation best')
for c,v in base['historical_test']['per_class_iou_pct_reported'].items():num(v,'evidence/historical_baseline_source.json',f'/historical_test/per_class_iou_pct_reported/{c}','rounded historical reference; not matched reproduction')
for i,r in enumerate(val['records']):
    if r['arm'] in ['C','S','D','SD'] and r['update'] in [20000,32000]:
        for fo in ['single','fixed_fusion','half_fusion']:
            num(r['outputs'][fo]['miou_pct'],'evidence/validation_source.json',f'/records/{i}/outputs/{fo}/miou_pct',f"{r['arm']} EMA {r['update']} val {fo}")
for i,r in enumerate(val['paired_registered_endpoint_contrasts']):
    for k,v in r['contrasts'].items():num(v,'evidence/validation_source.json',f'/paired_registered_endpoint_contrasts/{i}/contrasts/{k}',f"val {r['update']} {r['formula']} {k}")
for k,v in derived.items():num(v,'evidence/derived_statistics.json','/'+k,'Recomputed in evidence/audit_evidence.py from integer counts or source paired image counts')
for label in ['fused','final']:
    entries.append({'value':h[label]['mIoU'],'source':'evidence/ma_correction_source.json','selector':{'kind':'json-pointer','pointer':f'/{label}/mIoU'},'representations':['raw','percent'],'tolerance':0.0000012,'note':'Historical paired correction on four-member flip mixture; separate from later frozen export'})
    for c,v in h[label]['per_class'].items():entries.append({'value':v,'source':'evidence/ma_correction_source.json','selector':{'kind':'json-pointer','pointer':f'/{label}/per_class/{c}'},'representations':['raw','percent'],'tolerance':0.000002,'note':label+' '+c+' exact integer-count result'})
for label in ['champion','selected_pipeline']:
    entries.append({'value':p0[label+'_miou'],'source':'evidence/frozen_partner_source.json','selector':{'kind':'json-pointer','pointer':'/'+label+'_miou'},'representations':['raw','percent'],'tolerance':0.0000012,'note':label+' later frozen export'})
    for i,c in enumerate(classes):entries.append({'value':p0[label+'_per_class'][i],'source':'evidence/frozen_partner_source.json','selector':{'kind':'json-pointer','pointer':f'/{label}_per_class/{i}'},'representations':['raw','percent'],'tolerance':0.000002,'note':label+' '+c})
def protocol_numbers(obj,pointer=''):
    if isinstance(obj,dict):
        for k,v in obj.items():protocol_numbers(v,pointer+'/'+k.replace('~','~0').replace('/','~1'))
    elif isinstance(obj,list):
        for i,v in enumerate(obj):protocol_numbers(v,pointer+'/'+str(i))
    elif isinstance(obj,(int,float)) and not isinstance(obj,bool):num(obj,'evidence/original_protocol.json',pointer,'Actual numeric protocol field; execution seed amendment overrides original two-seed scope')
protocol_numbers(read(E/'original_protocol.json'))
num(1600,'evidence/ma_correction_source.json','/heads/MA','selected MA head update')
num(2000,'evidence/locked_test_source.json','/conditional_bootstrap_from_original_final_test/resamples','paired bootstrap resamples')
import re
fusion_values=[float(v) for v in re.findall(r'(?<![A-Za-z0-9])(\.\d+)\*',val['reference_formulas']['fixed_fusion'])]
save(E/'fusion_recipe_numeric.json',{'source':'evidence/validation_source.json#/reference_formulas/fixed_fusion','literal_formula':val['reference_formulas']['fixed_fusion'],'partner_weight':fusion_values[0],'branch_weight':fusion_values[1]})
for key,value in zip(['partner_weight','branch_weight'],fusion_values):entries.append({'value':value,'source':'evidence/fusion_recipe_numeric.json','selector':{'kind':'json-pointer','pointer':'/'+key},'representations':['raw','percent'],'tolerance':1e-8,'note':'Numeric parse of the recorded locked fixed_fusion formula; not a fitted weight'})
save(OUT/'numeric_evidence.json',{'schema_version':'ai-research-writing/numeric-evidence-v2','entries':entries})
save(E/'source_snapshot_manifest.json',{'sources':sources})
later=read(ROOT/'HRDecoder_ArchitectureDesign_20261008/final_report/report_data.json')
later_rows=[]
for name,r in later['runs'].items():
    rec=r['records'];latest=rec[-1] if rec else {};best=max(rec,key=lambda z:z['mIoU']) if rec else {}
    later_rows.append({'family':'ArchitectureDesign 20261008','run':name,'description':r.get('description'),'source':r.get('source'),'latest_complete_validation':latest,'best_validation':best,'main_fixed40k_completed':latest.get('step')==40000,'test_result':None,'interpretation':'screening/history; not matched evidence for selected ensemble; no performance inferred beyond recorded endpoints'})
np_source=ROOT/'HRDecoder_NativePyramid_20261011/server_execution_20261016/completion_20261017/completion_report.json'
np_record=read(np_source)
later_rows.append({'family':'NativePyramid','run':'v3 run_seed42_launch02','source':str(np_source),'source_sha256':sha(np_source),'latest_complete_validation':np_record['final_validation'],'best_validation':np_record['best_validation_secondary'],'test_accessed':False,'identity':np_record['source_sha256'],'interpretation':'40k raw val only, separate architecture; no test result'})
later_rows.append({'family':'JointContext','run':'r4','source':str(ROOT/'HRDecoder_NativePyramid_20261011/review_20261018/jointcontext_spec_r4.json'),'status':'SPECIFICATION_AND_REDUCED_CPU_REFERENCE_ONLY','trained_full_model':False,'interpretation':'no trained accuracy result; cannot inherit old scores'})
save(E/'later_experiment_ledger.json',{'scope':'internal historical evidence; excluded from manuscript failed-experiment tables under current author scope','records':later_rows})
prior=read(OLD/'workspace_source_hashes.json');availability=[]
for r in prior['sources']:
    p=Path(r['path']);exists=p.is_file();size=p.stat().st_size if exists else None
    status=('missing' if not exists else 'hash-cap metadata only' if size>64*1024**2 else 'match' if sha(p)==r['sha256'] else 'MISMATCH')
    availability.append({'original':r['path'],'exists':exists,'bytes':size,'reported_sha256':r['sha256'],'check':status})
csvwrite(E/'prior_source_availability.csv',availability)
print(json.dumps({'files':len(files),'archives':len(archive_rows),'archive_members':len(members),'checks':len(checks),'all_pass':all(x['pass'] for x in checks),'numeric_entries':len(entries),'agents':agents},ensure_ascii=False))
