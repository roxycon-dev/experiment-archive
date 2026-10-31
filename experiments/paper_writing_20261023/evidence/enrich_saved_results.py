"""Derive descriptive tables and re-layout existing real figure panels; no inference."""
from pathlib import Path
import base64,csv,hashlib,io,json,shutil,xml.etree.ElementTree as ET
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path(__file__).resolve().parents[1];ROOT=OUT.parent;E=OUT/'evidence';F=OUT/'figures'
TABLE=E/'tables';TABLE.mkdir(exist_ok=True)
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p,a):Path(p).write_text(json.dumps(a,ensure_ascii=False,indent=2),encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def pixels(a):return hashlib.sha256(np.asarray(a,dtype=np.uint8).tobytes()).hexdigest()
def csvsave(p,rows):
    with open(p,'w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def metrics(d):
    tp,fp,fn=[d[k] for k in ['tp','fp','fn']]
    return {'tp':tp,'fp':fp,'fn':fn,'iou_pct':100*tp/(tp+fp+fn),'dice_pct':200*tp/(2*tp+fp+fn),'precision_pct':100*tp/(tp+fp),'recall_pct':100*tp/(tp+fn)}

t=read(E/'locked_test_source.json');h=read(E/'ma_correction_source.json');v=read(E/'validation_source.json')
cl=['EX','HE','SE','MA'];classrows=[];stats={'source_inputs':{'test':'evidence/locked_test_source.json','correction':'evidence/ma_correction_source.json','validation':'evidence/validation_source.json'},'complete_system':{},'correction':{},'spatial_class_gains':[]}
for c in cl:
    m=metrics(t['test_per_class'][c]);stats['complete_system'][c]=m;classrows.append({'class':c,**m})
csvsave(E/'revision2_complete_class_metrics.csv',classrows)
for label in ['fused','final']:
    ct={k.lower():a for k,a in h[label]['counts']['MA'].items()};stats['correction'][label]=metrics(ct)
before,after=stats['correction']['fused'],stats['correction']['final']
stats['correction']['net_change']={k:after[k]-before[k] for k in before}
stats['correction']['net_change']['fn_relative_reduction_pct']=100*(before['fn']-after['fn'])/before['fn']
stats['correction']['net_change']['fp_relative_reduction_pct']=100*(before['fp']-after['fp'])/before['fp']
stats['correction']['net_change']['fp_reduction_pixels']=before['fp']-after['fp']
stats['correction']['net_change']['fn_reduction_pixels']=before['fn']-after['fn']
csvsave(E/'revision2_ma_correction_decomposition.csv',[{'stage':k,**stats['correction'][k]} for k in ['fused','final']])
for update in [20000,32000]:
    a={r['arm']:r for r in v['records'] if r['arm'] in ['C','S'] and r['update']==update}
    for formula in ['single','fixed_fusion']:
        z={'update':update,'formula':formula}
        for c in cl:
            z[c+'_iou_gain_pp']=a['S']['outputs'][formula]['per_class'][c]['iou_pct']-a['C']['outputs'][formula]['per_class'][c]['iou_pct']
        stats['spatial_class_gains'].append(z)
csvsave(E/'revision2_spatial_class_gains.csv',stats['spatial_class_gains'])
save(E/'revision2_result_statistics.json',stats)

lines=[r'\begin{tabular}{lrrrr}',r'\hline',r'Lesion & IoU (\%) & Dice (\%) & Precision (\%) & Recall (\%)\\\hline']
for row in classrows:lines.append(row['class']+' & '+' & '.join(f'{row[k]:.2f}' for k in ['iou_pct','dice_pct','precision_pct','recall_pct'])+r'\\')
lines.extend([r'\hline',r'\end{tabular}'])
(TABLE/'complete_class_metrics.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')
lines=[r'\begin{tabular}{lrrrrrrr}',r'\hline',r'Correction & IoU (\%) & Dice (\%) & Precision (\%) & Recall (\%) & TP & FP & FN\\\hline']
for label,title in [('fused','Before'),('final','After')]:
    row=stats['correction'][label]
    lines.append(title+' & '+' & '.join(f'{row[k]:.2f}' for k in ['iou_pct','dice_pct','precision_pct','recall_pct'])+' & '+' & '.join(f'{row[k]:,}' for k in ['tp','fp','fn'])+r'\\')
lines.extend([r'\hline',r'\end{tabular}'])
(TABLE/'ma_correction_metrics.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')

# The original unquantized probabilities are absent; use exact existing embedded raster panels.
FD=ROOT/'HRDecoder_FinalDecision_20261005';old=next(FD.glob('*/figures/F11_predeclared_validation_cases.svg')).parent.parent
source=old/'figures/F11_predeclared_validation_cases.svg'
delivery=read(old/'report_delivery_manifest.json')
record=next(x for x in delivery['files'] if x['path']=='figures/'+source.name)
assert sha(source)==record['sha256']
planpath=FD/'execution/reports/publication_server_evidence_plan.json';plan=read(planpath)
derpath=FD/'execution/server_results/reports/publication_server_evidence/derivation.json';der=read(derpath)
rawpath=FD/'execution/server_results/runs/S_seed42/val_u20000/metrics.json';raw=read(rawpath)
assert sha(rawpath)==plan['metric_sha256']
assert raw['update']==20000 and raw['split']=='val' and raw['n_images']==76 and raw['strict_greater']
assert der['plan_sha256']==sha(planpath)
arrays=[];panel_records=[];cases=plan['cases'][3:]
nodes=ET.parse(source).getroot().findall('.//{http://www.w3.org/2000/svg}image')
assert len(nodes)==10
pd=F/'qualitative_panels';pd.mkdir(exist_ok=True)
for i,node in enumerate(nodes):
    payload=base64.b64decode(node.attrib['{http://www.w3.org/1999/xlink}href'].split(',',1)[1])
    image=Image.open(io.BytesIO(payload)).convert('RGB');a=np.asarray(image)
    name=['rgb','gt','partner','spatial','complete'][i%5];case=cases[i//5]
    filename=f'{case["sid"]}_{name}.png';target=pd/filename;target.write_bytes(payload)
    assert pixels(Image.open(target).convert('RGB'))==pixels(a)
    # Matplotlib's SVG stores image rows inverted and displays them with scale(1,-1).
    # Reproduce that recorded transform; do not show the encoded rows upside down.
    assert node.attrib['transform'].startswith('scale(1 -1)')
    display=np.flipud(a).copy();arrays.append(display)
    panel_records.append({'sid':case['sid'],'panel':name,'svg_image_index':i,'local_panel':target.relative_to(OUT).as_posix(),'embedded_png_sha256':hashlib.sha256(payload).hexdigest(),'encoded_rgb_pixel_sha256':pixels(a),'display_rgb_pixel_sha256':pixels(display),'width':a.shape[1],'height':a.shape[0],'pixel_change':'Exact embedded PNG bytes retained; display reproduces SVG scale(1,-1) by lossless row reversal; no sample values changed'})
case_audit=[]
for case in cases:
    sid=case['sid'];rgbpath=Path('D:/HRDecoder/data/DDR_split_613/images/val')/(sid+'.jpg');gtpath=Path('D:/HRDecoder/data/DDR_split_613/labels/val')/(sid+'.png')
    assert sha(gtpath)==case['label_sha256']
    qsource=next(z for z in raw['probability_files'] if z['sid']==sid)
    psource=next(z for z in der['files'] if z['sid']==sid)
    qpath=rawpath.parent/(sid+'.npz');ppath=FD/'execution/server_results'/psource['path']
    case_audit.append({'sid':sid,'split':'validation','roi_xyxy_original_pixels':case['crop_xyxy'],'selection_rule':case['rule'],'selection_is_illustrative_not_random':True,'rgb_source':str(rgbpath),'rgb_sha256_current':sha(rgbpath),'ground_truth_source':str(gtpath),'ground_truth_sha256_matches_plan':True,'spatial_model':'S_seed42 EMA @20000; recorded selected endpoint','spatial_full_probability_record':qsource,'partner_crop_record':psource,'spatial_npz_currently_available':qpath.is_file(),'partner_npz_currently_available':ppath.is_file(),'complete_formula':'.75*P0+.25*S_seed42_EMA_u20000','threshold':'strict > .5','raw_probabilities_not_loaded_in_this_revision':True})
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'pdf.fonttype':42,'svg.fonttype':'none','svg.hashsalt':'hrdecoder-qualitative-v2'})
fig=plt.figure(figsize=(7.25,3.72),facecolor='white')
left,right=.012,.992;gap=.008;panelw=(right-left-4*gap)/5
titles=['Image','Ground truth','Fixed partner','Spatial branch','Complete system']
for r in range(2):
    y=.505 if r==0 else .075
    for c in range(5):
        ax=fig.add_axes([left+c*(panelw+gap),y,panelw,.369])
        ax.imshow(arrays[r*5+c],interpolation='nearest');ax.axis('off')
        if r==0:ax.set_title(titles[c],fontsize=8,pad=4)
    case=cases[r];yy=.477 if r==0 else .047
    fig.text(.015,yy,f'{chr(65+r)}  {case["sid"]}   |   ROI {tuple(case["crop_xyxy"])}',fontsize=6.5,va='center')
fig.text(.5,.012,'EX orange   |   HE green   |   SE yellow   |   MA blue',fontsize=7,ha='center')
for ext in ['pdf','png','svg']:
    options={'metadata':{'CreationDate':None,'ModDate':None}} if ext=='pdf' else {'metadata':{'Date':None}} if ext=='svg' else {}
    fig.savefig(F/('qualitative_validation.'+ext),dpi=300,facecolor='white',**options)
plt.close(fig)
qa={'source_svg':str(source),'source_svg_sha256':sha(source),'source_matches_20261005_delivery_manifest':True,'source_generation_script':str(old/'analysis/build_scientific_figures.py'),'source_script_sha256':sha(old/'analysis/build_scientific_figures.py'),'plan_source':str(planpath),'plan_sha256':sha(planpath),'derivation_source':str(derpath),'derivation_sha256':sha(derpath),'current_validation_metrics_sha256_matches_plan':True,'new_inference_or_training':False,'new_model_or_case_selection':False,'source_raster_resolution':[206,206],'rendering_policy':'Lossless extraction of original SVG embedded PNG panels and exact reproduction of its scale(1,-1) display transform; new whitespace/labels only. Nearest-neighbor presentation; no recoloring, redrawing or AI modification. Native 768x768 ROI was downsampled in archived source figure; original prediction pixels cannot be reconstructed from this panel.','cases':case_audit,'panels':panel_records,'outputs':{ext:{'path':'figures/qualitative_validation.'+ext,'sha256':sha(F/('qualitative_validation.'+ext))} for ext in ['png','pdf','svg']}}
save(E/'revision2_qualitative_provenance.json',qa)
crop_source=old/'evidence/qualitative_crop_counts.json'
shutil.copyfile(crop_source,E/'revision2_qualitative_counts_source.json')
crop_counts=read(crop_source)
stats['source_inputs']['qualitative_crops']='evidence/revision2_qualitative_counts_source.json'
stats['qualitative_crops']=[]
for case in cases:
    r=next(x for x in crop_counts if x['sid']==case['sid'])
    a={'sid':r['sid'],'roi_xyxy':r['crop_xyxy'],'source_note':'archived native-ROI count derivation, not recalculated from downsampled figure panels','MA':{}}
    for system,counts in r['crop_counts'].items():
        a['MA'][system]=metrics(dict(zip(['tp','fp','fn'],counts[3])))
    a['MA']['final_minus_partner']={k:a['MA']['locked'][k]-a['MA']['P0'][k] for k in a['MA']['P0']}
    stats['qualitative_crops'].append(a)
save(E/'revision2_result_statistics.json',stats)
registry=read(OUT/'numeric_evidence.json');registry['entries']=[r for r in registry['entries'] if r['source']!='evidence/revision2_result_statistics.json']
def register(a,path=''):
    if isinstance(a,dict):
        for k,x in a.items():register(x,path+'/'+k)
    elif isinstance(a,list):
        for i,x in enumerate(a):register(x,path+'/'+str(i))
    elif isinstance(a,(float,int)) and not isinstance(a,bool):
        registry['entries'].append({'value':float(a),'source':'evidence/revision2_result_statistics.json','selector':{'kind':'json-pointer','pointer':path},'representations':['raw'],'tolerance':.0051/max(abs(a),1),'note':'Offline descriptive recomputation from saved integer counts; table display supports two decimal places. No model execution or selection.'})
register(stats);save(OUT/'numeric_evidence.json',registry)
fs=read(F/'figure_sources.json')
for p in [source,planpath,derpath,E/'revision2_qualitative_provenance.json',Path(__file__),F/'make_qualitative_validation.py']:
    key=str(p.relative_to(OUT)) if OUT in p.parents else str(p)
    fs[key]={'sha256':sha(p),'bytes':p.stat().st_size}
save(F/'figure_sources.json',fs)
print(json.dumps({'count_delta':stats['correction']['net_change'],'n_registry_entries':len(registry['entries']),'source_figure_sha_verified':True,'real_cases':[x['sid'] for x in cases],'raw_npz_available':[x['spatial_npz_currently_available'] for x in case_audit]},ensure_ascii=False))
