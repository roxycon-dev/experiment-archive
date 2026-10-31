"""Label-aware table audit, complementing the skill's broad numeric scanner."""
from pathlib import Path
import json,re,hashlib,math
OUT=Path(__file__).resolve().parents[1];P=OUT/'paper';E=OUT/'evidence'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
t=read(E/'locked_test_source.json');h=read(E/'ma_correction_source.json');v=read(E/'validation_source.json');b=read(E/'historical_baseline_source.json')
checks=[]
def check(name,observed,expected,tolerance=5.1e-5,note=''):
    ok=(math.isclose(observed,expected,rel_tol=0,abs_tol=tolerance))
    checks.append({'check':name,'observed':observed,'expected':expected,'absolute_tolerance':tolerance,'pass':ok,'note':note})
def table(path,label):
    s=path.read_text(encoding='utf-8');pos=s.index('\\label{'+label+'}');end=s.index('\\end{table',pos)
    text=s[pos:end];body=re.search(r'\\begin\{tabular\}\{[^\n]+\}\s*(.*?)\\end\{tabular\}',text,re.S).group(1)
    rows={}
    for chunk in re.split(r'\\\\',body):
        cells=[re.sub(r'\\hline','',x).strip() for x in chunk.split('&')]
        if len(cells)>1:rows[cells[0]]=cells[1:]
    return rows
def number(s):return float(re.search(r'[-+]?[\d,]+(?:\.\d+)?',s).group().replace(',',''))
r=table(P/'sections/results.tex','tab:main')
for c in ['EX','HE','SE','MA']:
    expected=[b['historical_test']['per_class_iou_pct_reported'][c]]+[t['test_per_class'][c][x] for x in ['iou_pct','dice_pct','precision_pct','recall_pct']]
    for i,e in enumerate(expected):check('main '+c+' column '+str(i),number(r[c][i]),e,5.1e-5 if i else .0051)
for i,e in enumerate([42.2,t['locked_test_miou_pct']]):check('main mIoU '+str(i),number(r['mIoU'][i]),e)
r=table(P/'sections/results.tex','tab:corrector')
for label,key in [('Before','fused'),('After','final')]:
    d=h[key]['counts']['MA'];tp,fp,fn=[d[x] for x in ['TP','FP','FN']]
    expected=[100*tp/(tp+fp+fn),100*tp/(tp+fp),100*tp/(tp+fn),100*h[key]['mIoU']]
    for i,e in enumerate(expected):check('MA corrector '+label+' column '+str(i),number(r[label][i]),e)
r=table(P/'sections/results.tex','tab:spatial')
for label,arm in [('Continued training','C'),('Spatial adaptation','S')]:
    values=[]
    for update in [20000,32000]:
        record=next(x for x in v['records'] if x['arm']==arm and x['update']==update)
        values.extend([record['outputs'][x]['miou_pct'] for x in ['single','fixed_fusion']])
    for i,e in enumerate(values):check('spatial '+label+' column '+str(i),number(r[label][i]),e)
r=table(P/'training_table.tex','tab:training')
expected_training=[('Image-detail expert',2,[.005],40000,36000),('Refined detail reference',2,[.0005],8000,3000),('Local student',2,[.0005],20000,2000),('Semantic MA residual',2,[.01],6000,1500),('MA post-correction',1,[.001],2000,1600),('ConvNeXt-S / U-Net',8,[.0001,.0003],12000,12000),('DINO partial fine-tuning',8,[.000005,.0003],24000,16000),('DINO full fine-tuning',8,[.00001,.00005],12000,12000),('Spatial DINO adaptation',8,[.00001,.00005,.0001],32000,20000)]
for label,batch,lrs,horizon,retained in expected_training:
    row=r[label]
    for col,expected in [(1,batch),(3,horizon),(4,retained)]:check('training '+label+' col'+str(col),number(row[col]),expected,0,'ConvNeXt batch reflects archived configuration, not recovered immutable original config identity' if 'ConvNeXt' in label else '')
    clean=row[2].replace(r'\!','');observed=[]
    for m in re.finditer(r'(?:(\d+)\\times)?10\^\{(-?\d+)\}',clean):observed.append(float(m.group(1) or 1)*10**int(m.group(2)))
    assert len(observed)==len(lrs),(label,observed,lrs)
    for i,(obs,expected) in enumerate(zip(observed,lrs)):check('training '+label+' LR'+str(i),obs,expected,1e-14)
save={'audit_type':'label-aware numeric table verification plus completed human semantic review','scope':'main classwise table, MA correction table, matched spatial table, retained training table; abstract/result prose, CI, qualitative identity, and new historical configuration paragraph manually reviewed','source_records':['evidence/locked_test_source.json','evidence/ma_correction_source.json','evidence/validation_source.json','evidence/historical_baseline_source.json','evidence/revision2_training_numeric_sources.json','evidence/revision2_qualitative_provenance.json'],'reviewed_tex_sha256':{str(p.relative_to(OUT)):sha(p) for p in sorted(P.rglob('*.tex'))},'checks':checks,'all_pass':all(x['pass'] for x in checks),'no_training_or_model_execution':True,'known_boundaries':['Rounded historical baseline; no reconstructed baseline precision/recall/Dice.','Selected20k is inside32k schedule; one seed.','ConvNeXt micro4/effective8 is an archived configuration declaration; checkpoint confirms2 microsteps per update, not standalone actual microbatch value.','No statistical uncertainty or lesion sensitivity inferred from selected crops.','Saved correction head seed unconfirmed.']}
save['source_input_sha256']={path:sha(OUT/path) for path in save['source_records']+['numeric_evidence.json','evidence/revision2_result_statistics.json','evidence/frozen_partner_source.json']}
save['audit_script_sha256']=sha(Path(__file__))
(E/'final_numeric_manual_audit.json').write_text(json.dumps(save,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'label_aware_checks':len(checks),'all_pass':save['all_pass']}))
if not save['all_pass']:raise SystemExit(1)
