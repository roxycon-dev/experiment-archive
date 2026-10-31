"""Author offline report from saved facts. No SSH, model inference or selection."""
from pathlib import Path
from datetime import datetime, timezone
import json, csv, re, hashlib, sys
from docx import Document
from docx.shared import Inches, Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

OUT=Path(__file__).resolve().parents[1]; EX=OUT.parent/'execution'; ROOT=OUT.parent.parent
E=OUT/'evidence'; A=OUT/'analysis'; SR=EX/'server_results'
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2),'utf8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def f(v):return f'{v:.4f}'
v=read(E/'four_arm_saved_validation.json');t=read(E/'final_locked_test_evidence.json'); b=read(E/'original_HRDecoder_baseline_evidence.json')
def row(a,u):return next(x for x in v['records'] if x['arm']==a and x['update']==u)
def s(a,u,fo='fixed_fusion'):return row(a,u)['outputs'][fo]['miou_pct']

# Fresh small-document source index; older curated archive is kept verbatim.
old=read(ROOT/'_全实验档案_20260920/02_HRDecoder逐实验事实档案.json')
archive_paths={}
for item in old['records']:
    paths=[]
    for reference in item.get('evidence','').split(';'):
        match=re.match(r'\s*(.+?\.(?:md|json|csv|yaml|py))(?::|\s*$)',reference)
        if match:paths.append((ROOT/match[1]).resolve())
    archive_paths[item['id']]=paths
known=[ROOT/'_全实验档案_20260920/02_HRDecoder逐实验事实档案.json',
 ROOT/'autodl100/最终实验结论.md',ROOT/'autodl100/results/trial_b/hr100_b.py',
 ROOT/'hrdecoder_plus3/repo/configs/lesion/exp/hrplus_patch_dice.py',
 ROOT/'HRDecoder_审查轮_20260916/audit_runs/audit_20260916_A/experiments/h7_test.json',
 ROOT/'HRDecoder_FusionAware_20260917/HRDecoder-DDR-阶段成果报告-20260917至0918.md',
 ROOT/'HRDecoder_FusionAware_20260917/ROUND5_RESULT_20260918.md',
 ROOT/'HRDecoder_FusionAware_20260917/ROUND6_RESULT_20260918.md',
 ROOT/'HRDecoder_FusionAware_20260917/execution_20260919/remote_recovery_20260920/runs/cs4000_v2/C_summary.json',
 ROOT/'HRDecoder_FusionAware_20260917/execution_20260919/remote_recovery_20260920/runs/cs4000_v2/S_summary.json',
 ROOT/'HRDecoder_FusionAware_20260917/execution_20260919/remote_recovery_20260920/runs/evalcand_test/evalcand_summary.json',
 ROOT/'HRDecoder_FullCompetition_20260920/runs/completion_summary.json',
 ROOT/'HRDecoder_Ready_20261002/legacy/runtime.py',
 ROOT/'HRDecoder_Ready_20261002/round2_highres/server_results/reports/round_result.json',
 ROOT/'HRDecoder_Q_20261003/reports/original_plan_experiment_result.json',
 ROOT/'HRDecoder_Q_20261003/server_results/reports/final_test/summary.json',
 ROOT/'HRDecoder_Q_20261003/server_results/continuation_evidence/reports/continuation_final_test/summary.json',
 ROOT/'HRDecoder_Next_20261003/reports/HRDecoder_严格实验最终报告_20261003.md',
 ROOT/'HRDecoder_NextPlan_20261004/execution/server_results/reports/最终实验报告.md',
 OUT.parent/'00_最后一轮决定性实验方案.md',OUT.parent/'01_执行协议.json',OUT.parent/'02_空间交互与随机流接口.md',
 EX/'reports/single_seed_parallel_amendment.json',EX/'reports/publication_report_scope.json',
 SR/'reports/final_selection_lock.json',SR/'reports/final_test.json',SR/'reports/gpu_readiness.json',SR/'reports/data_identity.json',
 SR/'reports/publication_server_evidence/derivation.json',EX/'reports/final_backup_receipt.json',EX/'reports/shutdown_receipt.json',
 EX/'parallel/node2/reports/final_backup_receipt.json',EX/'parallel/node2/reports/shutdown_receipt.json',
 Path('D:/HRDecoder/data/DDR_split_613/split_assignment.json')]
known += [Path(x['path']) for x in b['source_files']]+[Path(x['path']) for x in t['sources']]
known += [Path(x['source_path']) for x in v['records']]
known += [x for x in (SR/'qlib').glob('*.py')]
known += [SR/name for name in ['spatial.py','final_model.py','engine.py','train_run.py','inputs.py','experiment.py','validation.py']]
known += [p for paths in archive_paths.values() for p in paths if p.is_file()]
known += [ROOT/'HRDecoder_诊断调研_20260910/HRDecoder_诊断与押注建议.md',
 ROOT/'HRDecoder_MA残差头与集成_20260915-16/01_结果报告/最终报告_两轮实验汇总.md',
 ROOT/'HRDecoder_重新诊断与三点提升方案_20260926/candidate_audit_summary.json',
 ROOT/'HRDecoder_MA残差头与集成_20260915-16/03_代码/configs/hrplus_mares.py',
 ROOT/'HRDecoder_MA残差头与集成_20260915-16/03_代码/ma_residual.py',
 EX/'reports/TJDR_manifest.json',EX/'reports/TJDR_visual_review.json',EX/'reports/TJDR_duplicate_pairs.json',
 EX/'reports/stream_identity.json',EX/'reports/publication_server_evidence_plan.json',
 EX/'reports/publication_report_server_release.json',EX/'reports/publication_server_evidence_recovery.json']
sources=[];seen=set();missing=[]
for p in known:
    p=p.resolve()
    if p in seen:continue
    seen.add(p)
    if not p.is_file():missing.append(str(p));continue
    sources.append({'id':f'E{len(sources)+1:03d}','path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)})
save(E/'workspace_source_hashes.json',{'generated_at':datetime.now(timezone.utc).isoformat(),'sources':sources,'unresolved_exact_paths':missing})
SOURCEMAP={x['path']:x for x in sources}
def refs(paths):return [SOURCEMAP[str(Path(p).resolve())]['id'] for p in paths if str(Path(p).resolve()) in SOURCEMAP]

old=read(ROOT/'_全实验档案_20260920/02_HRDecoder逐实验事实档案.json')
ledger=[]
for x in old['records']:
    ledger.append({'id':'ARCHIVE_'+x['id'],'period':'through 20260920 historical snapshot','state':x['status'],
      'purpose_change':'historical curated record; exact original fields preserved in archive_record',
      'parent':'see original source/config','datasets':x.get('split',''), 'budget':'see original source',
      'val_miou_original':x.get('val_miou',''),'test_miou_original':x.get('test_miou',''),
      'evidence_grade':'historical curated snapshot; not re-inferred as a new run',
      'source_ids':refs([ROOT/'_全实验档案_20260920/02_HRDecoder逐实验事实档案.json']+archive_paths[x['id']]),
      'source_paths':[str(p) for p in archive_paths[x['id']]],
      'referenced_source_exists':{str(p):p.is_file() for p in archive_paths[x['id']]}, 'archive_record':x})
def add(i,parent,change,data,budget,val,test,paths,state='COMPLETE',infer='',opt='',checkpoint=''):
    ledger.append({'id':i,'period':'primary source reviewed 20261006','state':state,'parent':parent,'purpose_change':change,
       'datasets':data,'budget':budget,'validation_miou_pct':val,'test_miou_pct':test,'inference':infer,'optimizer':opt,
       'checkpoint_identity':checkpoint,'source_ids':refs(paths),'source_paths':[str(Path(p).resolve()) for p in paths],
       'evidence_grade':'saved primary configuration/result; current count audit where available'})
add('B0_original','ImageNet HRNet-W48','original HRDecoder HR learning/FCN','DDR454/76/227','40k;best32k',39.39,42.20,[Path(x['path']) for x in b['source_files']],infer='2048 resize; whole; no flip TTA',opt='SGD .005 momentum .9 wd .0005')
diag=ROOT/'HRDecoder_诊断调研_20260910/HRDecoder_诊断与押注建议.md'
oldresult=ROOT/'HRDecoder_MA残差头与集成_20260915-16/01_结果报告/最终报告_两轮实验汇总.md'
add('E4_detail','ImageNet HRNet-W48 plus new RGB detail branch','stride2/2/1 shallow RGB branch; shared scalar logit addition; original 36k best correction','DDR454/76/227','40k;best36k',38.766,41.626,[diag,oldresult],infer='whole2048; saved historical result',checkpoint='historical archive asset E4 c5a3d0f0b990d2430e545146745bc809538bbe764979baf54f62b450785268e0')
add('B_continuation','E4@36k','no new semantic branch; lower LR, frozen BN, MA auxiliary .02 changed together','DDR454/76/227','8k;best3k',39.205,41.918,[ROOT/'autodl100/results/trial_b/hr100_b.py',ROOT/'autodl100/最终实验结论.md',oldresult],opt='SGD .0005; original poly schedule',infer='whole2048; no flip')
add('E4_B_equal','E4 and B predictions','equal FP32 probability fusion','DDR454/76/227','no new training; saved historical recipe',39.54,42.35,[ROOT/'autodl100/最终实验结论.md'],infer='.5E4+.5B; no flip')
add('D_old_local_student','B strict state in student and frozen reference','1024 local windows; masked Dice only; no KD or hard term; fixed reference retained in inference','DDR454/76/227','20k;best2k',41.615,42.576,[ROOT/'hrdecoder_plus3/repo/configs/lesion/exp/hrplus_patch_dice.py',oldresult],opt='SGD .0005 momentum .9 wd .0005; poly200 warmup; clip5',infer='.5 tiled student + .5 whole frozen B; this is a composite output',checkpoint='student@2k historical asset 5fe18717b62d8fc3ab4846a50c046588e2af05abbbfdb3a958f4794018b319ea')
add('E_old_MA_residual','B','MA-only residual trained on frozen mature B; other class outputs retained','DDR454/76/227','6k;best1500',40.255,42.559,[oldresult,ROOT/'HRDecoder_MA残差头与集成_20260915-16/03_代码/configs/hrplus_mares.py',ROOT/'HRDecoder_MA残差头与集成_20260915-16/03_代码/ma_residual.py'],opt='SGD .01 momentum .9 wd .0005; poly300 warmup; clip5',infer='tiled residual student plus .5 frozen B reference; no flip',checkpoint='historical asset a18ff92b4da24d870554f17856983d4be042386cacbcaba0d4e13f3f248ed5e5')
add('L_Lovasz','B','local student loss adds .1Lovasz','DDR454/76/227','saved historical best',41.056,{'no_TTA':41.777,'flip':41.645},[oldresult],infer='historical composite; not a retained final member')
add('four_member_flip','D_old,E_old,B,E4 outputs','fixed 3:2:1:1 probability mixture with horizontal flip','DDR454/76/227','no new training',None,43.2289242453,[oldresult],infer='D_old already contains frozen B; weights are outer weights')
add('Sept26_candidate_audit','saved champion predictions','predeclared nontraining candidate coverage grid; no new training or test','DDR validation76','analysis only',None,None,[ROOT/'HRDecoder_重新诊断与三点提升方案_20260926/candidate_audit_summary.json'],state='ANALYSIS_COMPLETE_NOT_TRAINING',infer='historical analysis only; no scan rerun during final report')
hp=ROOT/'HRDecoder_审查轮_20260916/audit_runs/audit_20260916_A/experiments/h7_test.json'
add('H7_H4_composite','D_old/E_old/B/E4 outputs','3:2:1:1 horizontal flip + H4 MA only','DDR','H4 head best1600',42.66942845,44.2115035614162,[hp],infer='weighted probabilities then nonlinear MA correction')
cs=ROOT/'HRDecoder_FusionAware_20260917/execution_20260919/remote_recovery_20260920/runs'
for a in ['C','S']:
    h=read(cs/f'cs4000_v2/{a}_summary.json')
    add('historical_sampling_'+a,'historical HRDecoder student','uniform' if a=='C' else 'conditional foreground window sampling','DDR','4k',[(z['step'],100*z['miou']) for z in h['history']],None,[cs/f'cs4000_v2/{a}_summary.json'],infer='new branch validation only')
hc=read(cs/'evalcand_test/evalcand_summary.json')
add('historical_sampling_locked_C2k','historical sampling C','fixed25 with H4 composite','DDR','2k selected',None,100*hc['results']['mix_C_step_2000']['miou'],[cs/'evalcand_test/evalcand_summary.json'],infer='.75 historical champion + .25 C2k')
add('U0_DDR','ImageNet ConvNeXt-S','independent U-Net','DDR','12k',38.44641257662971,None,[ROOT/'HRDecoder_FullCompetition_20260920/runs/completion_summary.json'],infer='historical native whole probability')
add('P0','C_old,U0 outputs','frozen .75/.25 system partner','DDR','zero-training recipe',43.48770322103657,44.78089145110081,[ROOT/'HRDecoder_Ready_20261002/legacy/runtime.py',SR/'reports/data_identity.json'],infer='.75 C_old + .25 U0')
for a,best in [('H1',35.5950),('U1',36.6934),('A0',9.6781)]:
    p=ROOT/f'HRDecoder_Ready_20261002/server_results/work_dirs/{a}_ddr_finetune/completion_summary.json'
    if p.is_file():
        info=read(p)
        add(a+'_Oct2','see completion configuration','external/alternative model exploration; not final parent','DDR and registered external source as applicable','see source',best,None,[p],state=info.get('status','saved terminal state'),infer='validation only')
hr=ROOT/'HRDecoder_Ready_20261002/round2_highres/server_results/reports/round_result.json'
add('U0_continuation2048','U0','paired low-resolution continuation','DDR','2k;best1k',43.3662987,None,[hr],infer='fixed25')
add('U0_highres','U0','high-resolution continuation','DDR','4k;best1k',43.7327634,None,[hr],infer='fixed25; this validation geometry differs from later G')
gn=ROOT/'HRDecoder_Next_20261003/reports/HRDecoder_严格实验最终报告_20261003.md'
add('G','U0_highres','registered G final candidate','DDR','see source',43.8679,44.6345,[gn],infer='fixed25; historical selected final')
for a,z in [('M0',41.0373),('M1',41.0441)]:add(a,'frozen DINO','global adapter' if a=='M0' else 'global plus RGB stride1/2 adapter','DDR','2k actual cost stop',z,None,[gn],state='SCIENTIFIC_COST_STOP_REGISTERED',infer='single; no test')
qp=ROOT/'HRDecoder_Q_20261003/reports/original_plan_experiment_result.json';qt=ROOT/'HRDecoder_Q_20261003/server_results/reports/final_test/summary.json'
add('Q_DDR16_R42','DINOv3 LVD','head then last4 DPT/RGB','DDR','16k',44.355692338538375,45.72155710642771,[qp,qt],infer='.75P0+.25Q16; single val37.8937186468')
add('Q_DDR16_R2024','DINOv3 LVD','archived prior Q route','DDR','16k',44.21734429280863,45.677066093957414,[qp,qt],infer='historical R fixed25; not rerun Oct5')
add('Q_MIX','DINOv3 LVD','4DDR+2IDRiD+2MAPLES first12k then DDR8','DDR, IDRiD, MAPLES','24k paired historical',None,None,[qp],infer='no independent newly selected final test; real node values in original result')
prev=ROOT/'HRDecoder_NextPlan_20261004/execution/server_results/reports/最终实验报告.md'
for a,sv,fv in [('E_NEG',37.009446,44.164305),('F4',38.741114,44.306921),('F12',40.241833600516024,44.58879155)]:
    add(a+'_42','paired historical Q_MIX initialization' if a=='E_NEG' else 'Q_DDR42@16k',
        'correct full negative semantics' if a=='E_NEG' else ('last4 continuation' if a=='F4' else 'all12 continuation'),
        'DDR/IDRiD/MAPLES' if a=='E_NEG' else 'DDR','24k' if a=='E_NEG' else '12k',{'single':sv,'fixed25':fv},None,[prev],infer='fixed original formulas; no new test')
for a in ['C','S','D','SD']:
    paths=[Path(row(a,u)['source_path']) for u in [20000,32000]]
    values={str(u):{fo:row(a,u)['outputs'][fo]['miou_pct'] for fo in ['single','fixed_fusion','half_fusion']} for u in [20000,32000]}
    add('FINAL_'+a+'_42','F12_42 raw strict shared','spatial='+str(a in ['S','SD'])+';TJDR='+str(a in ['D','SD']),
      'DDR + TJDR439TRAIN' if a in ['D','SD'] else 'DDR','32k actual;20/32k retained',values,None,paths+[OUT.parent/'01_执行协议.json',EX/'reports/single_seed_parallel_amendment.json'],opt='fresh AdamW;EMA .999',infer='EMA only;single/fixed25/half')
add('FINAL_LOCKED_S20_fixed25','S20EMA + P0 outputs','validation selected one frozen recipe','DDR','one locked test',45.32183122738841,t['locked_test_miou_pct'],[Path(x['path']) for x in t['sources']],infer='.75P0+.25S_EMA',checkpoint=t['locked_recipe']['checkpoints']['42']['sha256'])
details={
 'B0_original':{'initialization_freezing':'ImageNet HRNet-W48; BN norm_eval=False','loss':'BinaryLoss Dice; HR auxiliary .1','EMA':'not configured','learning_rate':'SGD peak .005; poly power1;1500 linear warmup'},
 'E4_detail':{'initialization_freezing':'ImageNet HRNet-W48 with RGB detail path; shared residual output initialized zero','loss':'historical Dice HRDecoder objective; see original record','EMA':'not reported'},
 'B_continuation':{'initialization_freezing':'strict E4@36k; BN frozen; no new semantic branch','loss':'Dice plus MA hard auxiliary .02','EMA':'not configured'},
 'D_old_local_student':{'initialization_freezing':'student and frozen reference strict B; reference eval; student BN frozen','loss':'masked Dice;hard_weight=0;lovasz=0','EMA':'not configured'},
 'E_old_MA_residual':{'initialization_freezing':'B strict; only new semantic MA residual parameters train; reference frozen','loss':'masked Dice;ma_hard_weight=0;lovasz=0','EMA':'not configured'},
 'L_Lovasz':{'loss':'Dice+.1Lovasz; inherited student/reference architecture'},
 'H7_H4_composite':{'initialization_freezing':'fixed historical members; train small postfusion MA head only','loss':'MA BCE/Dice .5/.5','learning_rate':'Adam .001','EMA':'not configured'},
 'Q_DDR16_R42':{'initialization_freezing':'DINOv3 generic pretraining;2k head only then last4; P0 frozen','loss':'original Q masked BCE/Dice .5/.5'},
 'F12_42':{'initialization_freezing':'Q_DDR42@16k raw;all12 blocks continue;P0 frozen','loss':'original Q masked BCE/Dice .5/.5','EMA':'not used for historical selection'}
}
for item in ledger:
    item.setdefault('initialization_freezing',item.get('parent','not extracted in historical snapshot'))
    item.setdefault('loss','not extracted here; consult cited actual configuration; not inferred from title')
    item.setdefault('learning_rate',item.get('optimizer','not extracted in historical snapshot'))
    item.setdefault('EMA','not extracted in historical snapshot; no assumed value')
    item.update(details.get(item['id'],{}))
    if item['id'].startswith('FINAL_') and item['id']!='FINAL_LOCKED_S20_fixed25':
        item.update({'initialization_freezing':'strict shared F12 raw;fresh AdamW;all relevant gradient-bearing blocks train',
          'loss':'masked .5BCE+.5Dice;DDRmean+.25TJDRmean only D/SD first20k',
          'learning_rate':'block1e-5*.8^(11-b),decoder5e-5,newspatial1e-4;500warmup then cosine to1%',
          'EMA':'full model .999;selected EMA only'})
save(E/'historical_experiment_ledger.json',{'records':ledger,'archive_snapshot_preserved':True,'new_runs_not_inferred_from_plans':True})
with (E/'historical_experiment_ledger.csv').open('w',encoding='utf-8-sig',newline='') as fp:
    fields=['id','period','state','parent','purpose_change','datasets','budget','initialization_freezing','loss','optimizer','learning_rate','EMA','validation_miou_pct','test_miou_pct','inference','checkpoint_identity','source_ids','source_paths','evidence_grade']
    w=csv.DictWriter(fp,fieldnames=fields,extrasaction='ignore');w.writeheader()
    for x in ledger:w.writerow({k:json.dumps(z,ensure_ascii=False) if isinstance(z,(list,dict)) else z for k,z in x.items()})

tables={}
def table(k,title,cols,rows,note,source):tables[k]={'title':title,'columns':cols,'rows':rows,'note':note,'sources':source}
table('T1','表1 数据与使用边界',['来源','实际规模','用途','身份及限制'],[
 ['DDR TRAIN','454图','全部四臂','固定新划分；原生图像'],['DDR VAL','76图','曲线与注册选型','全部节点同一SID集合'],['DDR TEST','227图','锁定系统测试','历史使用已记录'],['TJDR TRAIN','448对；保留439','D/SD前20k','896文件SHA；排除9；23叠加图'],['TJDR TEST','未下载未使用','不参与','官方113图隔离'],['IDRiD/MAPLES','历史Q_MIX清单','历史额外监督探索','未进入S父链']],
 '标签通道为EX/HE/SE/MA；TJDR原ID需映射。图像去重不等于患者级独立性。', ['data_identity','TJDR acceptance','Q original result'])
table('T2','表2 最终训练与接口设置',['设置','固定实现','解释'],[
 ['父模型','F12 raw@12k','严格同状态；新优化器'],['每臂更新','32000','4k验证；20k/32k部署端点'],['DDR窗口','micro2 × accum4','4前景+4均匀'],['TJDR窗口','micro2 × accum2','仅D/SD前20k'],['窗口几何','1024 / center768 / halo128','原生像素；有效区域'],['损失','DDRmean + 0.25TJDRmean','每micro .5BCE+.5Dice'],['AdamW','β=.9/.999;ε=1e−8;wd=.01','bias/norm无decay；clip1'],['学习率','块1e−5层衰减.8；旧头5e−5；新空间1e−4','500warmup；余弦至1%'],['EMA','全模型.999','每成功step更新；仅EMA选型'],['空间路径','strides4/8/16/32;64/128/256/256','GN/GELU；12头4点'],['交互时点','inject before0/3/6/9;extract after2/5/8/11','官方RoPE/CLS/register保留']],
 'D/SD的独立source头不用于DDR推理；来源退出不重置学习率。', ['frozen protocol','spatial.py','final_model.py'])
table('T3','表3 主要历史变化与去向',['阶段','真实改动','验证 %','测试 %','最终作用'],[
 ['B0','原HRDecoder复现','39.390','42.20','主要基线'],['E4','RGB细节支路','38.766','41.626','保留在C_old'],['B','低LR/冻结BN/MA辅助','39.205','41.918','后续父模型及成员'],['E4+B','固定半半','39.54','42.35','互补探索'],['D_old','局部学生+冻结B','41.615','42.576','保留成员'],['E_old','B上的MA残差','40.255','42.559','保留成员'],['四成员flip','3:2:1:1加权TTA','见原计数',f(43.2289242453),'H4输入'],['H4终配方','MA后融合修正','42.6694',f(44.2115035614),'保留C_old'],['U0/P0','独立ConvNeXt输出加入','43.4877',f(44.7808914511),'固定伙伴'],['G','U0高分辨率延续','43.8679','44.6345','未采纳'],['Q16/R','DINO分支固定25%','44.3557',f(t['R_test_miou_pct']),'强参考'],['F12','全部12块追加12k','44.5888','未新增','本轮父模型'],['S20固定25%','空间交互+既定日程','45.3218',f(t['locked_test_miou_pct']),'最终锁定']],
 '跨阶段为描述性历史，不是等预算消融。验证列在U0/P0及之后显示固定25%系统；早期按原阶段推理定义。', ['historical ledger','raw configurations','locked count audit'])
table('T4','表4 四臂注册端点全部输出',['臂','节点','单分支 %','固定25% %','固定50% %'],[
 [a,str(u//1000)+'k',f(s(a,u,'single')),f(s(a,u)),f(s(a,u,'half_fusion'))] for a in ['C','S','D','SD'] for u in [20000,32000]],
 '所有记录为全76图EMA结果。固定25%=.75P0+.25X；固定50%=.50P0+.50X。测试仅最终锁定配方。',['four_arm_saved_validation.json'])
contr=v['paired_registered_endpoint_contrasts']
table('T5','表5 匹配差值与交互 百分点',['节点','输出','S−C','D−C','SD−C','SD−S−D+C'],[
 [str(u//1000)+'k','单分支' if fo=='single' else '固定25%',*[f(next(x for x in contr if x['update']==u and x['formula']==fo)['contrasts'][key]) for key in ['S_minus_C_pp','D_minus_C_pp','SD_minus_C_pp','SD_minus_S_minus_D_plus_C_pp']]] for u in [20000,32000] for fo in ['single','fixed_fusion']],
 '只在同公式同节点比较；监督因子包含额外数据、头及计算，交互是描述性数值。',['four_arm_saved_validation.json contrasts'])
table('T6','表6 最佳成果与同划分参考',['对象','验证 mIoU %','测试 mIoU %','最终测试差值 pp'],[
 ['原HRDecoder B0','39.3900','42.20','约 '+f(t['approx_gain_vs_baseline_pp'])],['C_old重导出','42.6694','44.2110','约 '+f(t['locked_test_miou_pct']-44.21100725782683)],['冻结P0','43.4877','44.7809',f(t['locked_test_miou_pct']-44.78089145110081)],['上一最佳R','44.3557',f(t['R_test_miou_pct']),f(t['gain_vs_R_pp'])],['最终S20固定25%','45.3218',f(t['locked_test_miou_pct']),'0.0000'],['既定目标','不适用','47.2115','尚差 '+f(t['gap_to_target_pp'])]],
 'B0按历史报告两位小数；增幅约8.6739%。S16k45.5310为诊断验证，不参与该表测试或部署选择。',['original baseline audit','final locked test audit'])
table('T7','表7 锁定测试类别质量与增益',['类','IoU %','Dice %','Precision %','Recall %','ΔB0 pp','ΔR pp'],[
 [c,f(t['test_per_class'][c]['iou_pct']),f(t['test_per_class'][c]['dice_pct']),f(t['test_per_class'][c]['precision_pct']),f(t['test_per_class'][c]['recall_pct']),f(t['test_per_class'][c]['iou_pct']-bb),f(t['gain_vs_R_per_class_pp'][c])] for c,bb in zip(['EX','HE','SE','MA'],[51.56,43.13,49.98,24.11])],
 'ΔB0为相对历史舍入类别值的近似差；ΔR从原始整数计数得出。没有四类全面超过R。',['final_locked_test_evidence.json','baseline original report'])
table('T8','表8 原图测试整数计数',['类','TP','FP','FN','IoU %'],[
 [c,*[f"{t['test_per_class'][c][k]:,}" for k in ['tp','fp','fn']],f(t['test_per_class'][c]['iou_pct'])] for c in ['EX','HE','SE','MA']],
 '先累积全部227图整数，再计算类别IoU和宏平均。图7的域漏检是另一种单分支诊断。',['LOCKED_TEST_seed42/metrics.json'])
plan=read(EX/'reports/publication_server_evidence_plan.json')
table('T9','表9 验证病例预定规则与原图坐标',['SID','规则','类','ROI x0 y0 x1 y1'],[
 [x['sid'],{'largest_positive_image_miou_difference_vs_saved_R':'整图正差最大','largest_negative_image_miou_difference_vs_saved_R':'整图负差最大','nearest_median_image_difference':'接近整图中位变化','largest_MA_false_negative_pixel_reduction':'MA漏像素减少最多','largest_MA_false_negative_pixel_increase':'MA漏像素增加最多'}[x['rule']],x['crop_lesion_class'],' '.join(str(i) for i in x['crop_xyxy'])] for x in plan['cases']],
 '选择在叠加图生成前固定；ROI由GT病灶确定，每行同坐标。病例用有效类别诊断选择，不与图8四类零并集约定混用。',['publication_server_evidence_plan.json','derivation.json'])
bench=read(SR/'reports/gpu_readiness.json')['benchmarks']
table('T10','表10 四路径完整验收计时',['路径','秒每更新','数据等待 ms','全76图验证 s','峰值GPU GiB'],[
 [a,f(bench[a]['seconds_per_update']),f(1000*bench[a]['data_wait_seconds_per_update']),f(bench[a]['full_val_seconds']),f(bench[a]['peak_gpu_bytes']/1024**3)] for a in ['C','S','D','SD']],
 '每路径100次完整更新；非整轮恒定速度，非最终多成员系统在线延迟。',['gpu_readiness.json benchmarks'])
table('T11','表11 论文主张与证据边界',['主张','主要证据','可支持范围'],[
 ['累计系统约+3.6604pp','B0原报告；锁定测试整数','同新DDR划分系统比较'],['空间分支改善','S−C20k/32k','当前设置验证效应'],['系统增量较小','锁定测试+0.1388pp；条件CI','未声明显著突破'],['监督包未胜出','D/SD注册终点','不是所有外部监督无效'],['MA仍为瓶颈','类别计数/域记录/案例','像素质量；非临床敏感性'],['可复算可恢复','配方SHA/端点/全备份','有身份的实际保存成果']],
 '没有独立测试的分支不填测试成绩；历史跨阶段差异不做单因素因果归因。',['tables T1–T10','source ledger'])
save(E/'report_tables.json',tables)
for key,tab in tables.items():
    with (E/(key+'_source.csv')).open('w',encoding='utf-8-sig',newline='') as fp:
        w=csv.writer(fp);w.writerow(tab['columns']);w.writerows(tab['rows'])

if '--evidence-only' in sys.argv:
    print('Evidence index and ledger refreshed offline; report documents untouched:',len(ledger),'records;',len(sources),'sources')
    sys.exit(0)

# The complete authored template is retained for deterministic rebuilding.
template=A/'report_source.md'
if not template.exists():template.write_text((OUT/'HRDecoder项目结题报告.md').read_text('utf8'),'utf8')
content=template.read_text('utf8')
def mdtable(m):
    tab=tables[m.group(1)];lines=['**'+tab['title']+'**','', '| '+' | '.join(tab['columns'])+' |','| '+' | '.join(['---']*len(tab['columns']))+' |']
    lines += ['| '+' | '.join(str(z).replace('|','/') for z in rr)+' |' for rr in tab['rows']]
    lines += ['',tab['note']+' 证据：'+', '.join(tab['sources'])+'。']
    return '\n'.join(lines)
expanded=re.sub(r'\[\[TABLE:(T\d+)\]\]',mdtable,content)
expanded=re.sub(r'\[\[FIG:([^|]+)\|([^\]]+)\]\]',lambda m:'!['+m[2]+'](figures/'+m[1]+'.png)\n\n'+m[2],expanded)
(OUT/'HRDecoder项目结题报告.md').write_text(expanded,'utf8')

doc=Document();sec=doc.sections[0]
sec.page_width=Cm(21);sec.page_height=Cm(29.7);sec.top_margin=Cm(2.2);sec.bottom_margin=Cm(2.1);sec.left_margin=Cm(2.1);sec.right_margin=Cm(2.1)
sec.header_distance=Cm(.9);sec.footer_distance=Cm(.9)
for st in ['Normal','Title','Subtitle','Heading 1','Heading 2','Heading 3','Caption']:
    style=doc.styles[st];style.font.name='Times New Roman';style.font.color.rgb=RGBColor(0,0,0)
    style._element.get_or_add_rPr().get_or_add_rFonts().set(qn('w:eastAsia'),'宋体')
normal=doc.styles['Normal'];normal.font.size=Pt(11);normal.paragraph_format.line_spacing=1.25;normal.paragraph_format.space_after=Pt(6)
normal.paragraph_format.widow_control=True
for name,size in [('Title',23),('Subtitle',15),('Heading 1',16),('Heading 2',13),('Heading 3',11.5)]:
    st=doc.styles[name];st.font.size=Pt(size);st.font.bold=True;st.paragraph_format.keep_with_next=True
    st.paragraph_format.space_before=Pt(12);st.paragraph_format.space_after=Pt(7)
doc.styles['Caption'].font.size=Pt(9);doc.styles['Caption'].paragraph_format.space_after=Pt(9)
for elem in doc.styles['Title']._element.xpath('.//w:pBdr'):elem.getparent().remove(elem)
header=sec.header.paragraphs[0];header.text='HRDecoder 项目结题报告   固定 DDR 划分';header.style='Caption'
footer=sec.footer.paragraphs[0];footer.alignment=WD_ALIGN_PARAGRAPH.CENTER
footer.add_run('第 ');field=OxmlElement('w:fldSimple');field.set(qn('w:instr'),'PAGE');footer._p.append(field);footer.add_run(' 页')
def inline(p,txt):
    # Preserve bold spans, and wrap long hexadecimal identities safely.
    for j,part in enumerate(re.split(r'\*\*(.*?)\*\*',txt)):
        part=re.sub(r'(?<![a-f0-9])[a-f0-9]{64}(?![a-f0-9])',lambda z:'\u200b'.join(z[0][k:k+16] for k in range(0,64,16)),part)
        rr=p.add_run(part);rr.bold=j%2==1
def addtable(tab):
    cp=doc.add_paragraph(tab['title']);cp.style='Caption';cp.paragraph_format.keep_with_next=True
    dt=doc.add_table(rows=1,cols=len(tab['columns']));dt.alignment=WD_TABLE_ALIGNMENT.CENTER;dt.autofit=False
    # Widths are planned per schema, not an equal default for narrative cells.
    weights={'T1':[1.2,1.4,1.6,2.6],'T2':[1.2,3.2,2.5],'T3':[1.2,2.4,1.0,1.0,1.3],
      'T4':[.7,.8,1.6,1.6,1.6],'T5':[.8,1.0,1.1,1.1,1.1,1.6],'T6':[2.1,1.5,1.5,1.8],
      'T7':[.45,1,1,1.15,1.1,1,1],'T8':[.6,1.7,1.7,1.7,1.2],'T9':[1.6,2.7,.5,2.1],
      'T10':[.7,1.3,1.6,1.8,1.5],'T11':[1.7,2.5,2.7]}
    key=next(k for k,z in tables.items() if z is tab);ww=weights[key];widths=[6.61*x/sum(ww) for x in ww]
    for i,c in enumerate(dt.columns):c.width=Inches(widths[i])
    for i,c in enumerate(dt.rows[0].cells):c.text=tab['columns'][i]
    repeat=OxmlElement('w:tblHeader');dt.rows[0]._tr.get_or_add_trPr().append(repeat)
    for rr in tab['rows']:
        cells=dt.add_row().cells
        for i,(cell,tx) in enumerate(zip(cells,rr)):cell.text=str(tx);cell.width=Inches(widths[i])
    compact=len(tab['rows'])<=8
    for ri,r in enumerate(dt.rows):
        # Avoid splitting individual data rows, but permit long tables over pages.
        cant=OxmlElement('w:cantSplit');r._tr.get_or_add_trPr().append(cant)
        for i,c in enumerate(r.cells):
            c.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
            pr=c._tc.get_or_add_tcPr();marg=OxmlElement('w:tcMar')
            for side in ['top','bottom','left','right']:
                el=OxmlElement('w:'+side);el.set(qn('w:w'),'65');el.set(qn('w:type'),'dxa');marg.append(el)
            pr.append(marg)
            if ri==0:
                sh=OxmlElement('w:shd');sh.set(qn('w:fill'),'ECEFF1');pr.append(sh)
            for p in c.paragraphs:
                p.paragraph_format.space_after=Pt(2);p.paragraph_format.space_before=Pt(2);p.paragraph_format.line_spacing=1.12
                p.paragraph_format.keep_with_next=compact or ri==0
                for run in p.runs:run.font.size=Pt(9);run.bold=ri==0
    dt.style='Table Grid'
    p=doc.add_paragraph(tab['note']+' 证据：'+', '.join(tab['sources'])+'。');p.style='Caption'

paras=re.split(r'\n\s*\n',content)
for block in paras:
    block=block.strip()
    if not block:continue
    tm=re.fullmatch(r'\[\[TABLE:(T\d+)\]\]',block)
    fm=re.fullmatch(r'\[\[FIG:([^|]+)\|([^\]]+)\]\]',block)
    if tm:addtable(tables[tm[1]]);continue
    if fm:
        p=doc.add_paragraph();p.alignment=WD_ALIGN_PARAGRAPH.CENTER;p.paragraph_format.keep_with_next=True
        p.add_run().add_picture(str(OUT/'figures'/(fm[1]+'.png')),width=Inches(6.60))
        p=doc.add_paragraph(fm[2]);p.style='Caption';continue
    if block.startswith('# '):p=doc.add_paragraph(block[2:],style='Title');continue
    if block.startswith('## '):
        txt=block[3:];p=doc.add_paragraph(txt,style='Heading 1')
        if txt in ['中文结构式摘要','English title and abstract','参考文献'] or txt.startswith('附录') or txt.startswith('1 '):p.paragraph_format.page_break_before=True
        continue
    if block.startswith('### '):doc.add_paragraph(block[4:],style='Heading 2');continue
    p=doc.add_paragraph();p.alignment=WD_ALIGN_PARAGRAPH.LEFT;inline(p,block.replace('\n',' '))
doc.core_properties.title='HRDecoder 眼底病灶分割项目结题报告'
doc.core_properties.subject='固定DDR划分下的模型演化 空间交互与系统增益'
doc.core_properties.author='HRDecoder 项目组';doc.core_properties.keywords='DDR HRDecoder DINOv3 spatial interaction audited results'
doc.save(OUT/'HRDecoder项目结题报告.docx')
body=expanded.split('## 参考文献')[0]
save(E/'authoring_receipt.json',{'generated_at':datetime.now(timezone.utc).isoformat(),
  'chinese_body_characters':len(re.findall('[\u4e00-\u9fff]',body)),'body_characters':len(body),
  'figures':11,'tables':11,'ledger_records':len(ledger),'primary_hashed_sources':len(sources),
  'unresolved_exact_paths':missing,'word_sha256':sha(OUT/'HRDecoder项目结题报告.docx'),
  'report_requires_render_QA':True,'servers_contacted':False,'new_training_or_inference':False})
print('Authored DOCX/Markdown; 11 tables and 11 figures; ledger',len(ledger),'body Chinese chars',len(re.findall('[\u4e00-\u9fff]',body)))
