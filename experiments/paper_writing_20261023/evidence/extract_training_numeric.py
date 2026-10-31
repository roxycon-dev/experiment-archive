"""Extract numeric training evidence without importing any research/model code."""
from pathlib import Path
import json,re,hashlib,math
OUT=Path(__file__).resolve().parents[1];ROOT=OUT.parent;E=OUT/'evidence'
rows=[]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def add(value,path,location,excerpt,role):
    rows.append({'value':float(value),'original_source':str(path),'original_sha256':sha(path),'location':location,'exact_excerpt':excerpt,'role':role})
def fields(rel,pointers,role):
    p=ROOT/rel;a=read(p)
    for pointer in pointers:
        v=a
        for token in pointer.strip('/').split('/'):v=v[int(token)] if isinstance(v,list) else v[token]
        assert isinstance(v,(float,int)) and not isinstance(v,bool)
        add(v,p,pointer,json.dumps(v),role)
def pyfields(rel,role):
    p=ROOT/rel
    pattern=r'\b(lr|max_iters|warmup_iters|warmup_ratio|momentum|weight_decay|max_norm|min_lr|samples_per_gpu|power)\s*=\s*([-+]?(?:\d*\.?\d+)(?:[eE][-+]?\d+)?)'
    for line,s in enumerate(p.read_text(encoding='utf-8-sig').splitlines(),1):
        code=s.split('#',1)[0]
        for m in re.finditer(pattern,code):add(float(m.group(2)),p,f'line {line}, {m.group(1)}',s.strip(),role)
vendor='HRDecoder_Next_20261003/vendor/champion_repo/'
for file,role in [('configs/lesion/exp/baseline_split613.py','inherited initial-detail and historical baseline schedule; E4 inherits this config'),('configs/_base_/schedules/poly10warm.py','initial-detail inherited warmup'),('configs/lesion/exp/hr100_a.py','refined B inherits A numerical settings but B overrides semantic_refinement=False'),('configs/lesion/exp/hrplus_patch_dice.py','retained local student'),('configs/lesion/exp/hrplus_mares.py','retained semantic MA residual expert')]:pyfields(vendor+file,role)
h=E/'ma_correction_source.json';a=read(h)
for i,m in enumerate(a['models']):
    match=re.search(r'iter_(\d+)\.pth',m['ckpt']);assert match
    add(int(match.group(1)),h,f'/models/{i}/ckpt',m['ckpt'],'retained checkpoint update parsed from immutable historical test recipe filename')
h4=next(ROOT.glob('HRDecoder_*20260916/audit_runs/audit_20260916_A/experiments/h4_post_fusion_corrector.json'))
fields(h4.relative_to(ROOT),['/iters','/lr','/best/step'],'actual bounded MA post-correction run; actual run length overrides script default')
fields('HRDecoder_Ready_20261002/reports/U0_historical_checkpoint_metadata.json',['/update','/micro_step','/total_updates','/scheduler_state/base_lrs/backbone','/scheduler_state/base_lrs/decoder','/scheduler_state/total_updates','/scheduler_state/warmup_updates','/scheduler_state/cosine_min_ratio'],'historical ConvNeXt checkpoint metadata; RNG deliberately omitted; micro_step is not microbatch size')
export=ROOT/'HRDecoder_Next_20261003/configs/U0_historical_native_export.yaml'
for line,s in enumerate(export.read_text(encoding='utf-8-sig').splitlines(),1):
    m=re.match(r'\s*(effective_batch|micro_batch|grad_accum):\s*(\d+)\s*$',s)
    if m:add(int(m.group(2)),export,f'line {line}, {m.group(1)}',s.strip(),'archived ConvNeXt export configuration declaration; original checkpoint configSHA not matched; not independently recovered actual microbatch metadata')
fields('HRDecoder_Q_20261003/server_results/configs/protocol.json',['/optimization/micro_batch','/optimization/accumulation','/optimization/frozen_updates','/optimization/head_lr','/optimization/backbone_lr','/optimization/weight_decay','/optimization/clip_norm','/optimization/head_warmup','/optimization/backbone_warmup','/optimization/schedule_horizon','/optimization/lr_floor_ratio','/experiment/base_endpoint'],'Q DDR partial-finetuning settings; retained16k uses24k learning-rate horizon')
fields('HRDecoder_NextPlan_20261004/execution/server_results/runs/F12_seed42/initialization.json',['/checkpoint_completed'],'actual F12 initial checkpoint completed update')
fields('HRDecoder_NextPlan_20261004/execution/configs/protocol.json',['/F4_F12/new_stage_updates','/F4_F12/warmup_updates','/F4_F12/final_norm_peak_lr','/F4_F12/head_peak_lr','/F4_F12/weight_decay','/F4_F12/clip_norm','/F4_F12/resources/micro_batch','/F4_F12/resources/accumulation'],'F12 stage protocol; actual saved endpoint established separately')
payload={'scope':'numeric extraction from actual retained recipes, configuration ancestry and checkpoint metadata; no execution','records':rows}
(E/'revision2_training_numeric_sources.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
reg=read(OUT/'numeric_evidence.json');reg['entries']=[x for x in reg['entries'] if x['source']!='evidence/revision2_training_numeric_sources.json']
for i,r in enumerate(rows):reg['entries'].append({'value':r['value'],'source':'evidence/revision2_training_numeric_sources.json','selector':{'kind':'json-pointer','pointer':f'/records/{i}/value'},'representations':['raw'],'tolerance':1e-10,'note':r['role']+'; exact source/line/JSON pointer and hash retained in source record'})
(OUT/'numeric_evidence.json').write_text(json.dumps(reg,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'extracted_numeric_training_records':len(rows),'total_registry_entries':len(reg['entries'])}))
