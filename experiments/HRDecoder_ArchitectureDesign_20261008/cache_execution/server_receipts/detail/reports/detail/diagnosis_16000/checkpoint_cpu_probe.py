"""Read-only CPU checkpoint diagnostic; never changes current training or checkpoints."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
import json,math,time,hashlib
from pathlib import Path
import torch
torch.set_num_threads(1)
ROOT=Path('/root/autodl-tmp/HRDecoder_v3_20261008')
WORK=ROOT/'training_runs/detail'
OUT=ROOT/'reports/detail/diagnosis_16000'
OUT.mkdir(parents=True,exist_ok=True)
started=time.time()
identity='046abab773f7cb7a9e4f0a0f9cc44c4ad32d48431b971f22720dc8bd7936168a'
best=torch.load(WORK/'best_val.pt',map_location='cpu')
last=torch.load(WORK/'last.pt',map_location='cpu')
assert best['identity_SHA']==identity and last['identity_SHA']==identity
assert best['completed']==12000 and last['completed']>=17000
assert 'rng' in last and last['optimizer']['state']
def summary(ck):
 state=ck['model']; nonfinite=[];negative=[];tiny=[];rows=[]
 for name,x in state.items():
  if x.is_floating_point() and not torch.isfinite(x).all().item():nonfinite.append(name)
  if name.endswith('running_var'):
   if x.min().item()<0:negative.append(name)
   if x.min().item()<1e-8:tiny.append({'name':name,'min':x.min().item()})
  if name.endswith('running_mean'):
   stem=name[:-len('running_mean')];v=state[stem+'running_var'];n=state.get(stem+'num_batches_tracked')
   ratio=x.abs()/(v.clamp_min(1e-12).sqrt())
   rows.append({'name':stem,'running_mean_absmax':x.abs().max().item(),'running_var_min':v.min().item(),'running_var_max':v.max().item(),'abs_mean_over_std_max':ratio.max().item(),'num_batches_tracked':int(n.item()) if n is not None else None})
 optim_bad=[]
 for key,value in ck.get('optimizer',{}).get('state',{}).items():
  for name,x in value.items():
   if torch.is_tensor(x) and x.is_floating_point() and not torch.isfinite(x).all().item():optim_bad.append(str(key)+'.'+name)
 return {'completed':ck['completed'],'identity_SHA':ck['identity_SHA'],'model_tensor_count':len(state),'model_nonfinite':nonfinite,'optimizer_nonfinite':optim_bad,'BN_layers':len(rows),'BN_negative_variance':negative,'BN_tiny_variance':tiny,'BN_rows':rows}
sb,sl=summary(best),summary(last)
diffs=[]
for name,x in last['model'].items():
 if not x.is_floating_point():continue
 y=best['model'][name]
 if name.endswith(('running_mean','running_var')) or ('decode_head' in name and any(k in name for k in ['conv_seg','cls_seg','bn'])):
  delta=(x-y).float();norm=y.float().norm().item();dn=delta.norm().item()
  diffs.append({'name':name,'reference_norm':norm,'current_norm':x.float().norm().item(),'delta_norm':dn,'relative_delta':dn/max(norm,1e-12),'delta_absmax':delta.abs().max().item()})
protocol=json.loads((WORK/'protocol.json').read_text());src=[]
for name,digest in protocol['identity']['code'].items():
 actual=hashlib.sha256((ROOT/'detail_execution'/name).read_bytes()).hexdigest()
 src.append({'name':name,'matches_registered_source':actual==digest,'sha256':actual})
assert all(q['matches_registered_source'] for q in src)
result={'status':'CPU_READ_ONLY_COMPLETED','time_unix':time.time(),'seconds':time.time()-started,'uses_GPU':False,'training_mutated':False,'checkpoint_overwritten':False,'full_data_or_cache_hash_rescan':False,'test_accessed':False,'scope':'best12k versus latest durable own checkpoint; exact16k checkpoint is not retained and cannot be inspected retrospectively','registered_identity_SHA':identity,'best':sb,'current':sl,'top_relative_deltas':sorted(diffs,key=lambda q:q['relative_delta'],reverse=True)[:40],'source_checks':src,'conclusion_limit':'Finite tensors and BN summaries alone do not establish why16k validation collapsed; no causal claim or recalibration performed'}
p=OUT/'CHECKPOINT_CPU_DIAGNOSTIC.json';tmp=p.with_suffix('.json.tmp');tmp.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');tmp.replace(p)
print(json.dumps({'status':result['status'],'seconds':result['seconds'],'best_step':sb['completed'],'current_step':sl['completed'],'best_nonfinite':len(sb['model_nonfinite']),'current_nonfinite':len(sl['model_nonfinite']),'optimizer_nonfinite':len(sl['optimizer_nonfinite']),'BN_layers':sl['BN_layers'],'negative_BN_variance':len(sl['BN_negative_variance']),'tiny_BN_variance':len(sl['BN_tiny_variance']),'source_matches':all(q['matches_registered_source'] for q in src)}))
