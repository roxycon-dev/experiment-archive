"""Train-only spatial/semantic residual diagnostic, no SGD or val/test."""
import sys,json,time,fcntl
from pathlib import Path
import torch
ROOT=Path('/root/autodl-tmp/HRDecoder_v3_20261008');HERE=ROOT/'aligned_execution';OUT=ROOT/'reports/aligned/spatial_probe';OUT.mkdir(parents=True,exist_ok=True);sys.path.insert(0,str(HERE))
import train as t
from baseline import build_baseline,install_legacy_bf16_interpolation
from model import SACDStudent
from data import TrainCacheDataset
from loader import ResumableBatches
from fast_objectives import masks
with open(ROOT/'reports/gpu.lock','a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);torch.set_num_threads(4);torch.backends.cudnn.benchmark=True;install_legacy_bf16_interpolation()
 ck=torch.load(ROOT/'training_runs/aligned/last.pt',map_location='cpu');assert ck['completed']==4000 and ck['identity_SHA']=='637a5e77db60fecb85c569b7bbbb9cdececdef317e8a62136f49a1f9ef1a5f04'
 base,_=build_baseline(t.REPO,t.W48,verified_weight_sha='d2186c55782308392ae14cf179ac4840143b09abdbba81aa3f9274c795ebc960');model=SACDStudent(base).cuda().eval();model.load_state_dict(ck['model'],strict=True);t.restore_rng(ck['rng']);del ck
 captured=[];hook=model.base.backbone.register_forward_hook(lambda module,args,output:captured.append(output))
 ds=TrainCacheDataset(ROOT,t.DATA,with_teachers=False);rows=[];start=time.time()
 for i,pair in enumerate(ResumableBatches(len(ds),4000,4008)):
  vals=[ds[x] for x in pair];batch={k:torch.stack([v[k] for v in vals]).cuda() for k in ('image','label','valid')}
  with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):out=model(batch['image'])
  features=captured.pop();assert not captured
  hooks=[p.register_forward_hook(lambda module,args,output:torch.zeros_like(output)) for p in model.private.features]
  with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):zero_semantic=model.private(batch['image'],features,out['base_logits'])
  for h in hooks:h.remove()
  y,m=masks(out['logits'],batch['label'],batch['valid']);r=out['residual_logits'].float();constant=r.mean((-2,-1),keepdim=True)
  maps={'base':out['base_logits'].float(),'final':out['logits'].float(),'mean_residual':out['base_logits'].float()+constant,'no_semantic':out['base_logits'].float()+zero_semantic.float()}
  data={'batch':i,'classes':[]}
  for c,name in enumerate(('EX','HE','SE','MA')):
   valid=m[:,c];pos=y[:,c]&valid;neg=~y[:,c]&valid;decisions={k:z[:,c]>0 for k,z in maps.items()};row={'name':name,'counts':{}}
   for k,p in decisions.items():row['counts'][k]=[int((p&pos).sum()),int((p&neg).sum()),int((~p&pos).sum())]
   row.update(residual_mean=r[:,c][valid].mean().item(),residual_std=r[:,c][valid].std().item(),full_vs_mean_hard_disagreement_fraction=((decisions['final']!=decisions['mean_residual'])&valid).sum().item()/int(valid.sum()),full_vs_no_semantic_hard_disagreement_fraction=((decisions['final']!=decisions['no_semantic'])&valid).sum().item()/int(valid.sum()))
   if bool(pos.any()):row['residual_GTpositive_mean']=r[:,c][pos].mean().item()
   if bool(neg.any()):row['residual_GTnegative_mean']=r[:,c][neg].mean().item()
   data['classes'].append(row)
  rows.append(data);print(json.dumps(data),flush=True);del vals,batch,out,features,r,constant,maps,decisions,y,m,zero_semantic
 hook.remove()
 receipt={'status':'COMPLETE','train_batches':8,'formal_SGD_updates':0,'parent_completed':4000,'GT_source':'train only for metrics; noGT model input','val_or_test_accessed':False,'checkpoint_mutated':False,'model_eval':True,'records':rows,'seconds':time.time()-start,'scope':'Paired diagnostic ablations only, no causal proof or fresh gate'}
 (OUT/'SPATIAL_DIAGNOSTIC.json').write_text(json.dumps(receipt,indent=2)+'\n')
