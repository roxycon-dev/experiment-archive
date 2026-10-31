"""No-SGD train-only per-class positive-gradient diagnostic."""
import sys,json,time,fcntl
from pathlib import Path
import torch
import torch.nn.functional as F
ROOT=Path('/root/autodl-tmp/HRDecoder_v3_20261008');HERE=ROOT/'detail_execution';OUT=ROOT/'reports/detail/diagnosis_16000';sys.path.insert(0,str(HERE))
import train as t
from baseline import build_baseline,install_legacy_bf16_interpolation
from model import SACDStudent
from data import TrainCacheDataset
from loader import ResumableBatches
from fast_objectives import gt_loss,masks
with open(ROOT/'reports/gpu.lock','a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);torch.set_num_threads(4);torch.backends.cudnn.benchmark=True;install_legacy_bf16_interpolation()
 ck=torch.load(ROOT/'training_runs/detail/aborted_parent_21000.pt',map_location='cpu');assert ck['completed']==21000
 base,_=build_baseline(t.REPO,t.W48,verified_weight_sha='d2186c55782308392ae14cf179ac4840143b09abdbba81aa3f9274c795ebc960');model=SACDStudent(base).cuda().train();model.load_state_dict(ck['model'],strict=True);t.restore_rng(ck['rng']);del ck
 ds=TrainCacheDataset(ROOT,t.DATA,with_teachers=False);results=[];start=time.time()
 for i,pair in enumerate(ResumableBatches(len(ds),21000,21012)):
  vals=[ds[x] for x in pair];batch={k:torch.stack([v[k] for v in vals]).cuda() for k in ('image','label','valid')}
  with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):out=model(batch['image'],batch['label'],batch['valid'])
  z=out['base_logits'].detach().float().requires_grad_(True);y,m=masks(z,batch['label'],batch['valid']);pos=y&m;neg=~y&m
  ld=gt_loss(z,batch['label'],batch['valid']);gd=torch.autograd.grad(ld,z,retain_graph=True)[0]
  dims=(-2,-1);np=pos.sum(dims);nn=neg.sum(dims);pb=(F.softplus(-z)*pos).sum(dims)/np.clamp_min(1);nb=(F.softplus(z)*neg).sum(dims)/nn.clamp_min(1)
  lb=.25*(.5*pb+.5*nb).mean();gb=torch.autograd.grad(lb,z)[0];record={'batch':i,'Dice':ld.item(),'balanced_BCE_weighted':lb.item(),'classes':[]}
  for c,name in enumerate(['EX','HE','SE','MA']):
   mask=pos[:,c];n=int(mask.sum());row={'name':name,'positive_pixels':n}
   if n:
    v=z[:,c][mask].detach();d=gd[:,c][mask].abs().mean().item();b=gb[:,c][mask].abs().mean().item();row.update(positive_logit_mean=v.mean().item(),positive_logit_min=v.min().item(),positive_logit_max=v.max().item(),positive_recall_at_half=(v>0).float().mean().item(),Dice_positive_grad_absmean=d,balanced_BCE_positive_grad_absmean=b,BCE_to_Dice_grad_ratio=b/max(d,1e-30))
   record['classes'].append(row)
  results.append(record);print(json.dumps(record),flush=True);del vals,batch,out,z,y,m,pos,neg,gd,gb
 value={'status':'COMPLETE','train_batches':len(results),'formal_SGD_updates':0,'parent_completed':21000,'GT_source':'train only','val_or_test_accessed':False,'identity_SHA':'046abab773f7cb7a9e4f0a0f9cc44c4ad32d48431b971f22720dc8bd7936168a','records':results,'seconds':time.time()-start,'checkpoint_mutated':False,'scope':'Local logits-gradient diagnostic; gradients do not alone prove cause or fresh training outcome'}
 (OUT/'BASE_GRADIENT_DIAGNOSTIC.json').write_text(json.dumps(value,indent=2)+'\n')
