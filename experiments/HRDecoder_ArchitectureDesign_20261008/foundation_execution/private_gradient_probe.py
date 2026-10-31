"""Registered train-only logit-gradient comparison; no optimizer updates."""
import sys,json,time,fcntl
from pathlib import Path
import torch
import torch.nn.functional as F
ROOT=Path('/root/autodl-tmp/HRDecoder_v3_20261008');HERE=ROOT/'foundation_execution';OUT=ROOT/'reports/foundation/private_gradient';OUT.mkdir(parents=True,exist_ok=True);sys.path.insert(0,str(HERE))
import train as t
from baseline import build_baseline,install_legacy_bf16_interpolation
from model import SACDStudent
from data import TrainCacheDataset
from loader import ResumableBatches,install_ram
from objectives import fixed_teacher
from fast_objectives import gt_loss,residual_gt,trusted_kl,balanced_bce,stratified_teacher_kl,masks
with open(ROOT/'reports/gpu.lock','a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);torch.set_num_threads(4);torch.backends.cudnn.benchmark=True;install_legacy_bf16_interpolation();install_ram(ROOT)
 ck=torch.load(ROOT/'training_runs/foundation/last.pt',map_location='cpu');assert ck['completed']==4000 and ck['identity_SHA']=='cc3a215c3e867779f4c5e2a6feedb5921c95b4cf48bfc9fe610003216be73bf8'
 base,_=build_baseline(t.REPO,t.W48,verified_weight_sha='d2186c55782308392ae14cf179ac4840143b09abdbba81aa3f9274c795ebc960');model=SACDStudent(base).cuda().train();model.load_state_dict(ck['model'],strict=True);t.restore_rng(ck['rng']);del ck
 ds=TrainCacheDataset(ROOT,t.DATA);results=[];start=time.time()
 for i,pair in enumerate(ResumableBatches(len(ds),4000,4008)):
  vals=[ds[x] for x in pair];batch={k:torch.stack([v[k] for v in vals]).cuda() for k in ('image','label','valid','A','U','S')}
  with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):out=model(batch['image'],batch['label'],batch['valid'])
  z=out['logits'].detach().float().requires_grad_(True);q=fixed_teacher(batch['A'],batch['U'],batch['S'],batch['label']);y,m=masks(z,batch['label'],batch['valid']);pos=y&m;neg=~y&m
  ld=gt_loss(z,batch['label'],batch['valid']);oldgt=residual_gt(z,batch['label'],batch['valid']);newgt=ld+.25*balanced_bce(z,batch['label'],batch['valid']);oldkd=.5*trusted_kl(z,q,batch['label'],batch['valid']);newkd=.5*stratified_teacher_kl(z,q,batch['label'],batch['valid'])
  loss_terms={'old_GT':oldgt,'balanced_GT':newgt,'old_KD':oldkd,'balanced_KD':newkd}
  gradients={n:torch.autograd.grad(v,z,retain_graph=True)[0] for n,v in loss_terms.items()}
  old=gradients['old_GT']+gradients['old_KD'];new=gradients['balanced_GT']+gradients['balanced_KD']
  row={'batch':i,'classes':[],'losses':{n:v.item() for n,v in loss_terms.items()}}
  for c,name in enumerate(('EX','HE','SE','MA')):
   p=pos[:,c];n=int(p.sum());data={'name':name,'GT_positive_pixels':n}
   if n:
    data.update(positive_recall_at_half=(z[:,c][p]>0).float().mean().item(),old_total_positive_gradient_mean=old[:,c][p].mean().item(),balanced_total_positive_gradient_mean=new[:,c][p].mean().item(),old_positive_lowering_fraction=(old[:,c][p]>0).float().mean().item(),balanced_positive_lowering_fraction=(new[:,c][p]>0).float().mean().item())
    for term,g in gradients.items():data[term+'_positive_gradient_absmean']=g[:,c][p].abs().mean().item()
   row['classes'].append(data)
  results.append(row);print(json.dumps(row),flush=True);del vals,batch,out,z,q,y,m,pos,neg,gradients,old,new,loss_terms
 receipt={'status':'COMPLETE','train_batches':len(results),'formal_SGD_updates':0,'parent_completed':4000,'identity_SHA':'cc3a215c3e867779f4c5e2a6feedb5921c95b4cf48bfc9fe610003216be73bf8','GT_source':'train only','val_or_test_accessed':False,'checkpoint_mutated':False,'records':results,'seconds':time.time()-start,'scope':'Logit-gradient evidence; not parameter-level causal proof or fresh generalization guarantee'}
 (OUT/'PRIVATE_GRADIENT_DIAGNOSTIC.json').write_text(json.dumps(receipt,indent=2)+'\n')
