"""Finite train-only logit gradient diagnostic, not SGD or val/test."""
import sys,json,time,fcntl
from pathlib import Path
import torch
ROOT=Path('/root/autodl-tmp/HRDecoder_v3_20261008');H=ROOT/'semantic_execution';P=ROOT/'reports/semantic/loss_probe';P.mkdir(parents=True,exist_ok=True);sys.path.insert(0,str(H))
import train as t
from baseline import build_baseline,install_legacy_bf16_interpolation
from model import SACDStudent
from data import TrainCacheDataset
from loader import ResumableBatches,install_ram
from fast_objectives import masks,gt_loss,balanced_bce,stratified_teacher_kl
from objectives import fixed_teacher

def lovasz_active(z,label,valid):
 y,m=masks(z,label,valid);values=[]
 for b in range(z.shape[0]):
  for c in range(4):
   zz=z[b,c][m[b,c]];yy=y[b,c][m[b,c]].to(z.dtype)
   if zz.numel()==0:continue
   errors=1-zz*(2*yy-1);active=errors>0
   gts=yy.sum();ee=errors[active];gg=yy[active]
   if ee.numel()==0:values.append(zz.sum()*0);continue
   es,perm=torch.sort(ee,descending=True);gs=gg[perm]
   intersection=gts-gs.cumsum(0);union=gts+(1-gs).cumsum(0)
   jaccard=1-intersection/union.clamp_min(1);weights=torch.cat((jaccard[:1],jaccard[1:]-jaccard[:-1]))
   values.append(torch.dot(es,weights))
 return torch.stack(values).mean() if values else z.sum()*0

with open(ROOT/'reports/gpu.lock','a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);torch.set_num_threads(4);torch.backends.cudnn.benchmark=True;install_legacy_bf16_interpolation();install_ram(ROOT)
 ck=torch.load(ROOT/'training_runs/semantic/last.pt',map_location='cpu');assert ck['completed']==4000 and ck['identity_SHA']=='9e21de915c24dc3c16322cc39babb53cdc3754271b07e5f27ef22b10b43ddf43'
 base,_=build_baseline(t.REPO,t.W48,verified_weight_sha='d2186c55782308392ae14cf179ac4840143b09abdbba81aa3f9274c795ebc960');model=SACDStudent(base).cuda().eval();model.load_state_dict(ck['model'],strict=True);t.restore_rng(ck['rng']);del ck
 ds=TrainCacheDataset(ROOT,t.DATA);rows=[];started=time.time()
 for i,pair in enumerate(ResumableBatches(len(ds),4000,4008)):
  vals=[ds[x] for x in pair];batch={k:torch.stack([v[k] for v in vals]).cuda() for k in ('image','label','valid','A','U','S')}
  with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):out=model(batch['image'])
  z=out['logits'].detach().float().requires_grad_();basez=out['base_logits'].detach().float();y,m=masks(z,batch['label'],batch['valid']);q=fixed_teacher(batch['A'],batch['U'],batch['S'],batch['label']);start=time.time()
  components={'dice':gt_loss(z,batch['label'],batch['valid']), 'weighted_BCE':.25*balanced_bce(z,batch['label'],batch['valid']), 'weighted_KL':.5*stratified_teacher_kl(z,q,batch['label'],batch['valid']), 'lovasz':lovasz_active(z,batch['label'],batch['valid'])}
  data={'batch':i,'component_losses':{k:v.item() for k,v in components.items()},'classes':[]}
  for c,name in enumerate(('EX','HE','SE','MA')):
   pred=basez[:,c]>0;pos=y[:,c]&m[:,c];neg=(~y[:,c])&m[:,c]
   groups={'base_FN':pos&~pred,'base_TP':pos&pred,'base_FP':neg&pred,'base_TN':neg&~pred};data['classes'].append({'name':name,'group_pixels':{k:int(v.sum()) for k,v in groups.items()},'gradient_absolute_mass':{}})
  for n,loss in components.items():
   grad=torch.autograd.grad(loss,z,retain_graph=True)[0];assert bool(torch.isfinite(grad).all())
   for c in range(4):
    pred=basez[:,c]>0;pos=y[:,c]&m[:,c];neg=(~y[:,c])&m[:,c];groups={'base_FN':pos&~pred,'base_TP':pos&pred,'base_FP':neg&pred,'base_TN':neg&~pred}
    data['classes'][c]['gradient_absolute_mass'][n]={k:grad[:,c][v].abs().sum().item() for k,v in groups.items()}
   del grad
  data['gradient_probe_seconds']=time.time()-start;rows.append(data);print(json.dumps(data),flush=True)
  (P/'LOSS_GRADIENT_PROGRESS.json').write_text(json.dumps({'completed_train_batches':i+1,'SGD_updates':0}))
  del vals,batch,out,z,basez,y,m,q,components,data,groups
 receipt={'status':'COMPLETE','train_batches':8,'formal_SGD_updates':0,'parent_completed':4000,'parent_identity_SHA':'9e21de915c24dc3c16322cc39babb53cdc3754271b07e5f27ef22b10b43ddf43','GT_source':'train only for losses; noGTmodel input','val_or_test_accessed':False,'checkpoint_mutated':False,'records':rows,'seconds':time.time()-started,'scope':'logit gradient mass, not parameter gradient or finalaccuracy proof'}
 (P/'LOSS_GRADIENT_DIAGNOSTIC.json').write_text(json.dumps(receipt,indent=2)+'\n')
