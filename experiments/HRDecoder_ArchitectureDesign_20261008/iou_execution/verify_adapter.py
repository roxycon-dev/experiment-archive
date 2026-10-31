"""Actual adapter CPU routing/init/resume checks; small provider fixture only."""
import copy
import hashlib
import json
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
from model import SACDStudent
from fast_objectives import gt_loss,residual_gt,trusted_kl,balanced_bce,stratified_teacher_kl,lovasz_hinge
from objectives import fixed_teacher,losses

torch.set_num_threads(4);torch.manual_seed(42)
HERE=Path(__file__).resolve().parent

class Head(nn.Module):
 def __init__(self):super().__init__();self.conv=nn.Conv2d(720,4,1)
 def forward(self,x):return self.conv(x[0])

class Base(nn.Module):
 def __init__(self):
  super().__init__();self.hr_scale=(32,32);self.decode_head=Head()
  self.layers=nn.ModuleList([nn.Conv2d(3,c,1) for c in (48,96,192,384)])
 def extract_feat(self,image):return tuple(layer(F.adaptive_avg_pool2d(image,(8//2**i,8//2**i))) for i,layer in enumerate(self.layers))
 def _transform_inputs(self,features):return torch.cat([F.interpolate(x,size=(8,8),mode='bilinear',align_corners=False) for x in features],1)
 def hr_slide_inference(self,size,fused,unused):return {'logits':[F.interpolate(self.decode_head([fused]),size=size,mode='bilinear',align_corners=False)],'boxes':[(0,size[0],0,size[1])]}
 def _fuse_logits(self,a,b):return .5*(a+b)

model=SACDStudent(Base(),canvas=64)
assert sum(p.numel() for p in model.private.parameters())==284496
optimizer=torch.optim.SGD([{'params':model.base.parameters(),'lr_multiplier':1},
 {'params':model.private.parameters(),'lr_multiplier':4}],lr=.004,momentum=.9,weight_decay=.0005,foreach=True)
image=torch.randn(2,3,64,64);label=torch.randint(0,5,(2,1,64,64));valid=torch.ones_like(label)
valid[:,:,-4:]=0;label[:,:,:2]=255
teacher=torch.rand(2,4,64,64)*.8+.1
output=model(image)
assert torch.equal(output['logits'],output['base_logits'].float())
assert output['residual_logits'].count_nonzero()==0
residual_gt(output['logits'],label,valid).backward()
assert all(p.grad is None for p in model.base.parameters())
assert model.private.readout.weight.grad.count_nonzero()>0
optimizer.zero_grad(set_to_none=True)
gt_loss(model(image)['base_logits'],label,valid).backward()
assert any(p.grad is not None and p.grad.count_nonzero()>0 for p in model.base.parameters())
assert all(p.grad is None for p in model.private.parameters())
def update(m,opt,private_scale=1.):
 opt.zero_grad(set_to_none=True)
 output=m(image)
 loss,_,_=losses(output,label,valid,teacher,teacher,teacher,4000)
 assert torch.isfinite(loss);loss.backward()
 assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
 for group in opt.param_groups:group['lr']=.004*group['lr_multiplier']
 for p in m.private_parameters():
  if p.grad is not None:p.grad.mul_(private_scale)
 torch.nn.utils.clip_grad_norm_(m.base.parameters(),5.,foreach=True)
 torch.nn.utils.clip_grad_norm_(m.private_parameters(),5.,foreach=True)
 opt.step()
for _ in range(3):update(model,optimizer)
parts={'features':model.private.features,'semantic':model.private.semantic,'rgb':model.private.rgb,'mix':model.private.mix,
       'blocks':model.private.blocks,'readout':model.private.readout}
assert all(any(p.grad is not None and p.grad.count_nonzero()>0 for p in part.parameters()) for part in parts.values())
state=copy.deepcopy(model.state_dict());optstate=copy.deepcopy(optimizer.state_dict())
model2=SACDStudent(Base(),canvas=64)
optimizer2=torch.optim.SGD([{'params':model2.base.parameters(),'lr_multiplier':1},
 {'params':model2.private.parameters(),'lr_multiplier':4}],lr=.004,momentum=.9,weight_decay=.0005,foreach=True)
model2.load_state_dict(state,strict=True);optimizer2.load_state_dict(copy.deepcopy(optstate))
update(model,optimizer);update(model2,optimizer2)
assert all(torch.equal(value,model2.state_dict()[name]) for name,value in model.state_dict().items())
assert [group['lr'] for group in optimizer2.param_groups]==[.004,.016]
empty=torch.zeros_like(valid)
assert gt_loss(model(image)['logits'],label,empty)==0

# The rescue loss must retain a positive gradient at extreme negative logits.
saturated=torch.full((2,4,8,8),-100.,requires_grad=True);positive_labels=torch.ones((2,1,8,8),dtype=torch.long);positive_valid=torch.ones_like(positive_labels)
rescue=balanced_bce(saturated,positive_labels,positive_valid);rescue.backward()
assert saturated.grad[:,0].lt(0).all() and saturated.grad[:,0].abs().mean()>1e-4
assert torch.isfinite(rescue) and torch.isfinite(saturated.grad).all()
model.zero_grad(set_to_none=True);z=model(image)['base_logits'];q=fixed_teacher(teacher,teacher,teacher,label)
stratified_teacher_kl(z,q,label,valid).backward()
assert any(p.grad is not None and p.grad.count_nonzero()>0 for p in model.base.parameters())
assert all(p.grad is None for p in model.private.parameters())
assert balanced_bce(z,label,torch.zeros_like(valid))==0
assert stratified_teacher_kl(z,q,label,torch.zeros_like(valid))==0
# Full-group normalization, not trusted-subset normalization.
test_z=torch.zeros((1,4,2,2));test_y=torch.ones((1,1,2,2),dtype=torch.long);test_v=torch.ones_like(test_y);test_q=torch.full_like(test_z,.75)
a=stratified_teacher_kl(test_z,test_q,test_y,test_v);test_q[:,0,0,0]=.25
b=stratified_teacher_kl(test_z,test_q,test_y,test_v)
assert torch.allclose(b,.75*a,atol=1e-7)

# Actual private objective retains the positive rescue derivative.
private_rescue=torch.full((2,4,8,8),-100.,requires_grad=True)
residual_gt(private_rescue,positive_labels,positive_valid).backward()
assert private_rescue.grad[:,0].lt(0).all() and private_rescue.grad[:,0].abs().mean()>1e-4
model.zero_grad(set_to_none=True)
stratified_teacher_kl(model(image)['logits'],q,label,valid).backward()
assert all(p.grad is None for p in model.base.parameters())
assert model.private.readout.weight.grad.count_nonzero()>0

# Matched actual loss/optimizer steps with only private gradient scale changed.
update(model,optimizer,private_scale=1.)
update(model2,optimizer2,private_scale=10000.)
assert all(torch.equal(v,model2.base.state_dict()[n]) for n,v in model.base.state_dict().items())
# Raw detached logits enter the readout without probability or normalization.
seen=[]
hook=model.private.readout.register_forward_pre_hook(lambda module,args:seen.append(args[0]))
out=model(image);hook.remove()
assert seen[0].shape[1]==80
assert torch.equal(seen[0][:,-16:],F.pixel_unshuffle(out['base_logits'].detach().float(),2))

# Exact active-margin implementation vs dense sorted-error definition.
def dense_lovasz(z,label,valid):
 y,m=__import__('fast_objectives').masks(z,label,valid);values=[]
 for b in range(z.shape[0]):
  for c in range(4):
   zz=z[b,c][m[b,c]];yy=y[b,c][m[b,c]].to(z.dtype)
   if not zz.numel():continue
   error=1-zz*(2*yy-1);es,idx=torch.sort(error,descending=True);gs=yy[idx]
   intersection=yy.sum()-gs.cumsum(0);union=yy.sum()+(1-gs).cumsum(0)
   j=1-intersection/union;weights=torch.cat((j[:1],j[1:]-j[:-1]))
   values.append(torch.dot(es.clamp_min(0),weights))
 return torch.stack(values).mean() if values else z.sum()*0
for seed in range(5):
 torch.manual_seed(900+seed)
 zz=(torch.randn(2,4,7,9,dtype=torch.float64)*4).requires_grad_();ll=torch.randint(0,5,(2,1,7,9));vv=torch.randint(0,2,ll.shape);ll[:,:,:1]=255
 aa=lovasz_hinge(zz,ll,vv);bb=dense_lovasz(zz,ll,vv)
 ga=torch.autograd.grad(aa,zz,retain_graph=True)[0];gb=torch.autograd.grad(bb,zz)[0]
 assert torch.allclose(aa,bb,atol=1e-12) and torch.allclose(ga,gb,atol=1e-12)
 perm=torch.randperm(63);zp=zz.detach().reshape(2,4,63)[:,:,perm].reshape_as(zz);lp=ll.reshape(2,1,63)[:,:,perm].reshape_as(ll);vp=vv.reshape(2,1,63)[:,:,perm].reshape_as(vv)
 assert torch.allclose(lovasz_hinge(zp,lp,vp),aa,atol=1e-12)
zz=torch.full((1,4,2,2),-100.,requires_grad=True);ll=torch.ones((1,1,2,2),dtype=torch.long);vv=torch.ones_like(ll)
ll[:, :,0,0]=0;zz.data[:,0,0,0]=100
loss=lovasz_hinge(zz,ll,vv);loss.backward();assert torch.isfinite(loss) and torch.isfinite(zz.grad).all()
assert zz.grad[:,0,0,0]>0 and zz.grad[:,0,1,1]<0
assert lovasz_hinge(zz,ll,torch.zeros_like(vv))==0
zz=torch.full((1,4,2,2),-2.,requires_grad=True);ll=torch.zeros((1,1,2,2),dtype=torch.long);assert lovasz_hinge(zz,ll,vv)==0
zz2=zz.detach().clone().requires_grad_();zz2.data[:,2,0,0]=2;empty_class_loss=lovasz_hinge(zz2,ll,vv);empty_class_loss.backward();assert empty_class_loss>0 and zz2.grad[:,2,0,0]>0

names=('model.py','private_modules.py','objectives.py','fast_objectives.py','verify_adapter.py')
report={'status':'PASS','torch':torch.__version__,'fixture_canvas':64,'batch':2,'private_parameters':284496,
 'checks':['zero_residual_exact_original_base_output','private_GT_KD_gradients_do_not_update_base',
 'base_GT_gradients_do_not_update_private','all6_adapter_parts_finite_nonzero_gradients',
 'single_SGD_two_groups4x_LR','exact_two_group_momentum_resume','empty_valid_mask_zero_loss','balanced_BCE_retains_positive_gradient_at_minus100','base_teacher_KL_updates_provider_not_private','base_KL_fullGT_group_denominators','separate_grad_clip5_resume_equivalence','actual_private_GT_retains_saturated_positive_gradient','balanced_private_KL_cannot_update_base','private_gradient_scaling_does_not_change_base_SGD_update','scale_preserving_concat_and_rawlogit_bypass','active_lovasz_matches_dense_value_and_gradient','lovasz_margin_gradient_survives_saturation','lovasz_false_positive_has_correct_gradient_sign','lovasz_empty_class_ignore_and_permutation'],
 'source_SHA256':{n:hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in names},
 'real_W48_GPU_gate_completed':False,'fresh_accuracy_verified':False,'test_accessed':False}
(HERE/'CPU_ADAPTER_PASS.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
