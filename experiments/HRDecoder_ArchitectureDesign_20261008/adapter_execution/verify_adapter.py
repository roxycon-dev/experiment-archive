"""Actual adapter CPU routing/init/resume checks; small provider fixture only."""
import copy
import hashlib
import json
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
from model import SACDStudent
from fast_objectives import gt_loss,residual_gt,trusted_kl
from objectives import fixed_teacher

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
assert sum(p.numel() for p in model.private.parameters())==68608
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
def update(m,opt):
 opt.zero_grad(set_to_none=True)
 output=m(image)
 loss=gt_loss(output['base_logits'],label,valid)+residual_gt(output['logits'],label,valid)
 loss=loss+.5*trusted_kl(output['logits'],fixed_teacher(teacher,teacher,teacher,label),label,valid)
 assert torch.isfinite(loss);loss.backward()
 assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
 for group in opt.param_groups:group['lr']=.004*group['lr_multiplier']
 opt.step()
for _ in range(3):update(model,optimizer)
parts={'features':model.private.features,'rgb':model.private.rgb,'probability':model.private.probability,
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
names=('model.py','private_modules.py','objectives.py','fast_objectives.py','verify_adapter.py')
report={'status':'PASS','torch':torch.__version__,'fixture_canvas':64,'batch':2,'private_parameters':68608,
 'checks':['zero_residual_exact_original_base_output','private_GT_KD_gradients_do_not_update_base',
 'base_GT_gradients_do_not_update_private','all5_adapter_parts_finite_nonzero_gradients',
 'single_SGD_two_groups4x_LR','exact_two_group_momentum_resume','empty_valid_mask_zero_loss'],
 'source_SHA256':{n:hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in names},
 'real_W48_GPU_gate_completed':False,'fresh_accuracy_verified':False,'test_accessed':False}
(HERE/'CPU_ADAPTER_PASS.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
