"""Numerical and real private-graph checks, independent of dataset/test/weights."""
import copy
import io
import json
from pathlib import Path
import random
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from model import CorrectionHead,SACDPrivate,SACDStudent,inference_probabilities
from fast_objectives import gt_loss,residual_gt,trusted_kl
from objectives import losses,fixed_teacher,learning_rate,lambda_k

torch.set_num_threads(4);torch.manual_seed(123)
checks=[]
def passed(name):checks.append(name)
head=CorrectionHead().double()
x=torch.randn(2,32,3,4,dtype=torch.float64)
p=torch.rand(2,4,12,16,dtype=torch.float64,requires_grad=True)
r=head(x,x,p)
assert r.shape==p.shape and torch.count_nonzero(r)==0
assert torch.autograd.grad(r.sum(),p,allow_unused=True)[0] is None
assert sum(v.numel() for v in head.parameters())==36544
assert sum(v.numel() for v in SACDPrivate().parameters())==6415232
passed('zero_residual_exact_base_initialization_and_counts')
with torch.no_grad():
    for h in head.readouts:h[-1].weight.normal_(0,.03)
features=torch.cat((x,x,F.pixel_unshuffle(p.detach(),4)),1)
ref=torch.cat([F.pixel_shuffle(h(features),4) for h in head.readouts],1)
packed=head(x,x,p)
assert torch.allclose(ref,packed,atol=1e-12,rtol=1e-12)
params=tuple(head.parameters())
a=torch.autograd.grad(ref.square().mean(),params,retain_graph=True)
b=torch.autograd.grad(packed.square().mean(),params)
assert all(torch.allclose(u,v,atol=1e-12,rtol=1e-10) for u,v in zip(a,b))
passed('grouped_heads_equal_independent_values_and_gradients')

z=torch.randn(2,4,5,7,dtype=torch.float64,requires_grad=True)
y=torch.randint(0,5,(2,1,5,7));m=torch.ones_like(y)
p=z.sigmoid()
# Independent original BinaryDice formula when every pixel is valid.
reference=torch.stack([1-(2*(p[:,c]*(y[:,0]==c+1)).flatten(1).sum(1)+1e-5)/
    (p[:,c].square().flatten(1).sum(1)+(y[:,0]==c+1).flatten(1).sum(1)+1e-5)
    for c in range(4)],1).mean()
actual=gt_loss(z,y,m)
assert torch.allclose(reference,actual,atol=1e-12)
assert torch.allclose(torch.autograd.grad(reference,z,retain_graph=True)[0],
                      torch.autograd.grad(actual,z,retain_graph=True)[0],atol=1e-12)
empty=torch.zeros_like(y)
assert gt_loss(z,empty,m)>0 and torch.autograd.grad(gt_loss(z,empty,m),z,retain_graph=True)[0].abs().sum()>0
for fun in (gt_loss,residual_gt):
    loss=fun(z,y,m*0);assert loss==0 and torch.autograd.grad(loss,z,retain_graph=True)[0].abs().sum()==0
    ignored=fun(z,torch.full_like(y,255),m);assert ignored==0
passed('original_Dice_equivalence_empty_classes_and_invalid_mask')

q=torch.rand_like(z)
target=fixed_teacher(q,q,q,y)
assert not target.requires_grad
assert torch.isfinite(trusted_kl(z,target,y,m))
# No student probability enters target construction; q=0/1 remains finite.
assert torch.isfinite(trusted_kl(z,torch.zeros_like(z),y,m))
assert torch.isfinite(trusted_kl(z,torch.ones_like(z),y,m))
passed('fixed_detached_teacher_targets_and_KL_endpoints')

class FakeHead(nn.Module):
    def __init__(self):super().__init__();self.conv=nn.Conv2d(720,4,1)
    def forward(self,value):return self.conv(value[0])
class FakeBase(nn.Module):
    def __init__(self):
        super().__init__();self.hr_scale=(32,32);self.decode_head=FakeHead()
        self.layers=nn.ModuleList([nn.Conv2d(3,c,1) for c in (48,96,192,384)])
    def extract_feat(self,image):
        return tuple(layer(F.adaptive_avg_pool2d(image,(8//2**i,8//2**i))) for i,layer in enumerate(self.layers))
    def _transform_inputs(self,feat):
        return torch.cat([F.interpolate(f,size=(8,8),mode='bilinear',align_corners=False) for f in feat],1)
    def hr_slide_inference(self,size,fused,unused):
        logits=F.interpolate(self.decode_head([fused]),size=size,mode='bilinear',align_corners=False)
        return {'logits':[logits],'boxes':[(0,size[0],0,size[1])]}
    def _fuse_logits(self,a,b):return .5*(a+b)

student=SACDStudent(FakeBase(),canvas=64).double().eval()
image=torch.randn(2,3,64,64,dtype=torch.float64)
with torch.no_grad():
    for h in student.private.head.readouts:h[-1].weight.normal_(0,.03)
out=student(image)
assert out['logits'].dtype==torch.float32
label=torch.randint(0,5,(2,1,64,64));mask=torch.ones_like(label)
teacher=torch.rand_like(out['logits'])
base_params=tuple(student.base.parameters());private_params=student.private_parameters()
private_loss=residual_gt(out['logits'],label,mask)+.5*trusted_kl(out['logits'],teacher,label,mask)
assert all(g is None for g in torch.autograd.grad(private_loss,base_params,allow_unused=True,retain_graph=True))
private_grad=torch.autograd.grad(private_loss,private_params,retain_graph=True)
assert all(torch.isfinite(g).all() for g in private_grad) and any(g.abs().sum()>0 for g in private_grad)
task,aux,_=losses(out,label,mask,teacher,teacher,teacher,4000)
base_only=gt_loss(out['base_logits'],label,mask)
bg=torch.autograd.grad(base_only,base_params,retain_graph=True)
tg=torch.autograd.grad(task,base_params,retain_graph=True)
assert all(torch.equal(g,h) for g,h in zip(bg,tg))
assert aux==0
passed('actual_student_wrapper_private_base_gradient_isolation')
native=inference_probabilities(out,(31,37))
assert torch.equal(native,F.interpolate(out['logits'].float(),(31,37),mode='bilinear',align_corners=False).sigmoid())
passed('native_logit_inference_no_teacher_or_GT_inputs')

# Save/reload the actual private head, optimizer momentum, and sampling RNG.
head=CorrectionHead();opt=torch.optim.SGD(head.parameters(),lr=.005,momentum=.9,weight_decay=.0005)
def step(module,optimizer):
    xx=torch.randn(1,32,2,2);pp=torch.rand(1,4,8,8)
    loss=module(xx,xx,pp).square().mean()+module(xx,xx,pp).mean()
    optimizer.zero_grad();loss.backward();optimizer.step();return loss.detach()
step(head,opt)
buffer=io.BytesIO();rng=torch.get_rng_state()
torch.save({'model':head.state_dict(),'optimizer':opt.state_dict(),'rng':rng},buffer)
expected=step(head,opt)
expected_state=copy.deepcopy(head.state_dict())
buffer.seek(0);state=torch.load(buffer,weights_only=False)
other=CorrectionHead();other.load_state_dict(state['model'])
other_opt=torch.optim.SGD(other.parameters(),lr=.005,momentum=.9,weight_decay=.0005)
other_opt.load_state_dict(state['optimizer']);torch.set_rng_state(state['rng'])
actual=step(other,other_opt)
assert torch.equal(expected,actual) and all(torch.equal(expected_state[k],v) for k,v in other.state_dict().items())
assert lambda_k(1500)==0 and lambda_k(4000)==.5 and abs(lambda_k(40000)-.1)<1e-12
assert abs(learning_rate(0)-.005e-6)<1e-15
passed('SGD_momentum_RNG_serialization_and_uncompressed40k_schedule')
report={'status':'PASS','torch':torch.__version__,'checks':checks,'formal_updates':0,
        'private_parameters':6415232,'total_parameters_real_student':71987652,
        'dataset_or_test_read':False,'accuracy_validated':False}
(Path(__file__).resolve().parent/'CPU_RESIDUAL_PASS.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
