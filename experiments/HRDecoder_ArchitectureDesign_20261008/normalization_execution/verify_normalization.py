"""Small exact normalization and provider-gradient checks, no dataset/test."""
import json,sys
from pathlib import Path
import torch
from torch import nn
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
sys.path.insert(0,str(ROOT/'residual_execution'))
from intervention import balanced_kl
from fast_objectives import trusted_kl,gt_loss,residual_gt
import objectives
from objectives import losses

torch.set_num_threads(4);torch.manual_seed(42)
bias=torch.full((4,),-.5,dtype=torch.float64,requires_grad=True)
z=bias[None,:,None,None].expand(1,4,1,2);y=torch.tensor([[[[1,0]]]]);m=torch.ones_like(y)
q=torch.full_like(z,.1);q[0,0,0,0]=.9
small=balanced_kl(z,q,y,m)
large_z=torch.cat((z[:,:,:,:1],z[:,:,:,1:].repeat(1,1,1,19)),3)
large_y=torch.cat((y[:,:,:,:1],y[:,:,:,1:].repeat(1,1,1,19)),3)
large_q=torch.cat((q[:,:,:,:1],q[:,:,:,1:].repeat(1,1,1,19)),3)
large=balanced_kl(large_z,large_q,large_y,torch.ones_like(large_y))
assert torch.allclose(small,large,atol=1e-12)
assert torch.allclose(torch.autograd.grad(small,bias,retain_graph=True)[0],torch.autograd.grad(large,bias)[0],atol=1e-12)
assert balanced_kl(z,q,y,m*0)==0
assert torch.isfinite(balanced_kl(z,torch.zeros_like(q),y,m))
assert torch.isfinite(balanced_kl(z,torch.ones_like(q),y,m))
base=torch.randn(1,4,2,3,requires_grad=True);delta=torch.randn_like(base,requires_grad=True)
label=torch.randint(0,5,(1,1,2,3));mask=torch.ones_like(label);teacher=torch.rand_like(base)
out={'logits':base.detach()+delta,'base_logits':base,'hr_loss':base.sum()*0}
control,_,_=losses(out,label,mask,teacher,teacher,teacher,8000)
old=objectives.trusted_kl;objectives.trusted_kl=balanced_kl
variant,_,_=losses(out,label,mask,teacher,teacher,teacher,8000)
assert torch.equal(torch.autograd.grad(control,base,retain_graph=True)[0],torch.autograd.grad(variant,base,retain_graph=True)[0])
expected=.5*(balanced_kl(out['logits'],objectives.fixed_teacher(teacher,teacher,teacher,label),label,mask)-old(out['logits'],objectives.fixed_teacher(teacher,teacher,teacher,label),label,mask))
assert torch.allclose(variant-control,expected,atol=5e-7)
assert torch.isfinite(torch.autograd.grad(variant,delta)[0]).all()
objectives.trusted_kl=old
receipt={'status':'PASS','torch':torch.__version__,'formal_updates':0,'dataset_or_test_read':False,
 'checks':['GT_stratum_value_and_tied_gradient_invariant_to_background_replication','invalid_masks_zero_and_q_endpoints_finite','unchanged_provider_GT_gradient','only_registered_KL_term_changes','finite_private_gradients'],
 'fresh40k_screen_pass':False,'deadline_extension':False}
(HERE/'NORMALIZATION_CPU_PASS.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
