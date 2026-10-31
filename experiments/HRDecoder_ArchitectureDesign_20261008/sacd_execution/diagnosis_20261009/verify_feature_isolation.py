"""CPU synthetic proof of a proposed cut; never import/train a live baseline."""
import copy
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import sys

import torch

HERE=Path(__file__).resolve().parent
CORE=HERE.parent
sys.path.insert(0,str(CORE))
from model import SACDPrivate
from fast_objectives import gt_loss
from objectives import transport_logits, complementary_targets

torch.set_num_threads(4)
torch.manual_seed(20261009)
old=SACDPrivate().double()
# Zero-initialized heads have zero feature gradients by construction. A small
# nonzero synthetic readout represents a learned state, not actual checkpoint.
with torch.no_grad():
    for head in old.head.readouts:
        head[-1].weight.normal_(std=.03)
        head[-1].bias.normal_(std=.1)
new=copy.deepcopy(old)
image=torch.randn(2,3,64,64,dtype=torch.float64)
shapes=[(48,8,8),(96,4,4),(192,2,2),(384,1,1)]
original=[torch.randn(2,*shape,dtype=torch.float64) for shape in shapes]
features_old=[f.clone().requires_grad_() for f in original]
features_new=[f.clone().requires_grad_() for f in original]
reference=torch.randn(2,4,64,64,dtype=torch.float64)
z_old=reference.clone().requires_grad_()
z_new=reference.clone().requires_grad_()
label=torch.randint(0,5,(2,1,64,64))
valid=torch.ones(2,1,64,64)
action_old=old(image,features_old,z_old)
action_new=new(image,tuple(f.detach() for f in features_new),z_new)
assert torch.equal(action_old,action_new)
logits_old,_=transport_logits(z_old,action_old)
logits_new,_=transport_logits(z_new,action_new)
assert torch.equal(logits_old,logits_new)
loss_old=gt_loss(logits_old,label,valid)
loss_new=gt_loss(logits_new,label,valid)
assert torch.equal(loss_old,loss_new)
loss_old.backward();loss_new.backward()
assert all(f.grad is not None and f.grad.abs().sum()>0 for f in features_old)
assert all(f.grad is None for f in features_new)
assert z_old.grad.abs().sum()>0 and torch.equal(z_old.grad,z_new.grad)
assert all(torch.equal(a.grad,b.grad) for a,b in zip(old.parameters(),new.parameters()))

# Establish the target jump from the actual reviewed implementation, not a
# simulation of training accuracy or the frequency of such transitions.
student=torch.tensor([.499,.501]).reshape(1,1,1,2).expand(1,4,1,2)
base=torch.logit(student)
positive=torch.ones(1,1,1,2,dtype=torch.long)
negative=torch.zeros_like(positive)
mask=torch.ones_like(positive,dtype=torch.float32)
teacher_high=torch.full_like(student,.9)
teacher_low=torch.full_like(student,.1)
pos=complementary_targets(teacher_high,teacher_high,teacher_high,positive,mask,base)
neg=complementary_targets(teacher_low,teacher_low,teacher_low,negative,mask,base)
positive_targets=pos['action_target'][0,0,:,0,:].T.tolist()
negative_targets=neg['action_target'][0,0,:,0,:].T.tolist()
assert positive_targets[0][1]>.79 and positive_targets[1]==[1.,0.,0.]
assert negative_targets[0]==[1.,0.,0.] and negative_targets[1][2]>.79
report={'status':'PASS','checked_at':datetime.now(timezone(timedelta(hours=8))).isoformat(),
    'torch':torch.__version__,'device':'CPU','dtype':'float64','synthetic_canvas':64,'batch':2,
    'synthetic_nonzero_readout':True,'actual_trained_checkpoint_used':False,
    'forward_and_loss_bitwise_equal':True,'private_parameter_gradients_bitwise_equal':True,
    'direct_z0_gradient_preserved_bitwise':True,'private_feature_gradient_cut':True,
    'old_feature_gradient_norms':[float(f.grad.norm()) for f in features_old],
    'positive_action_targets_keep_on_off_at_p0499_p0501':positive_targets,
    'negative_action_targets_keep_on_off_at_p0499_p0501':negative_targets,
    'patch_SHA256':hashlib.sha256((HERE/'feature_isolation.patch').read_bytes()).hexdigest(),
    'formal_training_changed':False,'GPU_used':False,'checkpoint_download':False,
    'accuracy_validated':False,'causal_degradation_root_cause_proven':False,
    'limitations':'Synthetic private-branch gradient property only. Requires separately authorized fresh full-budget experiment; direct z0 coupling and moving targets remain.'}
(HERE/'FEATURE_ISOLATION_CPU_PASS.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print(json.dumps(report))
