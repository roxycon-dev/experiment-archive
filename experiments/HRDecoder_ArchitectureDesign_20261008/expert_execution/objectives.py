"""Independent base and residual supervision; no moving student action targets."""
import torch
from fast_objectives import gt_loss,residual_gt,trusted_kl

def lambda_k(update):
    assert 0<=update<=40000
    if update<=1500:return 0.
    if update<=4000:return .5*(update-1500)/2500
    if update<=32000:return .5
    return .5-.4*(update-32000)/8000

def learning_rate(index):
    assert 0<=index<40000
    value=.005*(1-index/40000)
    if index<1500:value*=1-(1-index/1500)*(1-1e-6)
    return value

@torch.no_grad()
def fixed_teacher(a,u,s,label):
    a,u,s=(x.detach().float() for x in (a,u,s))
    q=.5625*a+.1875*u+.25*s
    high=torch.maximum(torch.maximum(a,u),s);low=torch.minimum(torch.minimum(a,u),s)
    labels=label[:,0] if label.ndim==4 else label
    y=torch.stack([labels==c for c in range(1,5)],1)
    # Correction depends only on cached train teachers and train GT, never on z0.
    return torch.where(y&(q<=.5)&(high>.5),high,
                       torch.where(~y&(q>.5)&(low<=.5),low,q))

def losses(output,label,valid,a,u,s,update):
    private_gt=gt_loss(output['private_logits'],label,valid)
    final_gt=gt_loss(output['logits'],label,valid)
    base_gt=gt_loss(output['base_logits'],label,valid)
    kd=trusted_kl(output['private_logits'],fixed_teacher(a,u,s,label),label,valid)
    lk=lambda_k(update)
    task=base_gt+.1*output['hr_loss']+private_gt+.5*final_gt+lk*kd
    zero=task.detach()*0
    return task,zero,{'gt_private':private_gt,'gt_final':final_gt,'gt_base':base_gt,
      'gt_hr_crop':output['hr_loss'],'kl_private_GT_strata_balanced':kd,'lambda_k':lk,
      'fusion_alpha_mean':output['fusion_alpha'].mean(),'task_loss':task,
      'ordinary_BCE_removed':True,'teacher_target_independent_of_student':True}

def routed_backward(task,auxiliary,parameters,*,action_active=True):
    # All desired paths are already encoded by detached provider boundaries.
    task.backward()
