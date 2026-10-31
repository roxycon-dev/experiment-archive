"""Analytic counterexamples and CPU float64 gradients. No training or optimizer."""
import json
from pathlib import Path
import torch
from torch.nn import functional as F

torch.set_num_threads(1)
torch.manual_seed(7)
torch.set_default_dtype(torch.float64)

def image_class_mean(value, valid):
    active=valid.sum((-2,-1))>0
    return torch.where(active,value,0.).sum()/active.sum().clamp_min(1)

def masked_task(z, y, teacher, valid, lambda_k=.5):
    v=valid.to(z.dtype).expand_as(z)
    p=z.sigmoid()
    dice=1-(2*(p*y*v).sum((-2,-1))+1e-5)/(((p.square()+y.square())*v).sum((-2,-1))+1e-5)
    t=teacher.detach()
    entropy=-(torch.xlogy(t,t)+torch.xlogy(1-t,1-t))
    kl=F.binary_cross_entropy_with_logits(z,t,reduction='none')-entropy
    kl=(kl*v).sum((-2,-1))/v.sum((-2,-1)).clamp_min(1)
    return image_class_mean(dice+lambda_k*kl,v)

def signed_aux(preds, targets, valid):
    losses=[]
    for pred,target in zip(preds,targets):
        d=target.detach()
        v=valid.to(pred.dtype).expand_as(pred)
        w=v*(1+4*d.abs())
        loss=F.smooth_l1_loss(pred,d,beta=.1,reduction='none')
        value=(w*loss).sum((-2,-1))/w.sum((-2,-1)).clamp_min(1)
        losses.append(image_class_mean(value,v))
    return sum(losses)/len(losses)

def lambda_k(t):
    if t<=1500: return 0.
    if t<=4000: return .5*(t-1500)/2500
    if t<=32000: return .5
    return .5-.4*(t-32000)/8000

def run():
    checks, detail = {}, {}
    def check(k, value, evidence=None):
        checks[k] = bool(value)
        if evidence is not None:
            detail[k] = evidence
        assert value, (k, evidence)
    a = torch.tensor([.25, .25, .5])
    tj = torch.tensor([.1, .3, .9])
    z = torch.tensor(.37, requires_grad=True)
    unweighted = (a*F.binary_cross_entropy_with_logits(z.expand(3), tj, reduction='none')).sum()
    mixed = F.binary_cross_entropy_with_logits(z, (a*tj).sum())
    grad = torch.autograd.grad(unweighted, z, retain_graph=True)[0]
    mg = torch.autograd.grad(mixed, z)[0]
    check('unweighted_stage_KD_shared_output_equivalence', torch.allclose(unweighted, mixed) and torch.allclose(grad, mg))
    y = torch.tensor(1.)
    c = y*tj+(1-y)*(1-tj)
    mass = (a*c).sum()
    teff = (a*c*tj).sum()/mass
    loss = (a*c*F.binary_cross_entropy_with_logits(z.expand(3), tj, reduction='none')).sum()
    g = torch.autograd.grad(loss, z)[0]
    check('v2_actual_weighted_shared_gradient_equivalence', torch.allclose(g, mass*(z.sigmoid()-teff)),
          {'effective_target': teff.item(), 'mass': mass.item(), 'ordinary_mixture': (a*tj).sum().item()})
    stage_matrix = torch.tensor([[1.,0.,0.],[.75,.25,0.],[.5625,.1875,.25]])
    expanded = a@stage_matrix
    check('initial_ordinary_KD_teacher_weights', torch.allclose(expanded, torch.tensor([.71875,.15625,.125])), expanded.tolist())
    # Branch gradients differ even when their output values are zero.
    base, r1, r2 = (torch.tensor(.37, requires_grad=True), torch.tensor(0., requires_grad=True), torch.tensor(0., requires_grad=True))
    stages = torch.stack([base, base+r1, base+r1+r2])
    branched = (a*F.binary_cross_entropy_with_logits(stages, tj, reduction='none')).sum()
    bg = torch.autograd.grad(branched, (base,r1,r2))
    check('v2_not_all_parameter_gradients_identical', not torch.allclose(bg[0],bg[1]) and not torch.allclose(bg[1],bg[2]), [v.item() for v in bg])
    teachers = torch.rand(1000,3)
    old,u,s = teachers.unbind(1)
    p0 = .75*old+.25*u
    mean = .75*p0+.25*s
    du,ds = u-old,s-p0
    old_r = mean-.25*du-.25*ds
    reconstructed = torch.stack([old_r,old_r+du,old_r+.25*du+ds],1)
    check('mean_signed_contrasts_inverse', torch.allclose(teachers,reconstructed,atol=1e-14), (teachers-reconstructed).abs().max().item())
    pair = torch.tensor([[.5,.1,.8],[.5,.9,.2]])
    pair_p0 = .75*pair[:,0]+.25*pair[:,1]
    pair_mean = .75*pair_p0+.25*pair[:,2]
    pair_d = torch.stack([pair[:,1]-pair[:,0],pair[:,2]-pair_p0],1)
    check('same_final_mean_different_signed_information', torch.allclose(pair_mean[0],pair_mean[1]) and not torch.allclose(pair_d[0],pair_d[1]),
          {'teachers':pair.tolist(),'final_mean':pair_mean.tolist(),'signed':pair_d.tolist()})
    check('absolute_contrasts_lose_sign', torch.allclose(pair_d[0].abs(),pair_d[1].abs()))
    # Equal hidden states do not imply equal independently read outputs.
    hidden = torch.tensor([1.])
    first, second = hidden*.4, hidden*(-.4)
    check('different_readouts_do_not_prove_hidden_diversity', first.item()!=second.item())
    # If two inputs are indistinguishable to a downsampled student, no capacity
    # removes conditional target variance: optimal scalar prediction is 0.5.
    target = torch.tensor([.1,.9])
    mean_prediction = target.mean()
    best_mse = ((target-mean_prediction)**2).mean()
    check('lost_input_information_has_conditional_error_floor', torch.allclose(best_mse, torch.tensor(.16)), best_mse.item())
    # Zero final residual projection can ignore perfect contrast predictions.
    correct_mean = torch.tensor(.5)
    final_a = correct_mean+torch.zeros(2).dot(pair_d[0])
    final_b = correct_mean+torch.zeros(2).dot(pair_d[1])
    check('perfect_contrasts_need_not_change_final_output', torch.equal(final_a,final_b))
    # For y=0, A=.05 U=.95 S=.05, recording U-A need not output U.
    weak = torch.tensor([.05,.95,.05])
    weak_mean = .5625*weak[0]+.1875*weak[1]+.25*weak[2]
    check('weak_teacher_error_not_forced_into_final_label', weak[1]>.5 and weak_mean<.5, {'weak_U':.95,'final_T':weak_mean.item()})
    # Actual v3 task/auxiliary formulas: ignore invariance and target detachment.
    logits=torch.randn(2,4,5,7,requires_grad=True)
    target=torch.rand_like(logits,requires_grad=True)
    label=(torch.rand_like(logits)>.9).to(logits.dtype)
    valid=torch.ones(2,1,5,7)
    valid[...,:2,:]=0
    pred=[torch.randn_like(logits,requires_grad=True) for _ in range(2)]
    diffs=[torch.rand_like(logits,requires_grad=True)*2-1 for _ in range(2)]
    changed=logits.detach().clone(); changed[...,:2,:]=100
    changed_label=label.clone(); changed_label[...,:2,:]=1-label[...,:2,:]
    changed_target=target.detach().clone(); changed_target[...,:2,:]=1-target[...,:2,:]
    check('actual_task_ignores_masked_positions',torch.allclose(masked_task(logits,label,target,valid),masked_task(changed,changed_label,changed_target,valid)))
    changed_pred=[p.detach().clone() for p in pred]
    for p in changed_pred: p[...,:2,:]=100
    check('actual_aux_ignores_masked_predictions',torch.allclose(signed_aux(pred,diffs,valid),signed_aux(changed_pred,diffs,valid)))
    combined=masked_task(logits,label,target,valid)+.1*signed_aux(pred,diffs,valid)
    gtarg=torch.autograd.grad(combined,[target]+diffs,allow_unused=True,retain_graph=True)
    check('all_teacher_targets_detached',all(g is None for g in gtarg))
    grads=torch.autograd.grad(combined,[logits]+pred,retain_graph=True)
    check('actual_losses_finite_gradients',all(torch.isfinite(g).all() for g in grads))
    empty=torch.zeros_like(valid)
    nothing=masked_task(logits,label,target,empty)+signed_aux(pred,diffs,empty)
    ng=torch.autograd.grad(nothing,[logits]+pred)
    check('all_invalid_actual_losses_graph_zero',nothing.item()==0 and all(g.abs().sum()==0 for g in ng))
    weights=torch.stack([1+4*d.detach().abs() for d in diffs])
    check('auxiliary_pixel_weights_bounded',weights.min()>=1 and weights.max()<=5)
    check('schedules_match_spec',lambda_k(1500)==0 and lambda_k(4000)==.5 and lambda_k(32000)==.5 and abs(lambda_k(40000)-.1)<1e-12)
    recovery = 1-.5/(45.8603838-42.20)
    detail['engineering_close_requires_recovery_of_historical_gain'] = recovery
    result = {'scope':__doc__,'version':3,'passed':len(checks),'failed':0,'checks':checks,'details':detail,
              'does_not_validate':['capacity sufficiency','generalization','innovation','accuracy forecast','full-model integration']}
    out=Path(__file__).parent/'evidence/v3_objective_checks.json'
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':
    run()
