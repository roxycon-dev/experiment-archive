"""Focused numerical, supervision, gradient routing and deployment checks."""
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from model import CorrectionHead, SACDPrivate, inference_probabilities
from objectives import complementary_targets, action_kl, transport_logits, routed_backward, lambda_k, learning_rate
from stable_objectives import gt_loss, trusted_kl
import fast_objectives

torch.set_num_threads(2)
checks = []

def passed(name): checks.append(name)

# Exact analytic initialization and PixelShuffle subpixel ordering.
head = CorrectionHead().double()
l = torch.randn(2, 32, 4, 4, dtype=torch.float64)
p0 = torch.rand(2, 4, 16, 16, dtype=torch.float64, requires_grad=True)
actions = head(l, l, p0)
z = torch.logit(p0)
zp, logs = transport_logits(z, actions)
assert torch.allclose(zp.sigmoid(), .98*p0+.01, atol=1e-9)
assert torch.equal(zp > 0, z > 0)
grad = torch.autograd.grad(actions.sum(), p0, allow_unused=True)[0]
assert grad is None, 'Condition input must be detached'
passed('initialization_and_detached_condition')
private = SACDPrivate()
assert sum(p.numel() for p in private.parameters()) == 6419456
passed('architecture_parameter_counts')

# Endpoint teachers, all four region types, ignored and geometrically invalid pixels.
a = torch.tensor([0., 0., 1., 1., 0., 1., .5, 0.]).reshape(1,1,1,8).repeat(1,4,1,1)
u = torch.tensor([1., 0., 0., 1., 0., 1., .5, 1.]).reshape(1,1,1,8).repeat(1,4,1,1)
s = u.clone()
label = torch.tensor([[[[1,1,0,0,0,255,0,1]]]])
valid = torch.ones_like(label); valid[..., -1] = 0
t = complementary_targets(a,u,s,label,valid,torch.logit(.5625*a+.1875*u+.25*s))
assert t['regions'][0][0,0,0,0] and t['regions'][1][0,0,0,2]
assert t['unknown'][0,0,0,1] and t['unknown'][0,0,0,3]
assert t['regions'][2][0,0,0,4] and t['regions'][2][0,0,0,6]
assert not any(region[...,5:].any() for region in (t['regions'][0],t['regions'][1]))
assert not t['valid'][...,5].any() and not t['valid'][...,-1].any()
assert torch.isfinite(t['action_target']).all()
assert torch.allclose(t['action_target'].sum(2),torch.ones_like(a))
assert t['corrected_teacher'][0,0,0,0] == 1 and t['corrected_teacher'][0,0,0,2] == 0
passed('complementary_regions_endpoints_unknown_and_invalid')

# Missing regions retain fixed 1/3 and all invalid returns differentiable zero.
fake_logs = torch.tensor([.5,.25,.25]).log().reshape(1,1,3,1,1).repeat(1,4,1,1,2).requires_grad_()
allzero = torch.zeros(1,4,1,2)
lab = torch.zeros(1,1,1,2,dtype=torch.long); mask = torch.ones_like(lab)
target = complementary_targets(allzero,allzero,allzero,lab,mask,torch.full_like(allzero,-1000.))
value = action_kl(fake_logs,target)
assert abs(value.item() - (-torch.log(torch.tensor(.5))/3).item()) < 1e-7
empty = action_kl(fake_logs,complementary_targets(allzero,allzero,allzero,lab,mask*0,torch.full_like(allzero,-1000.)))
assert empty.item() == 0 and torch.autograd.grad(empty,fake_logs)[0].abs().sum() == 0
passed('action_fixed_three_regions_and_all_invalid')

# Error action has a nonvanishing BCE recovery gradient at extreme logits.
for positive in (True,False):
    anchor = torch.full((1,4,1,1), -1000. if positive else 1000.)
    raw = torch.full((1,4,3,1,1),-1000.)
    raw[:,:,2 if positive else 1] = 1000.; raw.requires_grad_()
    out, lp = transport_logits(anchor,raw)
    loss = F.binary_cross_entropy_with_logits(out,torch.ones_like(out) if positive else torch.zeros_like(out),reduction='sum')
    gradient = torch.autograd.grad(loss,raw)[0]
    assert torch.isfinite(out).all() and torch.isfinite(gradient).all()
    bad_action = 2 if positive else 1
    assert torch.allclose(gradient[:,:,bad_action],lp.exp()[:,:,bad_action],atol=1e-5)
passed('extreme_probabilities_and_action_recovery_gradient')

# Sparse positive+negative batch does not encourage class collapse.
bias = torch.full((4,),-20.,requires_grad=True)
logits = bias[None,:,None,None].expand(2,4,2,2)
labels = torch.zeros(2,1,2,2,dtype=torch.long); labels[0,0,0,0]=1
g = gt_loss(logits,labels,torch.ones_like(labels))
gradient = torch.autograd.grad(g,bias)[0]
assert gradient[0] < 0 and torch.isfinite(gradient).all()
ignored = gt_loss(logits,torch.full_like(labels,255),torch.ones_like(labels))
assert ignored.item() == 0
# A teacher wrong on all pixels contributes zero, denominator remains all valid.
q = torch.ones_like(logits); labs = torch.zeros_like(labels)
assert trusted_kl(logits,q,labs,torch.ones_like(labs)).item() == 0
passed('GT_class_recovery_ignore_and_KD_trust')

for blank in (False, True):
    zz=torch.randn(2,4,3,5,dtype=torch.float64,requires_grad=True)
    yy=torch.randint(0,5,(2,1,3,5)); yy[0,0,0,0]=255
    mm=torch.randint(0,2,(2,1,3,5)); mm=mm*0 if blank else mm
    qq=torch.rand_like(zz); qq[0,0,0,:2]=torch.tensor([0.,1.])
    for ref,fast,extra in ((gt_loss,fast_objectives.gt_loss,()),(trusted_kl,fast_objectives.trusted_kl,(qq,))):
        ra=ref(zz,*extra,yy,mm); fa=fast(zz,*extra,yy,mm)
        assert torch.allclose(ra,fa,atol=1e-12)
        rg=torch.autograd.grad(ra,zz,retain_graph=True)[0]
        fg=torch.autograd.grad(fa,zz,retain_graph=True)[0]
        assert torch.allclose(rg,fg,atol=1e-12)
passed('vectorized_objectives_match_reference_values_and_gradients')

# The selected auxiliary gradient never enters the shared provider.
shared = torch.nn.Linear(3,3); own = torch.nn.Linear(3,3)
x=torch.randn(2,3); feat=shared(x); value=own(feat)
task=value.square().mean(); auxiliary=(value-1).square().mean()
allparams=tuple(shared.parameters())+tuple(own.parameters())
task_grad=torch.autograd.grad(task,allparams,retain_graph=True)
aux_grad=torch.autograd.grad(auxiliary,tuple(own.parameters()),retain_graph=True)
routed_backward(task,auxiliary,own.parameters())
for p,g in zip(shared.parameters(),task_grad[:2]): assert torch.allclose(p.grad,g)
for p,g,h in zip(own.parameters(),task_grad[2:],aux_grad): assert torch.allclose(p.grad,g+h)
passed('auxiliary_private_only_gradient_routing')

# Native-resolution inference first resizes the two independent logits.
z=torch.randn(1,4,2,2); raw=torch.randn(1,4,3,2,2)
out={'base_logits':z,'action_logits':raw}
prob=inference_probabilities(out,(3,5))
native_z=F.interpolate(z,(3,5),mode='bilinear',align_corners=False)
native_pi=F.interpolate(raw.flatten(1,2),(3,5),mode='bilinear',align_corners=False).reshape(1,4,3,3,5).softmax(2)
assert torch.equal(prob,native_pi[:,:,0]*native_z.sigmoid()+native_pi[:,:,1])
assert lambda_k(1500)==0 and lambda_k(4000)==.5 and lambda_k(32000)==.5 and abs(lambda_k(40000)-.1)<1e-12
assert abs(learning_rate(0)-.005e-6)<1e-15
passed('native_resolution_inference_and_schedules')
# Direct counterexamples to teacher-error alignment, and detached moving targets.
probs=torch.tensor([.2,.8,.1,.8,.5,0.,1.]).reshape(1,1,1,7).repeat(1,4,1,1)
q=torch.tensor([.9,.1,.9,.9,.9,1.,0.]).reshape(1,1,1,7).repeat(1,4,1,1)
y=torch.tensor([[[[1,0,0,0,1,1,0]]]])
m=torch.ones_like(y)
z0=torch.logit(probs).requires_grad_()
target=complementary_targets(q,q,q,y,m,z0)
assert target['regions'][0][0,0,0,0] and target['regions'][1][0,0,0,1]
assert target['regions'][2][0,0,0,2], 'Student-correct pixel must be preserved even if teachers are wrong'
assert target['unknown'][0,0,0,3], 'No correct teacher evidence: exclude action supervision'
assert target['regions'][0][0,0,0,4], 'Strict >.5: .5 is a false negative for foreground'
assert not target['action_target'].requires_grad
keep,on,off=target['action_target'].unbind(2)
transported=keep*probs+on
assert torch.allclose(transported[0,0,0,:3],torch.tensor([.9,.1,.1]),atol=1e-6)
assert torch.isfinite(target['action_target']).all()
assert torch.allclose(transported[0,0,0,5:],torch.tensor([1.,0.]))
passed('student_errors_receive_directional_targets_and_correct_student_is_preserved')

# Independent nonlinear readouts retain shared representation but separate output parameters.
head=CorrectionHead()
feat=torch.randn(1,32,2,2)
condition=torch.rand(1,4,8,8)
raw=head(feat,feat,condition)
raw[:,0].sum().backward()
assert head.readouts[0][-1].weight.grad.abs().sum()>0
assert all(part[-1].weight.grad is None or part[-1].weight.grad.abs().sum()==0 for part in head.readouts[1:])
passed('per_class_output_parameters_do_not_receive_other_class_readout_loss')

# Packed kernels preserve the original four-head computation and gradients.
head=CorrectionHead().double()
with torch.no_grad():
    for part in head.readouts:part[-1].weight.normal_(0,.1)
feat=torch.randn(2,32,3,4,dtype=torch.float64,requires_grad=True)
condition=torch.rand(2,4,12,16,dtype=torch.float64)
features=torch.cat((feat,feat,F.pixel_unshuffle(condition,4)),dim=1)
reference=torch.stack([F.pixel_shuffle(part(features),4) for part in head.readouts],dim=1)
packed=head(feat,feat,condition)
assert torch.allclose(reference,packed,atol=1e-12,rtol=1e-12)
params=tuple(head.parameters())+(feat,)
ra=torch.autograd.grad(reference.square().mean(),params,retain_graph=True)
pa=torch.autograd.grad(packed.square().mean(),params)
assert all(torch.allclose(a,b,atol=1e-12,rtol=1e-10) for a,b in zip(ra,pa))
passed('packed_class_kernels_match_separate_values_and_gradients')

report={'status':'PASS','checks':checks,'torch':torch.__version__,'formal_updates':0}
(Path(__file__).resolve().parent/'CPU_SACD_PASS.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
