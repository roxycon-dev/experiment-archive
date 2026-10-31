"""A separate fresh private-expert candidate, registered before any candidate data."""
import json
from pathlib import Path
import py_compile

ROOT=Path(__file__).resolve().parent.parent
OLD=ROOT/'residual_execution';NEW=ROOT/'expert_execution'
NEW.mkdir(exist_ok=True)
assert not (NEW/'SEALED_LOCAL_SOURCE.json').exists()
for name in ('baseline.py','private_modules.py','stable_objectives.py','data.py','loader.py'):
    (NEW/name).write_bytes((OLD/name).read_bytes())
s=(OLD/'model.py').read_text(encoding='utf-8')
s=s.replace('        self.head = CorrectionHead()', '        self.head = CorrectionHead()\n        self.fusion_logits = nn.Parameter(torch.zeros(4))')
s=s.replace('== 6415232','== 6415236')
s=s.replace('        residual = self.private(image, features, z0)','        private_logits = self.private(image, features, z0)')
s=s.replace('        logits = z0.float().detach() + residual.float()',
'''        alpha = self.private.fusion_logits.float().sigmoid().view(1,4,1,1)
        logits = alpha*z0.float().detach() + (1-alpha)*private_logits.float()''')
s=s.replace("'residual_logits': residual", "'private_logits': private_logits, 'fusion_alpha': alpha")
s=s.replace('Numerically this is a continuous signed logit residual, in FP32.',
            'The private expert has its own stationary segmentation target; FP32 convex logit fusion.')
(NEW/'model.py').write_text(s,encoding='utf-8')
s=(OLD/'fast_objectives.py').read_text(encoding='utf-8')
start=s.index('def residual_gt(');end=s.index('\ndef trusted_kl',start)
s=s[:start]+'''def residual_gt(logits,label,valid):
    # Compatibility name for the existing numerical/routing verification only.
    return gt_loss(logits,label,valid)
'''+s[end:]
start=s.index('    return macro((kl*m*trusted)')
s=s[:start]+'''    positive=m&y;negative=m&~y
    pair=.5*(kl*positive*trusted).sum((-2,-1))/positive.sum((-2,-1)).clamp_min(1)
    pair=pair+.5*(kl*negative*trusted).sum((-2,-1))/negative.sum((-2,-1)).clamp_min(1)
    return macro(pair,m)
'''
(NEW/'fast_objectives.py').write_text(s,encoding='utf-8')
s=(OLD/'objectives.py').read_text(encoding='utf-8')
start=s.index('def losses(');end=s.index('\ndef routed_backward',start)
s=s[:start]+'''def losses(output,label,valid,a,u,s,update):
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
'''+s[end:]
(NEW/'objectives.py').write_text(s,encoding='utf-8')
protocol=json.loads((OLD/'architecture_spec.json').read_text())
protocol.update(design_version=9,name='HRDecoder-PrivateExpert',
 GT='base original-formula Dice + .1 HRcrop; private standalone Dice + .5 fused Dice; no ordinary BCE',
 KD='private standalone fixed GT-consistent T* KL; positive/negative GT strata independently normalized at fixed .5/.5',
 readout='four independent private segmentation logits zero initialized; four learned per-class convex logit fusion weights initialized .5',
 gradient_route='features and p0 detached; final=alpha*base.float().detach()+(1-alpha)*private.float(); base supervised independently',
 initialization_parity='At zero private logits and alpha=.5, hard decisions equal base; probabilities intentionally differ',
 screen_updates=4000,validation_extra_screen_probe=2000,accuracy_guaranteed=False)
(NEW/'architecture_spec.json').write_text(json.dumps(protocol,indent=2)+'\n')
screen={'registered_before_any_new_candidate_results':True,'screen_updates':4000,
 'early_full_val_probe':2000,'original_deadline_unix':1791568800,'criteria':{
 '4k_mIoU_at_least':30.475812328118998,'gain_2k_to4k_pp_at_least':5.,
 'final_minus_base_4k_pp_at_least':1.,'max_class_IoU_loss_final_vs_base_pp':2.,
 'HE_MA_reject_if_FN_growth_over':1.1,'HE_MA_requires_FP_ratio_when_FN_grows_at_most':.9},
 'registered_basis':'Distinct architecture;4k floor is previous residual4k+1pp. Extra2k provides measured within-run growth. Earlier rejected8k gate remains immutable.',
 'interpretation':'Practical shortscreen only; not accuracy guarantee or held-out test selection',
 'test_access_for_screening':False,'reject_next_action':'autonomous evidence-based revision within remaining authorized time'}
(NEW/'screen_protocol.json').write_text(json.dumps(screen,indent=2)+'\n')
s=(OLD/'train.py').read_text(encoding='utf-8').replace('residual','expert')
start=s.index('PROTOCOL = ');end=s.index('\ndef write_json',start)
s=s[:start]+'PROTOCOL = '+repr(protocol)+'\n'+s[end:]
s=s.replace("'CPU_RESIDUAL_PASS.json'","'CPU_EXPERT_PASS.json'")
s=s.replace('71987652','71987656').replace('6415232','6415236')
s=s.replace('if completed%4000==0:', 'if completed%4000==0 or completed==2000:')
s=s.replace('if completed == 8000:', 'if completed == 4000:')
s=s.replace("screen_checkpoint['completed']==8000", "screen_checkpoint['completed']==4000")
s=s.replace("if completed == 8000 and", "if completed == 4000 and")
s=s.replace("validation['completed']=completed", "validation['completed']=completed\n                validation['validation_role']='screen_probe' if completed==2000 else 'standard4k_interval'")
(NEW/'train.py').write_text(s,encoding='utf-8')
s=(OLD/'test_eval.py').read_text(encoding='utf-8').replace('residual','expert')
(NEW/'test_eval.py').write_text(s,encoding='utf-8')
s=(OLD/'verify_residual.py').read_text(encoding='utf-8')
s=s.replace('6415232','6415236').replace('71987652','71987656').replace('CPU_RESIDUAL_PASS','CPU_EXPERT_PASS')
s=s.replace("student=SACDStudent(FakeBase(),canvas=64).double().eval()", "student=SACDStudent(FakeBase(),canvas=64).double().eval()")
needle='with torch.no_grad():\n    for h in student.private.head.readouts:h[-1].weight.normal_(0,.03)'
assert needle in s
s=s.replace(needle,'''initial=student(image)
assert torch.count_nonzero(initial['private_logits'])==0
assert torch.all(initial['fusion_alpha']==.5)
assert torch.equal(initial['logits']>0,initial['base_logits']>0)
assert not torch.equal(initial['logits'].sigmoid(),initial['base_logits'].float().sigmoid())
passed('independent_zero_expert_half_gate_initial_hard_decisions_preserved_not_probabilities')
with torch.no_grad():
    for h in student.private.head.readouts:h[-1].weight.normal_(0,.03)''')
needle="passed('fixed_detached_teacher_targets_and_KL_endpoints')"
assert needle in s
s=s.replace(needle,needle+'''
# Background replication must not dilute correct rare-foreground teacher loss.
zz=torch.full((1,4,1,2),-.5,dtype=torch.float64,requires_grad=True)
yy=torch.tensor([[[[1,0]]]]);mm=torch.ones_like(yy)
qq=torch.full_like(zz,.1);qq[0,0,0,0]=.9
small=trusted_kl(zz,qq,yy,mm)
large_z=torch.cat((zz[:,:,:,:1],zz[:,:,:,1:].repeat(1,1,1,19)),3)
large_y=torch.cat((yy[:,:,:,:1],yy[:,:,:,1:].repeat(1,1,1,19)),3)
large_q=torch.cat((qq[:,:,:,:1],qq[:,:,:,1:].repeat(1,1,1,19)),3)
large=trusted_kl(large_z,large_q,large_y,torch.ones_like(large_y))
assert torch.allclose(small,large,atol=1e-12)
assert torch.allclose(torch.autograd.grad(small,zz,retain_graph=True)[0],torch.autograd.grad(large,zz)[0],atol=1e-12)
passed('GT_stratum_KL_invariant_to_easy_background_replication_value_and_gradient')
''')
s=s.replace('packed_heads', 'packed_heads')
(NEW/'verify_expert.py').write_text(s,encoding='utf-8')
for path in NEW.glob('*.py'):py_compile.compile(str(path),doraise=True)
print(json.dumps({'status':'PREPARED','path':str(NEW),'formal_updates':0,'new_private_parameters':6415236}))
