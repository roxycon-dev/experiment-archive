"""Local CPU regression checks on actual private graph; no dataset, weights or GPU."""
import copy
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys
import torch
from torch import nn
import torch.nn.functional as F

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(ROOT/'residual_execution'))
sys.path.insert(0,str(ROOT/'normalization_execution'))
from model import SACDStudent,inference_probabilities
from fast_objectives import gt_loss,residual_gt,trusted_kl
from objectives import fixed_teacher
from intervention import balanced_kl
from diagnostic_invariants import (FrozenProviderStudent,named_private_optimizer_state,
    restore_immutable_parent,assert_frozen_provider,provider_snapshot,assert_provider_unchanged)

torch.set_num_threads(4)
torch.use_deterministic_algorithms(True)
torch.manual_seed(42)
checks=[]
sources=[HERE/'diagnostic_invariants.py',HERE/'verify_diagnostic_invariants.py']
sources.extend(ROOT/'residual_execution'/name for name in
               ('model.py','private_modules.py','stable_objectives.py','fast_objectives.py','objectives.py'))
sources.append(ROOT/'normalization_execution'/'intervention.py')
source_sha={path.relative_to(ROOT).as_posix():hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
report_path=HERE/'DIAGNOSTIC_INVARIANTS_CPU_PASS.json'
report_path.write_text(json.dumps({'status':'RUNNING_NOT_PASS','source_SHA256':source_sha})+'\n',encoding='utf-8')


class FakeHead(nn.Module):
    def __init__(self):
        super().__init__();self.drop=nn.Dropout2d(.1);self.conv=nn.Conv2d(720,4,1)
    def forward(self,value):return self.conv(self.drop(value[0]))


class FakeBase(nn.Module):
    def __init__(self):
        super().__init__();self.hr_scale=(32,32);self.decode_head=FakeHead()
        self.layers=nn.ModuleList([nn.Sequential(nn.Conv2d(3,c,1),nn.BatchNorm2d(c)) for c in (48,96,192,384)])
    def extract_feat(self,image):
        return tuple(layer(F.adaptive_avg_pool2d(image,(8//2**i,8//2**i))) for i,layer in enumerate(self.layers))
    def _transform_inputs(self,features):
        return torch.cat([F.interpolate(f,size=(8,8),mode='bilinear',align_corners=False) for f in features],1)
    def hr_slide_inference(self,size,fused,unused):
        logits=F.interpolate(self.decode_head([fused]),size=size,mode='bilinear',align_corners=False)
        return {'logits':[logits],'boxes':[(0,size[0],0,size[1])]}
    def _fuse_logits(self,a,b):return .5*(a+b)


def step(model,optimizer,batch,kl):
    image,label,valid,teacher=batch
    optimizer.zero_grad(set_to_none=True)
    # The real private graph can be exercised locally without mmseg's HRcrop.
    # Only the frozen diagnostic wrapper accepts labels to test its explicit
    # image-only provider override; the seed fixture uses the ordinary path.
    output=model(image,label,valid) if isinstance(model,FrozenProviderStudent) else model(image)
    loss=gt_loss(output['base_logits'],label,valid)+residual_gt(output['logits'],label,valid)
    loss=loss+.5*kl(output['logits'],fixed_teacher(teacher,teacher,teacher,label),label,valid)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    optimizer.step()
    return output


# Reproduce an unsafe same-device optimizer reload without touching real experiments.
p=nn.Parameter(torch.tensor([1.]))
unsafe=torch.optim.SGD([p],lr=.01,momentum=.9)
p.grad=torch.ones_like(p);unsafe.step()
retained=copy.deepcopy(unsafe.state_dict());before=retained['state'][0]['momentum_buffer'].clone()
unsafe.load_state_dict(retained);p.grad=torch.ones_like(p);unsafe.step()
unsafe_alias_observed=not torch.equal(before,retained['state'][0]['momentum_buffer'])
# load_state_dict alias behavior differs by torch version. Record it; the
# required invariant is that our retained parent survives every restore/update.
checks.append('CPU_optimizer_alias_behavior_recorded_for_exact_torch_version')

# Freeze-once fails when the existing evaluation flow restores model.train().
unsafe_mode=SACDStudent(FakeBase(),canvas=64)
unsafe_mode.base.eval();unsafe_mode.eval();unsafe_mode.train(True)
assert unsafe_mode.base.training
checks.append('freeze_once_train_transition_negative_control_reproduced')
del unsafe_mode

seed_model=SACDStudent(FakeBase(),canvas=64)
seed_optimizer=torch.optim.SGD(seed_model.parameters(),lr=.004,momentum=.9,weight_decay=.0005,foreach=True)
g=torch.Generator().manual_seed(61008)
batches=[]
for index in range(19):
    image=torch.randn(2,3,64,64,generator=g)
    label=torch.randint(0,5,(2,1,64,64),generator=g);valid=torch.ones_like(label)
    if index%3==0:valid[:,:,-4:]=0
    target=torch.stack([(label[:,0]==c) for c in range(1,5)],1).float()
    teacher=.1+.75*target
    batches.append((image,label,valid,teacher))
for batch in batches[:3]:step(seed_model,seed_optimizer,batch,trusted_kl)
parent_model=copy.deepcopy(seed_model.state_dict())
parent_optimizer=copy.deepcopy(seed_optimizer.state_dict())
source_names=[name for name,_ in seed_model.named_parameters()]
assert parent_optimizer['state']
retained_parent=copy.deepcopy(parent_optimizer)
retained_model=copy.deepcopy(parent_model)
del seed_optimizer,seed_model

arms={}
reference_provider=None
for arm,kl in [('control',trusted_kl),('gt_stratified',balanced_kl)]:
    student=FrozenProviderStudent(FakeBase(),canvas=64)
    optimizer=torch.optim.SGD(student.diagnostic_parameters(),lr=.004,momentum=.9,weight_decay=.0005,foreach=True)
    filtered=named_private_optimizer_state(parent_optimizer,source_names,student)
    if arm=='control':
        try:named_private_optimizer_state(parent_optimizer,list(reversed(source_names)),student)
        except AssertionError:pass
        else:raise AssertionError('Unmatched parent parameter order was accepted')
        wrong_shape=copy.deepcopy(parent_optimizer)
        slot=wrong_shape['param_groups'][0]['params'][source_names.index('private.'+next(iter(dict(student.private.named_parameters()))))]
        wrong_shape['state'][slot]['momentum_buffer']=torch.zeros(1)
        try:named_private_optimizer_state(wrong_shape,source_names,student)
        except AssertionError:pass
        else:raise AssertionError('Invalid private momentum shape was accepted')
        del wrong_shape
    filtered_reference=copy.deepcopy(filtered)
    restore_immutable_parent(student,optimizer,parent_model,filtered)
    assert_frozen_provider(student,optimizer)
    before_provider=provider_snapshot(student)
    # Discarded smoke and two restores exercise momentum alias protection.
    step(student,optimizer,batches[3],kl)
    restore_immutable_parent(student,optimizer,parent_model,filtered)
    for slot,values in filtered['state'].items():
        assert torch.equal(values['momentum_buffer'],filtered_reference['state'][slot]['momentum_buffer'])
    private_initial={n:v.clone() for n,v in student.private.state_dict().items()}
    provider_outputs=[]
    for index,batch in enumerate(batches[3:]):
        # The real evaluate() flow toggles the wrapper, including before first update.
        if index in (0,7,15):
            was=student.training;student.eval()
            with torch.no_grad():inference_probabilities(student(batch[0]),(31,37))
            student.train(was)
        assert_frozen_provider(student,optimizer)
        output=step(student,optimizer,batch,kl)
        assert_provider_unchanged(student,before_provider)
        assert all(p.grad is None for p in student.base.parameters())
        provider_outputs.append(output['base_logits'].detach().clone())
    assert any(not torch.equal(private_initial[n],v) for n,v in student.private.state_dict().items())
    if reference_provider is None:reference_provider=provider_outputs
    else:assert all(torch.equal(a,b) for a,b in zip(reference_provider,provider_outputs))
    arms[arm]={'provider_outputs_bitwise_equal_at_all16_steps':True,
      'provider_parameters_and_BN_unchanged':True,'private_updates_nonzero':True,
      'mode_transitions_checked':[0,7,15]}
    private_final={n:v.clone() for n,v in student.private.state_dict().items()}
    if arm=='control':reference_private=private_final
    else:assert any(not torch.equal(reference_private[n],v) for n,v in private_final.items())
    del student,optimizer

for slot,values in parent_optimizer['state'].items():
    assert torch.equal(values['momentum_buffer'],retained_parent['state'][slot]['momentum_buffer'])
assert all(torch.equal(value,retained_model[name]) for name,value in parent_model.items())
checks.extend(['safe_parent_optimizer_snapshots_immutable_across_discarded_smoke',
 'explicit_named_private_optimizer_mapping_preserves_parent_momentum',
 'actual_private_graph_updates_with_frozen_provider',
 'provider_parameters_BN_and_outputs_unchanged_across16_updates_and_eval_train_cycles',
 'paired_private_interventions_have_exact_provider_outputs_and_different_private_updates'])
checks.append('frozen_provider_forward_has_no_random_HRcrop_even_with_training_label_arguments')
checks.append('unmatched_parameter_order_and_private_momentum_shape_rejected')
report={'status':'CPU_INVARIANT_PASS_GPU_PENDING','torch':torch.__version__,
 'verified_at_UTC':datetime.now(timezone.utc).isoformat(),
 'checks':checks,'arms':arms,'device':'cpu','fixture_canvas':64,'batch':2,
 'fixture_private_parameters':6415232,'toy_updates_per_arm':16,'formal_updates':0,
 'unsafe_same_device_optimizer_alias_observed':unsafe_alias_observed,
 'past_GPU_divergence_root_cause_proven':False,'server_access':False,
 'dataset_or_test_or_checkpoint_read':False,'full2048_GPU_verification_completed':False,
 'real_parent_RNG_replay_verified':False,
 'fresh_architecture_or_accuracy_verified':False,'goal_achieved':False,
 'source_SHA256':source_sha}
report_path.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print(json.dumps(report))
