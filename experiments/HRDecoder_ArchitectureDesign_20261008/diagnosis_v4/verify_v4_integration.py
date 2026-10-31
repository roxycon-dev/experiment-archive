"""Actual ImageNet HRNet-W48 integration at small CPU canvas, no optimizer."""
import json,sys,os
from pathlib import Path
import torch
if os.name=='nt':
    here=Path(__file__).resolve().parent;root=here.parent;work=root.parent
    repo=work/'HRDecoder_Ready_20261002/vendor/champion_repo'
    pretrained=work/'HRDecoder_Ready_20261002/assets/hrnetv2_w48-d2186c55.pth'
else:
    root=Path('/root/autodl-tmp/HRDecoder_v3_20261008');here=root/'diagnosis_v4'
    repo=root/'vendor/champion_repo';pretrained=Path('/root/.cache/torch/hub/checkpoints/hrnetv2_w48-d2186c55.pth')
sys.path.insert(0,str(root/'training_execution/python_deps'))
sys.path.insert(0,str(root/'training_execution'));sys.path.insert(0,str(here))
from model import build_baseline,tensor_state_sha
from v4_architecture import V4Student,losses,routed_backward
torch.set_num_threads(1)
base,_=build_baseline(repo,pretrained,canvas=128)
before=tensor_state_sha(base.backbone)
student=V4Student(base,canvas=128).train()
assert tensor_state_sha(student.base.backbone)==before
calls=[];hook=student.base.backbone.register_forward_hook(lambda *x:calls.append(1))
image=torch.randn(2,3,128,128);label=torch.randint(0,5,(2,1,128,128));valid=torch.ones_like(label).float()
output=student(image,label,valid);hook.remove()
assert len(calls)==1 and output['logits'].shape==(2,4,128,128)
teachers=[torch.rand_like(output['logits']) for _ in range(3)]
task,comp=losses(output,label,valid,*teachers,4000)
routed_backward(task,comp,student.component_parameters())
assert all(p.grad is None or torch.isfinite(p.grad).all() for p in student.parameters())
report={'status':'PASS','device':'CPU','canvas':128,'batch':2,'actual_ImageNet_HRNet_W48':True,
        'backbone_initialization_preserved':True,'backbone_forward_calls':len(calls),
        'task':float(task.detach()),'components':float(comp.detach()),'HR_loss':float(output['hr_loss'].detach()),
        'all_parameter_gradients_finite':True,'total_parameters':sum(p.numel() for p in student.parameters()),
        'formal_updates':0,'optimizer_created':False,'accuracy_validated':False}
out=(here/'evidence/v4_integration_checks.json') if os.name=='nt' else (root/'reports/diagnostics/v4/v4_integration_checks.json')
out.parent.mkdir(exist_ok=True,parents=True);out.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
