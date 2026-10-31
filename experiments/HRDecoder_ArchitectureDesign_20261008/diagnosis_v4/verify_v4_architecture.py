"""CPU structure, arithmetic, initial decision and gradient-route checks; no SGD."""
import json,sys
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
from v4_architecture import V4Private,WEIGHTS,routed_backward,inference_probabilities
from v4_objectives import gt_loss,trusted_kl,mixture_logits
torch.set_num_threads(1);torch.manual_seed(20261009)
here=Path(__file__).resolve().parent
module=V4Private()
parameters=sum(p.numel() for p in module.parameters());assert parameters==6382912
rgb=torch.randn(2,3,128,128)
features=[torch.randn(2,c,128//s,128//s,requires_grad=True) for c,s in zip((48,96,192,384),(8,16,32,64))]
mac=[0]
def hook(m,x,y):
    if isinstance(m,nn.Conv2d):mac[0]+=y.numel()*m.in_channels//m.groups*m.kernel_size[0]*m.kernel_size[1]
    elif isinstance(m,nn.Linear):mac[0]+=y.numel()*m.in_features
handles=[m.register_forward_hook(hook) for m in module.modules() if isinstance(m,(nn.Conv2d,nn.Linear))]
u,s=module(rgb,features)
for h in handles:h.remove()
conv_linear_full=mac[0]//2*(2048//128)**2
attn_full=4*2*(32*32)**2*256
za=torch.randn_like(u);zm=mixture_logits((za,u,s))
assert torch.count_nonzero(u)==torch.count_nonzero(s)==0
assert torch.equal(zm>0,za>0)
assert torch.allclose(zm.sigmoid(),.5625*za.sigmoid()+.21875,atol=1e-6)
native=inference_probabilities({'member_logits':(za,u,s)},(119,131))
anchor_native=F.interpolate(za,size=(119,131),mode='bilinear',align_corners=False).sigmoid()
assert torch.equal(native>.5,anchor_native>.5)
# Nonzero readouts in this diagnostic copy expose the mature gradient paths.
with torch.no_grad():
    module.head_u.output.weight.normal_(0,.02);module.head_s.output.weight.normal_(0,.02)
u,s=module(rgb,features)
zanchor=F.interpolate(features[0][:,:4],size=(128,128),mode='bilinear',align_corners=False)
zm=mixture_logits((zanchor,u,s));label=torch.randint(0,5,(2,1,128,128));valid=torch.ones_like(label).float()
t=torch.rand_like(zm);task=gt_loss(zm,label,valid)+.25*gt_loss(zanchor,label,valid)
aux=.25*(trusted_kl(u,t,label,valid)+trusted_kl(s,t,label,valid))
chosen=(features[0],module.local.p_r.conv.weight,module.global_path.p_f34.conv.weight,module.head_u.output.weight)
tg=torch.autograd.grad(task,chosen,retain_graph=True)
ag=torch.autograd.grad(aux,chosen,retain_graph=True)
assert ag[0].abs().sum()>0  # The excluded auxiliary shared path actually exists.
routed_backward(task,aux,module.parameters())
errors=[float((chosen[i].grad-(tg[i] if i==0 else tg[i]+ag[i])).abs().max()) for i in range(len(chosen))]
assert max(errors)<1e-6
assert tg[0].abs().sum()>0
report={'status':'PASS','device':'CPU','formal_updates':0,'canvas_probe':128,'batch_probe':2,
        'new_parameters':parameters,'new_parameters_groups':{n:sum(p.numel() for p in m.parameters()) for n,m in module.named_children()},
        'new_MAC_2048_batch1':{'conv_linear':conv_linear_full,'attention_QK_AV':attn_full,'total':conv_linear_full+attn_full},
        'initial_threshold_decisions_match_anchor':True,'initial_probabilities_match_anchor':False,
        'native_resolution_mixture_checked':True,'component_aux_excludes_shared_gradient':True,
        'raw_component_aux_shared_gradient_nonzero':True,'task_shared_gradient_nonzero':True,
        'gradient_max_errors':errors,'diagnostic_readout_randomization_not_saved':True,
        'limitations':'Module probes use random feature tensors; no performance claim or end-to-end training validation.'}
(here/'evidence/v4_architecture_checks.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
