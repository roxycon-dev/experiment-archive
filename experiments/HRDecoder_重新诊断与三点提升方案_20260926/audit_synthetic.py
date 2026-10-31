"""CPU-only diagnostic; no training, no dataset inference, no source edits."""
import sys, json
from pathlib import Path
import torch
import torch.nn.functional as F

torch.set_num_threads(2)
torch.manual_seed(20260926)
out=Path(__file__).resolve().parent
repo=Path(r'D:\dataset\experiment\HRDecoder_FullCompetition_20260920\refiner_v1')
sys.path.insert(0,str(repo))
from refiner.model import HaarHeteroRefiner
from refiner.losses import refiner_losses

with torch.no_grad():
    x=torch.rand(1,8,64,80)
    gn=torch.nn.GroupNorm(4,8).eval()
    a=gn(x);b=gn(F.pad(x,(0,80,0,64)))[:,:,:64,:80]
    gn_result={'unpadded_shape':list(x.shape),'padded_shape':[1,8,128,160],
               'valid_region_max_abs_difference':(a-b).abs().max().item(),
               'valid_region_mean_abs_difference':(a-b).abs().mean().item()}
    ck_path=repo/'logs/formal_20260921_remote/failure_20260921_122218/stage2__last.pth'
    ck=torch.load(str(ck_path),map_location='cpu',weights_only=True)
    model=HaarHeteroRefiner().eval(); model.load_state_dict(ck['model'],strict=True)
    rgb=torch.rand(1,3,64,80);p=torch.rand(1,2,64,80)
    _,a=model.forward_refine(rgb,p)
    _,b=model.forward_refine(F.pad(rgb,(0,80,0,64)),F.pad(p,(0,80,0,64)))
    b=b[:,:,:64,:80]
    d=(a-b).abs()
    learned={'checkpoint':str(ck_path),'epoch1':ck.get('epoch1'),'step':ck.get('step'),
             'synthetic_only':True,'max_abs_difference':d.max().item(),
             'interior_mean_abs_difference':d[:,:,16:-16,16:-16].mean().item(),
             'decision_changes_at_0_5':int(((a>0.5)!=(b>0.5)).sum().item()),
             'total_channel_pixels':a.numel()}

r=torch.full((1,2,1,1),-1.0,requires_grad=True)
p=torch.full_like(r,0.01)
prediction=(p+0.5*torch.tanh(r)).clamp(0,1)
loss,_=refiner_losses(prediction,torch.ones_like(r),torch.ones_like(r))
loss.backward()
dead={'base_p':0.01,'residual_logit':-1.0,'target':1,'prediction':prediction.detach().flatten().tolist(),
      'loss':loss.item(),'gradient_wrt_residual':r.grad.flatten().tolist(),
      'scope':'demonstrates zero local gradient on wrong saturated pixels; not prevalence in real data'}

d=json.loads((repo/'audit/candidate_audit_shipping.json').read_text(encoding='utf-8'))
brief={k:v for k,v in d.items() if k not in ('per_image','images')}
results={'groupnorm_padding':gn_result,'trained_refiner_padding':learned,'wrong_clamped_pixel_gradient':dead}
(out/'synthetic_audit_results.json').write_text(json.dumps(results,indent=2,ensure_ascii=False),encoding='utf-8')
(out/'candidate_audit_summary.json').write_text(json.dumps(brief,indent=2,ensure_ascii=False),encoding='utf-8')
print(json.dumps(results,indent=2,ensure_ascii=False))
print('candidate_audit_keys',list(d))
