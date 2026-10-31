"""Actual train-mode bias gradients at frozen 20k; no parameter updates."""
import copy,json,sys,gc
from pathlib import Path
root=Path('/root/autodl-tmp/HRDecoder_v3_20261008')
sys.path.insert(0,str(root/'training_execution'));sys.path.insert(0,str(root/'vendor/champion_repo'))
import torch
import torch.nn.functional as F
import numpy as np
from mmcv import Config
from mmseg.models import build_segmentor
import model as model_module
from model import V3Student
from data import TrainCacheDataset
from objectives import masked_dice,masked_kl,teacher_coordinates
torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
dataset=TrainCacheDataset(root,Path('/root/HRDecoder/data/DDR_split_613'))
index={r['sid']:i for i,r in enumerate(dataset.rows)}
pairs=[('007-3396-200','007-1811-100'),('007-3448-200','007-1824-100'),('007-3520-200','007-1831-100')]
settings=copy.deepcopy(Config.fromfile(str(root/'vendor/champion_repo/configs/lesion/exp/baseline_split613.py')).model);settings.pretrained=None
student=V3Student(build_segmentor(settings)).cuda().requires_grad_(False)
saved=torch.load(root/'reports/diagnostics/checkpoint_20000.pt',map_location='cpu')
parameter=student.base.decode_head.conv_seg.bias
parameter.requires_grad_(True)
captured={}
def capture(z,y,v):
    captured.update(logits=z,label=y,valid=v)
    return masked_dice(z,y,v)
model_module.masked_dice=capture
def repaired(z,label,valid):
    z=z.float()
    if z.shape[-2:]!=label.shape[-2:]:z=F.interpolate(z,size=label.shape[-2:],mode='bilinear',align_corners=False)
    y=torch.cat([(label==c+1).float() for c in range(4)],1);m=valid.expand_as(y).float();p=z.sigmoid()
    pos=m*y;neg=m*(1-y);n=pos.sum((-2,-1));nn=neg.sum((-2,-1));active=m.sum((-2,-1))>0
    d=(1-(2*(p*pos).sum((-2,-1))+1e-5)/(((p.square()+y)*m).sum((-2,-1))+1e-5))*(n>0)
    b=(F.softplus(-z)*pos).sum((-2,-1))/n.clamp_min(1)+(F.softplus(z)*neg).sum((-2,-1))/nn.clamp_min(1)
    return ((d+.25*b)*active).sum()/active.sum().clamp_min(1)
report={'checkpoint':20000,'formal_updates':0,'optimizer_created':False,'mode':'train, actual BN and Dropout; frozen weights except classifier bias requires_grad',
        'limitation':'Three selected two-image batches, current gradient only; not full momentum trajectory or a recovery training experiment.',
        'records':[]}
for i,sids in enumerate(pairs):
    student.load_state_dict(saved['model'],strict=True);student.train();captured.clear()
    rows=[dataset[(index[sid],42000)] for sid in sids]
    batch={k:torch.stack([r[k] for r in rows]).cuda() for k in ('image','label','valid','A','U','S')}
    torch.manual_seed(20261009);np.random.seed(20261009)
    prediction=student(batch['image'],batch['label'],batch['valid'])
    t,_,_=teacher_coordinates(batch['A'],batch['U'],batch['S'])
    kd=.5*masked_kl(prediction['logits'],t,batch['valid'])
    oldgt=masked_dice(prediction['logits'],batch['label'],batch['valid'])+.1*prediction['hr_loss']
    newgt=repaired(prediction['logits'],batch['label'],batch['valid'])+.1*repaired(captured['logits'],captured['label'],captured['valid'])
    losses={'old_GT_including_HR':oldgt,'weighted_KD':kd,'old_task':oldgt+kd,'repaired_GT_including_HR':newgt,'GT_repaired_same_KD':newgt+kd}
    record={'sids':list(sids),'seed':42000,'gradients':{}}
    for name,loss in losses.items():
        g=torch.autograd.grad(loss,parameter,retain_graph=True)[0]
        record['gradients'][name]={'loss':float(loss.detach()),'base_classifier_bias_gradient_EX_HE_SE_MA':g.detach().cpu().tolist()}
    report['records'].append(record);print(json.dumps(record),flush=True)
    del rows,batch,prediction,t,kd,oldgt,newgt,losses,loss,g;captured.clear();gc.collect();torch.cuda.empty_cache()
(root/'reports/diagnostics/v4/classifier_gradients.json').write_text(json.dumps(report,indent=2)+'\n')
