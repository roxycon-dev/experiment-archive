"""Frozen 20k model, real augmented train samples, teacher and logit gradients.

No optimizer is created. Gradient tests operate on detached predictions; they
measure the objective's direct output pressure, not the full SGD trajectory.
"""
import copy,json,sys,time,gc
from pathlib import Path
root=Path('/root/autodl-tmp/HRDecoder_v3_20261008')
sys.path.insert(0,str(root/'training_execution'))
sys.path.insert(0,str(root/'vendor/champion_repo'))
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from mmcv import Config
from mmseg.models import build_segmentor
from model import V3Student
from data import TrainCacheDataset
from objectives import teacher_coordinates

torch.set_num_threads(2)
torch.set_num_interop_threads(1)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
data=Path('/root/HRDecoder/data/DDR_split_613')
dataset=TrainCacheDataset(root,data)
chosen={'both_positive':[],'both_absent':[]}
for index,row in enumerate(dataset.rows):
    with Image.open(data/'labels/train'/(row['sid']+'.png')) as f: y=np.asarray(f)
    key='both_positive' if (y==1).any() and (y==3).any() else ('both_absent' if not (y==1).any() and not (y==3).any() else None)
    if key and len(chosen[key])<3:chosen[key].append(index)
    if all(len(v)==3 for v in chosen.values()):break
settings=copy.deepcopy(Config.fromfile(str(root/'vendor/champion_repo/configs/lesion/exp/baseline_split613.py')).model)
settings.pretrained=None
model=V3Student(build_segmentor(settings)).cuda().eval()
saved=torch.load(root/'reports/diagnostics/checkpoint_20000.pt',map_location='cpu')
assert saved['completed']==20000
model.load_state_dict(saved['model'],strict=True);del saved
model.requires_grad_(False)
result={'status':'RUNNING','checkpoint':20000,'device':'cuda','precision':'FP32_TF32_disabled',
        'formal_updates':0,'teacher_geometry':'actual TrainCacheDataset augmentation',
        'limitation':'Six deliberately selected training images, fixed augmentations; output gradients, not full-parameter causal replay.',
        'records':[]}
out=root/'reports/diagnostics/v4/real_objectives.json'
out.parent.mkdir(parents=True,exist_ok=True)
def write():out.write_text(json.dumps(result,indent=2)+'\n')

def mean_on(value,mask):return (value*mask).sum()/mask.sum().clamp_min(1)

for kind,indices in chosen.items():
 for index in indices:
    for seed in range(42000,42020):
        batch=dataset[(index,seed)]
        ycpu=batch['label'];v=batch['valid']>0
        if kind=='both_absent' or (((ycpu==1)&v).any() and ((ycpu==3)&v).any()):break
    label=batch['label'].unsqueeze(0).cuda();valid=batch['valid'].unsqueeze(0).cuda()
    image=batch['image'].unsqueeze(0).cuda()
    with torch.no_grad():
        pred=model(image)
        a,u,s=[batch[k].unsqueeze(0).cuda() for k in ('A','U','S')]
        teacher,du,ds=teacher_coordinates(a,u,s)
    record={'kind':kind,'sid':batch['sid'],'augmentation_seed':seed,'valid_pixels':int(valid.sum()),'classes':[]}
    for c,name in enumerate(('EX','HE','SE','MA')):
        z=pred['logits'][:,c:c+1].detach().clone().requires_grad_()
        p=z.sigmoid(); y=(label==c+1).float(); m=valid.float(); pos=m*y; neg=m*(1-y)
        q=teacher[:,c:c+1].detach()
        old=1-(2*(p*y*m).sum()+1e-5)/((p.square()+y.square())*m).sum().add(1e-5)
        kl=mean_on(F.binary_cross_entropy_with_logits(z,q,reduction='none')+(torch.xlogy(q,q)+torch.xlogy(1-q,1-q)),m)
        present=bool(pos.sum()>0)
        repaired=(old if present else z.sum()*0)+.25*(mean_on(F.softplus(-z),pos)+mean_on(F.softplus(z),neg))
        objectives={'old_dice':old,'weighted_old_KL':.5*kl,'old_main_without_hr':old+.5*kl,'repaired_GT':repaired}
        stats={}
        for k,loss in objectives.items():
            g=torch.autograd.grad(loss,z,retain_graph=True)[0]
            stats[k]={'loss':float(loss.detach()),'common_shift_gradient':float(g.sum()),
                      'positive_gradient_sum':float((g*pos).sum()),'negative_gradient_sum':float((g*neg).sum()),
                      'L1_gradient':float(g.abs().sum())}
        teachers={}
        for key,value in zip(('A','U','S','T'),(a,u,s,teacher)):
            q0=value[:,c:c+1]
            teachers[key]={'GT_positive_mean':float(mean_on(q0,pos)) if present else None,
                           'GT_negative_mean':float(mean_on(q0,neg)),
                           'GT_recall_at_half':float(mean_on((q0>.5).float(),pos)) if present else None}
        row={'class':name,'GT_positive':int(pos.sum()),'probability_max':float(p.max()),
             'positive_probability_mean':float(mean_on(p,pos)) if present else None,
             'sum_probability_squared':float((p.square()*m).sum()),'objectives':stats,'teachers':teachers}
        record['classes'].append(row)
    result['records'].append(record);write()
    print(json.dumps({'sid':record['sid'],'kind':kind,'classes':[{k:r[k] for k in ('class','GT_positive','probability_max','objectives')} for r in record['classes']]}),flush=True)
    del batch,pred,label,valid,image,a,u,s,teacher,du,ds,z,p,y,m,pos,neg,q,old,kl,repaired,objectives,loss,g
    gc.collect();torch.cuda.empty_cache()
result['status']='COMPLETE';write()
