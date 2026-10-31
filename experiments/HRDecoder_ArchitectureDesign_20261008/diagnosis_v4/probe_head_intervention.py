"""Frozen-checkpoint interventions in memory, no optimizer or checkpoint writes."""
import copy,json,sys,gc
from pathlib import Path
root=Path('/root/autodl-tmp/HRDecoder_v3_20261008')
sys.path.insert(0,str(root/'training_execution'));sys.path.insert(0,str(root/'vendor/champion_repo'))
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from mmcv import Config
from mmseg.models import build_segmentor
from model import V3Student
from data import evaluation_image
torch.set_num_threads(2)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
settings=copy.deepcopy(Config.fromfile(str(root/'vendor/champion_repo/configs/lesion/exp/baseline_split613.py')).model)
settings.pretrained=None
model=V3Student(build_segmentor(settings)).cuda().eval().requires_grad_(False)
best=torch.load(root/'training_runs/v3/best_val.pt',map_location='cpu')['model']
old={k:v for k,v in best.items() if k.startswith('base.decode_head.')};del best
saved=torch.load(root/'reports/diagnostics/checkpoint_20000.pt',map_location='cpu')
state=saved['model'];del saved
model.load_state_dict(state,strict=True)
rows=[]
for c,name in enumerate(('EX','HE','SE','MA')):
    a=old['base.decode_head.conv_seg.weight'][c].flatten();b=state['base.decode_head.conv_seg.weight'][c].flatten()
    rows.append({'class':name,'cosine_8k_20k':float(F.cosine_similarity(a,b,dim=0)),
                 'mean_weight_8k':float(a.mean()),'mean_weight_20k':float(b.mean())})
data=Path('/root/HRDecoder/data/DDR_split_613')
samples=[('train','007-3396-200'),('train','007-3448-200'),('val','007-3873-200'),('val','007-4960-300')]
report={'formal_updates':0,'checkpoint':20000,'classifier_row_comparison':rows,'records':[],
        'limitation':'Hybrid checkpoints are mechanistic interventions only; no full-val accuracy claim.'}
for mode in ('original_20k','EX_SE_classifier_rows_from_8k','whole_decoder_from_8k'):
    model.load_state_dict(state,strict=True)
    if mode=='EX_SE_classifier_rows_from_8k':
        with torch.no_grad():
            for name in ('weight','bias'):
                target=getattr(model.base.decode_head.conv_seg,name)
                source=old['base.decode_head.conv_seg.'+name].to(target)
                target[[0,2]]=source[[0,2]]
    if mode=='whole_decoder_from_8k':model.base.decode_head.load_state_dict({k[len('base.decode_head.'):]:v for k,v in old.items()},strict=True)
    for split,sid in samples:
        path=next(p for p in (data/'images'/split).iterdir() if p.stem==sid)
        with Image.open(data/'labels'/split/(sid+'.png')) as f:label=np.asarray(f).copy()
        image=evaluation_image(path).unsqueeze(0).cuda()
        with torch.no_grad():
            output=model(image)
            p=F.interpolate(output['logits'],size=label.shape,mode='bilinear',align_corners=False).sigmoid()[0].cpu().numpy()
        classes=[]
        for c,name in enumerate(('EX','HE','SE','MA')):
            y=label==c+1;b=(p[c]>.5)&(label!=255)
            classes.append({'class':name,'max_probability':float(p[c].max()),'counts':[int((y&b).sum()),int((~y&b).sum()),int((y&~b).sum())]})
        row={'mode':mode,'split':split,'sid':sid,'classes':classes};report['records'].append(row)
        print(json.dumps(row),flush=True)
        del image,output,p;gc.collect()
out=root/'reports/diagnostics/v4/head_intervention.json'
out.write_text(json.dumps(report,indent=2)+'\n')
