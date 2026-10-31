"""Read retained checkpoints; CPU-only counterfactual BN inference, no SGD.

Only this diagnostic process's model buffers can change. Training files and
checkpoints are never written. Batch-stat outputs are diagnostic, not val scores.
"""
import argparse
import copy
import gc
import json
import os
from pathlib import Path
import sys
import threading
import time

parser = argparse.ArgumentParser()
parser.add_argument('--root', default='/root/autodl-tmp/HRDecoder_v3_20261008')
args = parser.parse_args()
root = Path(args.root)
outdir = root/'reports/diagnostics/v4'
outdir.mkdir(parents=True, exist_ok=True)
outfile = outdir/'normalization_probe.json'

def memory_guard():
    limit_path = Path('/sys/fs/cgroup/memory.max')
    if not limit_path.exists() or limit_path.read_text().strip() == 'max': return
    limit = int(limit_path.read_text())
    while True:
        if limit-int(Path('/sys/fs/cgroup/memory.current').read_text()) < 1024**3:
            print('DIAGNOSTIC_ABORT_LOW_MEMORY', flush=True)
            os._exit(12)
        time.sleep(.5)

threading.Thread(target=memory_guard, daemon=True).start()
sys.path.insert(0, str(root/'training_execution'))
sys.path.insert(0, str(root/'vendor/champion_repo'))
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from mmcv import Config
from mmseg.models import build_segmentor
from model import V3Student
from data import evaluation_image

torch.set_num_threads(2)
torch.set_num_interop_threads(1)
data = Path('/root/HRDecoder/data/DDR_split_613')
identity = json.loads((root/'metadata/data_identity.json').read_text())
samples = []
for split in ('train', 'val'):
    selected = 0
    images = {p.stem:p for p in (data/'images'/split).iterdir()}
    for row in identity['splits'][split]['samples']:
        with Image.open(data/'labels'/split/(row['sid']+'.png')) as f: y=np.asarray(f)
        counts = [int((y==c+1).sum()) for c in range(4)]
        if counts[0] and counts[2]:
            samples.append((split,row,images[row['sid']],counts))
            selected += 1
            if selected == 2: break
report = {'status':'RUNNING','device':'CPU','dtype':'FP32','canvas':2048,
          'formal_updates':0,'dropout_enabled':False,'batch_stat_probe_batch':1,
          'limitation':'Purposefully selected positive images; not full-val or unbiased generalization estimate.',
          'records':[]}

def save():
    temp = outfile.with_suffix('.tmp')
    temp.write_text(json.dumps(report, indent=2)+'\n')
    os.replace(temp, outfile)

save()
def tensor_stats(value):
    return [{'min':float(v.min()),'mean':float(v.mean()),'max':float(v.max()),
             'positive':int((v>0).sum()),'finite':bool(torch.isfinite(v).all())}
            for v in value[0].float()]

for checkpoint_name in ('training_runs/v3/best_val.pt','reports/diagnostics/checkpoint_20000.pt'):
    settings=copy.deepcopy(Config.fromfile(str(root/'vendor/champion_repo/configs/lesion/exp/baseline_split613.py')).model)
    settings.pretrained=None
    model=V3Student(build_segmentor(settings)).cpu().eval()
    checkpoint=torch.load(root/checkpoint_name, map_location='cpu')
    step=checkpoint['completed']
    model.load_state_dict(checkpoint['model'], strict=True)
    del checkpoint
    for split,row,image_path,gtcounts in samples:
        x=evaluation_image(image_path).unsqueeze(0)
        with Image.open(data/'labels'/split/(row['sid']+'.png')) as f: label=np.asarray(f).copy()
        for mode in ('running_stats','batch_stats'):
            model.eval()
            if mode=='batch_stats':
                for module in model.modules():
                    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                        module.train()
                        module.momentum=0.0
            started=time.monotonic()
            with torch.no_grad():
                output=model(x)
                components={k:tensor_stats(output[k]) for k in ('base_logits','residual','logits')}
                pred=F.interpolate(output['logits'].float(), size=row['hw'],mode='bilinear',align_corners=False).sigmoid()[0].numpy()
                classes=[]
                for c,name in enumerate(('EX','HE','SE','MA')):
                    y=label==c+1; v=label!=255; p=pred[c]; b=(p>.5)&v
                    tp=int((b&y).sum()); fp=int((b&~y).sum()); fn=int((~b&y&v).sum())
                    classes.append({'class':name,'counts':[tp,fp,fn], 'GT_pixels':gtcounts[c],
                       'GT_probability_mean':float(p[y].mean()) if y.any() else None,
                       'GT_probability_max':float(p[y].max()) if y.any() else None,
                       'max_probability':float(p.max()),'predicted_positive':int(b.sum())})
            result={'checkpoint':step,'split':split,'sid':row['sid'],'BN_mode':mode,
                    'components':components,'classes':classes,'seconds':time.monotonic()-started}
            report['records'].append(result); save()
            print(json.dumps({'step':step,'split':split,'sid':row['sid'],'mode':mode,
                              'counts':[c['counts'] for c in classes],
                              'max_probability':[c['max_probability'] for c in classes]}),flush=True)
            del output,pred
            gc.collect()
        del x
    del model
    gc.collect()
report['status']='COMPLETE'; save()
print('PROBE_COMPLETE',flush=True)
