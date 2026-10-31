"""Export one locked candidate's val probabilities, float32 in original coords."""
import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from trainer.config import load_yaml,ddr_data_config
from trainer.models import build_model
from trainer.checkpoint import load_checkpoint
from trainer.data.ddr_dataset import DDRSegDataset
from trainer.data.common import normalize_image
from trainer.eval import pad_normalized_image,sliding_window_logits
from trainer.metrics import SegMetricsAccumulator
from tools.configure_server import sha
from tools.probability_cache import save_probability

def export(cfg_path,checkpoint,out,flip=False):
    cfg=load_yaml(cfg_path);dc=ddr_data_config(cfg)
    fast=bool(cfg.get('execution_fast_kernels'))
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=fast;torch.backends.cudnn.deterministic=not fast
    torch.backends.cuda.matmul.allow_tf32=fast;torch.backends.cudnn.allow_tf32=fast
    source=load_checkpoint(checkpoint)
    if source['is_calibration'] or source['stage']!='ddr_finetune':raise ValueError('Formal DDR checkpoint required')
    if source['family']!=cfg['structure']['family']:raise ValueError('Model family mismatch')
    model_cfg=copy.deepcopy(cfg);model_cfg['initialization']['external_pretrain_checkpoint']=None
    model_cfg['initialization']['ddr_warm_start_checkpoint']=None
    model,meta=build_model(model_cfg,stage='ddr_finetune',gradient_checkpointing=False)
    model.load_state_dict(source['model_state'],strict=True);del source
    model.to('cuda' if torch.cuda.is_available() else 'cpu').eval()
    device=next(model.parameters()).device
    ds=DDRSegDataset(dc['ddr_root'],'val',mean=dc['mean'],std=dc['std'])
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    acc=SegMetricsAccumulator();samples=[]
    geometry=cfg['metrics'].get('resize_to')
    with torch.no_grad():
        for i in range(len(ds)):
            rgb,target,valid,sid=ds.load_raw(i);orig=rgb.shape[:2]
            if geometry:rgb=cv2.resize(rgb,(geometry[1],geometry[0]),interpolation=cv2.INTER_LINEAR)
            chw=normalize_image(rgb,dc['mean'],dc['std']).transpose(2,0,1)
            x,hw=pad_normalized_image(np.ascontiguousarray(chw));x=torch.from_numpy(x[None]).to(device)
            def predict(t):
                if cfg['metrics'].get('eval_mode')=='sliding_window':
                    l=sliding_window_logits(model,t,cfg['metrics']['window_size'],cfg['metrics']['window_stride'])
                else:l=model.eval_forward(t)
                l=l[...,:hw[0],:hw[1]]
                return torch.sigmoid(F.interpolate(l.float(),size=orig,mode='bilinear',align_corners=False))
            p=predict(x)
            if flip:p=(p+predict(x.flip(-1)).flip(-1))*.5
            p=p[0].float().cpu().numpy();save_probability(out/sid,p)
            acc.update(p,target,valid);samples.append(dict(sample_id=sid,shape=list(p.shape)))
            if (i+1)%10==0:print(f'export {i+1}/{len(ds)}',flush=True)
    report=acc.compute();index=dict(split='val',n_images=len(ds),samples=samples,dtype='float32',
        coordinates='original_image',channel_order=['EX','HE','SE','MA'],checkpoint_sha256=sha(checkpoint),
        config_sha256=sha(cfg_path),flip_tta=flip,metrics=report)
    index.update(storage='npz_deflate1_lossless',fast_kernels=fast,resize_to=geometry)
    (out/'index.json').write_text(json.dumps(index,indent=2),encoding='utf-8')
    return report

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True)
    ap.add_argument('--checkpoint',required=True);ap.add_argument('--out',required=True)
    ap.add_argument('--flip',action='store_true');a=ap.parse_args()
    print(json.dumps(export(a.config,a.checkpoint,a.out,a.flip),indent=2))

if __name__=='__main__':main()
