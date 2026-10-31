"""One frozen external branch cache per arriving weight, with validation canary first."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
from PIL import Image
import torch

ROOT=Path(__file__).resolve().parents[1];EXP=ROOT.parent


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()


def write(p,value):
    t=p.with_suffix('.tmp');t.write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8');os.replace(t,p)


def make_cohort():
    original=json.loads((ROOT/'evidence/external_holdout_manifest.json').read_text(encoding='utf-8'))
    for r in original['samples']:
        r['client_image_path']=r['image_path'];r['image_path']=str(EXP/'data/idrid/images/testing'/(r['sid']+'.jpg'))
        r['client_polygon_path']=r['polygon_path'];r['polygon_path']=str(EXP/'data/idrid/labels/testing'/(r['sid']+'.txt'))
        for m in r['masks'].values():
            m['client_path']=m['path'];normalized=m['path'].replace('\\','/')
            m['path']=str(EXP/normalized[normalized.index('HRDecoder_Q_20261003/'):])
    return original


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--arm',choices=['S','C','U'],required=True);args=parser.parse_args()
    arm=args.arm;cache=ROOT/'results/remote_probability_cache'/arm;cache.mkdir(parents=True,exist_ok=True)
    if (cache/'completion.json').exists():raise RuntimeError('Branch already complete; do not repeat inference')
    cohort=make_cohort();write(cache/'cohort.json',cohort)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    paths={'S':EXP/'HRDecoder_ArchitectureDesign_20261008/cache_execution/inference_weights/S20_ema_inference.pth',
           'C':EXP/'HRDecoder_ArchitectureDesign_20261008/cache_execution/inference_weights/C20_ema_inference.pth',
           'U':EXP/'HRDecoder_ArchitectureDesign_20261008/cache_execution/inference_weights/U0_inference.pth'}
    p=paths[arm]
    if arm in ['S','C']:
        sys.path.insert(0,str(EXP/'HRDecoder_FinalDecision_20261005/execution'))
        sys.path.insert(0,str(EXP/'HRDecoder_Q_20261003/vendor/dinov3'))
        from dinov3.hub.backbones import dinov3_vitb16
        from qlib.model import Q
        from final_model import FinalModel
        from qlib.geometry import tiles,rgb_window,center_logits
        torch.manual_seed(42);model=FinalModel(Q(dinov3_vitb16(pretrained=False)),arm,42,backend='reference')
        saved=torch.load(p,map_location='cpu',weights_only=False,mmap=True)
        assert saved.get('completed',saved.get('source_completed'))==20000
        model.load_state_dict(saved['ema'],strict=True);del saved
    else:
        torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
        sys.path.insert(0,str(EXP/'HRDecoder_Ready_20261002'))
        from trainer.models.convnext_unet import ConvNeXtUNet
        from trainer.eval import pad_normalized_image
        from trainer.data.common import normalize_image
        model=ConvNeXtUNet();saved=torch.load(str(p),map_location='cpu')
        model.load_state_dict(saved['model_state'],strict=True);del saved
    model=model.cuda().eval().requires_grad_(False)
    write(cache/'manifest.json',{'arm':arm,'weight_path':str(p),'weight_sha256':sha(p),'script_sha256':sha(Path(__file__)),
        'remote_amendment_sha256':sha(ROOT/'plan/remote_runtime_amendment.md'),'torch':torch.__version__,'cuda':torch.version.cuda,
        'GPU':torch.cuda.get_device_name(),'registered_parameters':sum(x.numel() for x in model.parameters()),
        'precision':'BF16 forward FP32 probabilities/reference sampling' if arm!='U' else 'native-whole FP32',
        'external_sample_ids':[r['sid'] for r in cohort['samples']],'threshold_tuning':False})
    canary=EXP/'data/canary/007-1774-100.jpg'
    rows=[{'sid':'canary_007-1774-100','image_path':str(canary),'image_sha256':sha(canary)}]+cohort['samples']
    started=time.perf_counter()
    for i,row in enumerate(rows):
        sid=row['sid'];assert sha(row['image_path'])==row['image_sha256']
        rgb=np.asarray(Image.open(row['image_path']).convert('RGB'));h,w=rgb.shape[:2]
        if 'hw' in row:assert [h,w]==row['hw']
        torch.cuda.reset_peak_memory_stats();t=time.perf_counter()
        with torch.no_grad():
            if arm in ['S','C']:
                q=np.empty((4,h,w),np.float32)
                for tile in tiles(h,w):
                    x=torch.from_numpy(rgb_window(rgb,tile))[None].cuda()
                    with torch.autocast('cuda',dtype=torch.bfloat16):z=center_logits(model(x))
                    q[:,tile.y:tile.y+tile.height,tile.x:tile.x+tile.width]=z.float().sigmoid()[0,:,:tile.height,:tile.width].cpu().numpy()
                del x,z
            else:
                chw=normalize_image(rgb,[123.675,116.28,103.53],[58.395,57.12,57.375]).transpose(2,0,1)
                padded,_=pad_normalized_image(np.ascontiguousarray(chw));x=torch.from_numpy(padded[None]).cuda().float()
                q=model.eval_forward(x)[...,:h,:w].float().sigmoid()[0].cpu().numpy();del x,chw,padded
        assert q.dtype==np.float32 and q.shape==(4,h,w) and np.isfinite(q).all()
        target=cache/(sid+'.npy');np.save(target,q,allow_pickle=False)
        result={'sid':sid,'arm':arm,'hw':[h,w],'probability_sha256':sha(target),'image_sha256':row['image_sha256'],
                'seconds':time.perf_counter()-t,'GPU_peak_allocated':torch.cuda.max_memory_allocated()}
        if i==0:
            gt=np.asarray(Image.open(EXP/'data/canary/007-1774-100.png'));g=np.stack([gt==c for c in range(1,5)]);b=q>.5;v=gt!=255
            result['validation_canary_counts']=np.stack([(b&g&v).sum((1,2)),(b&~g&v).sum((1,2)),(~b&g&v).sum((1,2))],axis=1).tolist()
            result['external_predictions_seen_before_canary']=False
        write(cache/(sid+'.json'),result)
        del q,rgb;torch.cuda.empty_cache()
        progress={'arm':arm,'completed':max(i,0),'expected_external':27,'last_sid':sid,'elapsed_seconds':time.perf_counter()-started}
        write(cache/'progress.json',progress);print(json.dumps(progress),flush=True)
    write(cache/'completion.json',{'status':'COMPLETE','external_images':27,'validation_canary_images':1,'seconds':time.perf_counter()-started})


if __name__=='__main__':main()
