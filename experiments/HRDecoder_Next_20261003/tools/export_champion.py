"""Export the historical champion using its original fp32 pipeline, val only."""
import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import torch
import cv2
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'legacy'))
sys.path.insert(0,str(ROOT))
from tools.configure_server import sha
from tools.probability_cache import save_probability

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--settings',required=True,type=Path)
    ap.add_argument('--out',required=True,type=Path);a=ap.parse_args()
    s=json.loads(a.settings.read_text(encoding='utf-8'))
    for rel,expected in s['expected_sha256'].items():
        p=Path(rel) if Path(rel).is_absolute() else Path(s['repo'])/rel
        if sha(p)!=expected:raise ValueError(f'Champion source hash mismatch: {p}')
    import runtime
    runtime.inject_repo(s['repo'])
    cfg=runtime.load_config(s['repo'],s['config'])
    from mmseg.datasets import build_dataset
    data=copy.deepcopy(cfg.data['val']);data.data_root=s['data_root']
    data.img_dir=str(Path(s['data_root'])/'images/val')
    data.ann_dir=str(Path(s['data_root'])/'labels/val')
    data.pipeline=copy.deepcopy(cfg.data['test']['pipeline']);data.test_mode=True
    ds=build_dataset(data)
    if len(ds)!=76:raise ValueError('Champion val must contain exactly 76 images')
    torch.set_num_threads(4)
    # Exact reference export keeps its fp32 arithmetic and deterministic
    # kernels; new training uses independently declared fast AMP kernels.
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    champ=runtime.ChampionEnsemble(s,s['champion'],device='cuda')
    a.out.mkdir(parents=True,exist_ok=True);samples=[];counts=[]
    progress_path=a.out/'progress.json'
    progress=dict(settings_sha256=sha(a.settings),images={})
    if progress_path.exists():
        progress=json.loads(progress_path.read_text(encoding='utf-8'))
        if progress['settings_sha256']!=sha(a.settings):raise ValueError('Partial champion export identity conflict')
    for i in range(len(ds)):
        sample=ds[i]
        img=runtime._unwrap_img(sample['img']).unsqueeze(0).cuda()
        meta=runtime._unwrap_meta(sample['img_metas'])
        sid=Path(meta[0].get('ori_filename',meta[0]['filename'])).stem
        if sid in progress['images']:
            saved=progress['images'][sid]
            if not (a.out/(sid+'.npz')).exists():raise ValueError('Partial champion cache file missing')
            samples.append(saved['sample']);counts.append(saved['counts']);continue
        p=champ.predict(img,meta).float().cpu().numpy()
        gt=np.asarray(Image.open(Path(s['data_root'])/'labels/val'/f'{sid}.png'))
        if gt.shape!=tuple(meta[0]['ori_shape'][:2]):
            raise ValueError('GT and original image must have identical coordinates')
        if p.shape!=(4,*gt.shape):raise ValueError('Champion geometry mismatch')
        save_probability(a.out/sid,p)
        image_counts=runtime.per_image_counts(p,gt)
        counts.append(image_counts);samples.append(dict(sample_id=sid,shape=list(p.shape)))
        progress['images'][sid]=dict(counts=image_counts,sample=samples[-1])
        temp=progress_path.with_suffix('.tmp')
        temp.write_text(json.dumps(progress,indent=2),encoding='utf-8');temp.replace(progress_path)
        print(f'champion export {i+1}/{len(ds)}',flush=True)
    total=np.asarray(counts,dtype=np.int64).sum(axis=0)
    miou,per=runtime.global_miou(total)
    metrics={'counts':total.tolist(),'mIoU_lesion4':miou,'per_class_iou':per}
    identity=hashlib.sha256(json.dumps(s['expected_sha256'],sort_keys=True).encode()).hexdigest()
    idx=dict(split='val',n_images=len(ds),samples=samples,dtype='float32',coordinates='original_image',
             channel_order=['EX','HE','SE','MA'],checkpoint_sha256=identity,
             config_sha256=sha(a.settings),champion_expected_hashes=s['expected_sha256'],metrics=metrics,
             includes_ma_head=True,flip_tta=True)
    idx['storage']='npz_deflate1_lossless'
    (a.out/'index.json').write_text(json.dumps(idx,indent=2),encoding='utf-8')
    print(json.dumps(metrics))

if __name__=='__main__':main()
