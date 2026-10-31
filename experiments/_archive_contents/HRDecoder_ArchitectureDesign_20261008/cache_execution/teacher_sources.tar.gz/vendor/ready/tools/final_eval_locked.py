"""One resumable test evaluation of a previously locked complete pipeline.

No standalone candidate test scores and no retuning. Store counts per image,
so a crash can resume the same lock without changing or repeating completed
decisions. The champion and the selected complete pipeline are evaluated on
the same images during this single evaluation.
"""
import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'legacy'))
from tools.configure_server import sha
from trainer.config import load_yaml,ddr_data_config
from trainer.models import build_model
from trainer.checkpoint import load_checkpoint
from trainer.data.common import normalize_image
from trainer.eval import pad_normalized_image,sliding_window_logits
from tools.crossfit_fusion import iou

def atomic(path,obj):
    p=path.with_suffix('.tmp');p.write_text(json.dumps(obj,indent=2),encoding='utf-8');p.replace(path)

def kernels(fast):
    torch.backends.cudnn.benchmark=fast;torch.backends.cudnn.deterministic=not fast
    torch.backends.cuda.matmul.allow_tf32=fast;torch.backends.cudnn.allow_tf32=fast

def load_candidate(member):
    if sha(member['checkpoint'])!=member['checkpoint_sha256'] or sha(member['config'])!=member['config_sha256']:
        raise ValueError('Locked candidate identity changed')
    cfg=load_yaml(member['config']);dc=ddr_data_config(cfg);build=copy.deepcopy(cfg)
    build['initialization']['external_pretrain_checkpoint']=None
    build['initialization']['ddr_warm_start_checkpoint']=None
    model,_=build_model(build,stage='ddr_finetune',gradient_checkpointing=False)
    ck=load_checkpoint(member['checkpoint'])
    if ck['stage']!='ddr_finetune' or ck['is_calibration'] or ck['family']!=cfg['structure']['family']:
        raise ValueError('Locked candidate must be a formal same-family DDR checkpoint')
    model.load_state_dict(ck['model_state'],strict=True);del ck
    model.cuda().eval()
    return model,cfg,dc

def candidate_probability(rgb,item):
    model,cfg,dc=item;orig=rgb.shape[:2];kernels(bool(cfg.get('execution_fast_kernels')))
    size=cfg['metrics'].get('resize_to')
    if size:rgb=cv2.resize(rgb,(size[1],size[0]),interpolation=cv2.INTER_LINEAR)
    chw=normalize_image(rgb,dc['mean'],dc['std']).transpose(2,0,1)
    padded,hw=pad_normalized_image(np.ascontiguousarray(chw))
    x=torch.from_numpy(padded[None]).cuda()
    if cfg['metrics'].get('eval_mode')=='sliding_window':
        logits=sliding_window_logits(model,x,cfg['metrics']['window_size'],cfg['metrics']['window_stride'])
    else:logits=model.eval_forward(x)
    logits=logits[...,:hw[0],:hw[1]]
    return torch.sigmoid(F.interpolate(logits.float(),size=orig,mode='bilinear',align_corners=False))[0].cpu().numpy()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--lock',required=True,type=Path);a=ap.parse_args()
    lock=json.loads(a.lock.read_text(encoding='utf-8'));identity=sha(a.lock)
    if lock['threshold']!=.5 or not lock['strict_greater']:raise ValueError('Strict threshold required')
    checkpoint=ROOT/'reports/finish/test_progress.json';out=ROOT/'reports/finish/final_test.json'
    if out.exists():
        if json.loads(out.read_text())['lock_sha256']!=identity:raise ValueError('Test was already evaluated under another lock')
        print('Same locked test already complete; no rerun');return
    if checkpoint.exists():
        progress=json.loads(checkpoint.read_text())
        if progress['lock_sha256']!=identity:raise ValueError('Cannot change a started test lock')
    else:
        progress=dict(lock_sha256=identity,images={},status='started');atomic(checkpoint,progress)
    settings=json.loads(Path(lock['champion_settings']).read_text())
    for rel,expected in settings['expected_sha256'].items():
        p=Path(rel) if Path(rel).is_absolute() else Path(settings['repo'])/rel
        if sha(p)!=expected:raise ValueError('Locked champion identity changed')
    import runtime
    runtime.inject_repo(settings['repo']);cfg=runtime.load_config(settings['repo'],settings['config'])
    from mmseg.datasets import build_dataset
    d=copy.deepcopy(cfg.data['test']);d.test_mode=True;d.data_root=lock['ddr_root']
    d.img_dir=str(Path(lock['ddr_root'])/'images/test');d.ann_dir=str(Path(lock['ddr_root'])/'labels/test')
    ds=build_dataset(d)
    if len(ds)!=227:raise ValueError('Test must contain the fixed 227 images')
    torch.set_num_threads(4);kernels(False)
    champion=runtime.ChampionEnsemble(settings,settings['champion'],device='cuda')
    used={o['family'] for o in lock['deploy'] if o['family']}
    candidates={family:[(load_candidate(m),m['weight']) for m in lock['candidates'][family]['members']] for family in used}
    with torch.no_grad():
        for i in range(len(ds)):
            sample=ds[i];meta=runtime._unwrap_meta(sample['img_metas']);sid=Path(meta[0]['ori_filename']).stem
            if sid in progress['images']:continue
            img=runtime._unwrap_img(sample['img']).unsqueeze(0).cuda()
            kernels(False);base=champion.predict(img,meta).float().cpu().numpy()
            rgb=np.asarray(Image.open(Path(lock['ddr_root'])/'images/test'/f'{sid}.jpg').convert('RGB'))
            gt=np.asarray(Image.open(Path(lock['ddr_root'])/'labels/test'/f'{sid}.png'))
            if base.shape!=(4,*gt.shape) or rgb.shape[:2]!=gt.shape:raise ValueError('Test original-coordinate mismatch')
            predictions={}
            for family,members in candidates.items():
                p=None;total=sum(w for _,w in members)
                for item,w in members:
                    q=candidate_probability(rgb,item)*float(w/total);p=q if p is None else p+q
                predictions[family]=p
            selected=base.copy()
            for c,o in enumerate(lock['deploy']):
                if o['family']:selected[c]=(1-o['alpha'])*base[c]+o['alpha']*predictions[o['family']][c]
            progress['images'][sid]=dict(champion=runtime.per_image_counts(base,gt),selected_pipeline=runtime.per_image_counts(selected,gt))
            atomic(checkpoint,progress)
            print(f'locked test progress {len(progress["images"])}/227',flush=True)
    if len(progress['images'])!=227:raise ValueError('Incomplete fixed test identity')
    base=sum(np.array(v['champion'],dtype=np.int64) for v in progress['images'].values())
    selected=sum(np.array(v['selected_pipeline'],dtype=np.int64) for v in progress['images'].values())
    bm=float(np.mean(iou(base)));sm=float(np.mean(iou(selected)))
    report=dict(lock_sha256=identity,n_images=227,champion_miou=bm,selected_pipeline_miou=sm,
        delta=sm-bm,champion_per_class=iou(base).tolist(),selected_pipeline_per_class=iou(selected).tolist(),
        champion_counts=base.tolist(),selected_pipeline_counts=selected.tolist(),
        deployment='selected_pipeline' if sm>bm else 'retain_champion',threshold=.5,strict_greater=True,
        historical_test_limitation='This split has been viewed in prior experiments; not an independent external validation.')
    atomic(out,report);progress['status']='complete';atomic(checkpoint,progress)
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
