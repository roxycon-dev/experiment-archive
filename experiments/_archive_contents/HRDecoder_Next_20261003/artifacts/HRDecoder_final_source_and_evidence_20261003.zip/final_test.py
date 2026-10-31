"""One resumable evaluation of an already frozen complete pipeline; no candidate/test search."""
import copy
import os
import sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from nextlib.common import ROOT,read_json,write_json,sha,digest,source_identity,trusted_load
from nextlib.model import build_fusion
from nextlib.train import load_local_state
from nextlib.g_stage import load_old
from nextlib.geometry import canvas
from nextlib.evaluate import counts,metrics,legacy_logits


def main():
    lock_path=ROOT/'work/final_lock.json';lock=read_json(lock_path);identity=sha(lock_path)
    settings=read_json(ROOT/'configs/settings.json')
    out=ROOT/'work/final_test.json';progress_path=ROOT/'work/test_progress.json'
    if out.exists():
        if read_json(out)['lock_sha256']!=identity:raise ValueError('Another final pipeline has already accessed test')
        print('Same final test already complete; no repeated inference');return
    if lock['source_sha256']!=source_identity()['sha256']:raise ValueError('Source changed after final lock')
    for spec in lock['assets'].values():
        if sha(spec['path'])!=spec['sha256']:raise ValueError('Final asset identity changed')
    if sha(lock['checkpoint'])!=lock['checkpoint_sha256']:raise ValueError('Selected M checkpoint changed')
    if sha(lock['champion_settings'])!=lock['champion_settings_sha256']:raise ValueError('Champion settings changed')
    if sha(ROOT/'configs/protocol.json')!=lock['protocol_sha256']:raise ValueError('Training protocol changed')
    progress=dict(lock_sha256=identity,images={})
    if progress_path.exists():
        progress=read_json(progress_path)
        if progress['lock_sha256']!=identity:raise ValueError('Started test lock is immutable')
    write_json(progress_path,progress)
    s=read_json(lock['champion_settings'])
    sys.path.insert(0,str(ROOT/'legacy'))
    import runtime
    cwd=os.getcwd()
    runtime.inject_repo(s['repo'])
    cfg=runtime.load_config(s['repo'],s['config'])
    from mmseg.datasets import build_dataset
    d=copy.deepcopy(cfg.data['test']);d.test_mode=True;d.data_root=lock['ddr_root']
    d.img_dir=str(Path(lock['ddr_root'])/'images/test');d.ann_dir=str(Path(lock['ddr_root'])/'labels/test')
    dataset=build_dataset(d)
    if len(dataset)!=227:raise ValueError('Fixed final test requires 227 images')
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    champion=runtime.ChampionEnsemble(s,s['champion'],'cuda')
    u0=load_old(settings,'U0')
    model,report=build_fusion(settings,lock['geometry']['start_asset'],lock['arm'],lock['seed'])
    ck=trusted_load(lock['checkpoint'])
    if ck['update']!=lock['update']:raise ValueError('Selected best checkpoint/node differs from final lock')
    load_local_state(model,ck['model_state']);del ck
    model.cuda().eval();protocol=read_json(settings['protocol'])
    for i in range(len(dataset)):
        sample=dataset[i];meta=runtime._unwrap_meta(sample['img_metas'])
        sid=Path(meta[0].get('ori_filename',meta[0]['filename'])).stem
        if sid in progress['images']:continue
        img=runtime._unwrap_img(sample['img']).unsqueeze(0).cuda()
        with Image.open(Path(lock['ddr_root'])/'images/test'/(sid+'.jpg')) as im:rgb=np.array(im.convert('RGB'))
        with Image.open(Path(lock['ddr_root'])/'labels/test'/(sid+'.png')) as im:gt=np.array(im)
        if gt.shape!=rgb.shape[:2] or gt.max()>4:raise ValueError('Test GT original-coordinate contract failed')
        with torch.no_grad():
            torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
            torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
            pc=champion.predict(img,meta).float().cpu().numpy()
            pu0=legacy_logits(u0,rgb,'native','whole').sigmoid().numpy()[0]
            old=(.75*pc+.25*pu0).astype(np.float32)
            torch.backends.cudnn.benchmark=True;torch.backends.cudnn.deterministic=False
            torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
            work,_=canvas(rgb,None,lock['geometry']['canvas'])
            logits=model.predict(work,protocol['center'],protocol['halo'],protocol['global_long_side'])
            if work.shape[:2]!=rgb.shape[:2]:logits=F.interpolate(logits,rgb.shape[:2],mode='bilinear',align_corners=False)
            new=logits.sigmoid().numpy()[0]
            selected=new if lock['output']=='solo' else .5*old+.5*new
        progress['images'][sid]=dict(old_complete=counts(old,gt).tolist(),selected_complete=counts(selected,gt).tolist(),
            image_sha256=sha(Path(lock['ddr_root'])/'images/test'/(sid+'.jpg')),
            label_sha256=sha(Path(lock['ddr_root'])/'labels/test'/(sid+'.png')))
        write_json(progress_path,progress)
        print(f'locked final test {len(progress["images"])}/227',flush=True)
    if len(progress['images'])!=227:raise ValueError('Incomplete test')
    baseline=metrics(np.asarray([r['old_complete'] for r in progress['images'].values()]).sum(0))
    selected=metrics(np.asarray([r['selected_complete'] for r in progress['images'].values()]).sum(0))
    from analyze_cache import image_bootstrap
    result=dict(lock_sha256=identity,n_images=227,old_complete_pipeline=baseline,selected_complete_pipeline=selected,
        delta=selected['mIoU_lesion4']-baseline['mIoU_lesion4'],target=.472115,
        target_met=selected['mIoU_lesion4']>=.472115,
        paired_image_uncertainty=image_bootstrap([r['selected_complete'] for r in progress['images'].values()],
                                               [r['old_complete'] for r in progress['images'].values()]),
        selection_basis='prelocked validation; no retuning after this test',
        historical_limitation=lock['historical_test_note'])
    write_json(out,result);os.chdir(cwd);print(result)


if __name__=='__main__':main()
