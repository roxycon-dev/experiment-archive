"""One locked final evaluation when an existing fixed-25% G pipeline wins on val."""
import copy
import os
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from nextlib.common import read_json,write_json,sha,source_identity
from nextlib.g_stage import load_old
from nextlib.evaluate import legacy_logits,counts,metrics
from analyze_cache import image_bootstrap

def main():
    lock_path=ROOT/'work/final_lock.json';lock=read_json(lock_path);identity=sha(lock_path)
    assert lock['kind']=='G_fixed25'
    assert lock['runner_sha256']==sha(Path(__file__))
    assert lock['source_sha256']==source_identity()['sha256']
    settings=read_json(ROOT/'configs/settings.json');s=read_json(settings['champion_settings'])
    assert sha(settings['champion_settings'])==lock['champion_settings_sha256']
    assert sha(ROOT/'configs/protocol.json')==lock['protocol_sha256']
    for spec in lock['assets'].values():assert sha(spec['path'])==spec['sha256']
    out=ROOT/'work/final_test.json';progress_path=ROOT/'work/test_progress.json'
    if out.exists():
        assert read_json(out)['lock_sha256']==identity;print('Same final G test complete; no rerun');return
    progress=read_json(progress_path) if progress_path.exists() else dict(lock_sha256=identity,images={})
    assert progress['lock_sha256']==identity;write_json(progress_path,progress)
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    sys.path.insert(0,str(ROOT/'legacy'));import runtime
    runtime.inject_repo(s['repo']);cfg=runtime.load_config(s['repo'],s['config'])
    from mmseg.datasets import build_dataset
    d=copy.deepcopy(cfg.data['test']);d.test_mode=True;d.data_root=settings['ddr_root']
    d.img_dir=str(Path(settings['ddr_root'])/'images/test');d.ann_dir=str(Path(settings['ddr_root'])/'labels/test')
    dataset=build_dataset(d);assert len(dataset)==227
    champion=runtime.ChampionEnsemble(s,s['champion'],'cuda');u0=load_old(settings,'U0')
    candidate=load_old(settings,lock['asset']) if lock['asset']!='U0' else u0
    for i in range(len(dataset)):
        sample=dataset[i];meta=runtime._unwrap_meta(sample['img_metas'])
        sid=Path(meta[0].get('ori_filename',meta[0]['filename'])).stem
        if sid in progress['images']:continue
        img=runtime._unwrap_img(sample['img']).unsqueeze(0).cuda()
        with Image.open(Path(settings['ddr_root'])/'images/test'/(sid+'.jpg')) as im:rgb=np.array(im.convert('RGB'))
        with Image.open(Path(settings['ddr_root'])/'labels/test'/(sid+'.png')) as im:gt=np.array(im)
        assert gt.shape==rgb.shape[:2] and gt.max()<=4
        with torch.no_grad():
            pc=champion.predict(img,meta).float().cpu().numpy()
            pu0=legacy_logits(u0,rgb,'native','whole').sigmoid().numpy()[0]
            p=legacy_logits(candidate,rgb,lock['canvas'],lock['inference']).sigmoid().numpy()[0]
            old=(.75*pc+.25*pu0).astype(np.float32)
            selected=(.75*pc+.25*p).astype(np.float32)
        progress['images'][sid]=dict(old_complete=counts(old,gt).tolist(),selected_complete=counts(selected,gt).tolist(),
            image_sha256=sha(Path(settings['ddr_root'])/'images/test'/(sid+'.jpg')),
            label_sha256=sha(Path(settings['ddr_root'])/'labels/test'/(sid+'.png')))
        write_json(progress_path,progress);print(f'locked G final test {len(progress["images"])}/227',flush=True)
    assert len(progress['images'])==227
    baseline=metrics(np.asarray([x['old_complete'] for x in progress['images'].values()]).sum(0))
    selected=metrics(np.asarray([x['selected_complete'] for x in progress['images'].values()]).sum(0))
    write_json(out,dict(lock_sha256=identity,n_images=227,old_complete_pipeline=baseline,
        selected_complete_pipeline=selected,delta=selected['mIoU_lesion4']-baseline['mIoU_lesion4'],
        target=.472115,target_met=selected['mIoU_lesion4']>=.472115,
        paired_image_uncertainty=image_bootstrap([x['selected_complete'] for x in progress['images'].values()],
            [x['old_complete'] for x in progress['images'].values()]),selection_basis='prelocked validation; no retuning after test',
        historical_limitation='227-image test inspected in past rounds; exploratory continuation'))

if __name__=='__main__':main()
