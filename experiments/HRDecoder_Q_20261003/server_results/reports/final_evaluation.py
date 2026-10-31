"""Freeze one val-selected pipeline, then evaluate it once for both seeds.

Legacy P0 runs in its original environment; Q runs in the accepted isolated
environment. No phase may read test labels before the two-seed gate passes.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from qlib.common import assert_source,digest,read_json,sha,utc_now,write_json
from qlib.decision import lock_selection

LOCK=ROOT/'reports/final_selection.json'
OUT=ROOT/'reports/final_test'
LEGACY=Path('/root/autodl-tmp/HRDecoder_Ready_20261002')


def freeze():
    if LOCK.exists():
        gate()
        print('Existing immutable final selection retained')
        return
    c=read_json(ROOT/'reports/campaign_state.json')
    assert c['next_action']=='FREEZE_SELECTION_THEN_FINAL_TEST'
    assert c['arms']==['Q_DDR'] and c['review_seed_authorized'] and not c['architecture_stopped']
    endpoint=c['common_endpoint']
    results={'Q_DDR':{}}
    candidate_metrics={}
    for seed in (42,2024):
        run=ROOT/'runs'/f'Q_DDR_seed{seed}'
        status=read_json(run/'status.json')
        assert status['stage']=='COMPLETE' and status['completed']==endpoint
        results['Q_DDR'][seed]={int(p.parent.name[5:]):read_json(p) for p in run.glob('val_u*/metrics.json')}
        candidate_metrics.update({str(p):sha(p) for p in run.glob('val_u*/metrics.json') if 12000<=int(p.parent.name[5:])<=endpoint})
    selection=lock_selection(results,endpoint)
    assert selection==c['selection'] and selection['selected']['test_eligible']
    selected=selection['selected']
    checkpoints={}
    for seed in (42,2024):
        run=ROOT/'runs'/f'Q_DDR_seed{seed}'
        path=run/f'checkpoint_u{selected["update"]}.pth'
        checkpoints[str(seed)]={'path':str(path),'sha256':sha(path),'identity':read_json(run/'identity.json')}
    source=assert_source(ROOT)
    original=ROOT/'configs/p0_original_lock.json'
    assert sha(original)==read_json(ROOT/'configs/p0_identity.json')['historical_lock_sha256']
    old=read_json(original)
    assert all(d['family']=='convnext_unet' and d['alpha']==.25 for d in old['deploy'])
    settings=Path(old['champion_settings'])
    legacy_files=[LEGACY/'tools/final_eval_locked.py',LEGACY/'legacy/runtime.py',
                  LEGACY/'legacy/fusion_core.py',LEGACY/'trainer/models/convnext_unet.py']
    legacy_hashes={str(p):sha(p) for p in legacy_files if p.is_file()}
    payload={'status':'LOCKED_TEST_AUTHORIZED','selection':selection,'created_at':utc_now(),
        'source_sha256':source['sha256'],'driver_sha256':sha(__file__),
        'runtime_sha256':sha(ROOT/'configs/runtime.server.json'),'checkpoints':checkpoints,
        'data_identity_sha256':sha(ROOT/'reports/data_identity.json'),
        'candidate_metrics_sha256':candidate_metrics,
        'p0_original_lock_sha256':sha(original),'champion_settings_sha256':sha(settings),
        'p0_reference_sha256':sha(ROOT/'reports/p0_historical_test_reference.json'),
        'legacy_source_sha256':legacy_hashes,'p0_cache':'/dev/shm/hrdecoder_q_final_p0_20261003',
        'target_test_pct':47.2115,'deployment_seed':42,'test_based_reselection_allowed':False,
        'historical_limitation':'DDR val and test have been used in previous experiments; exploratory continuation.'}
    write_json(LOCK,payload)
    print('One val-only final selection frozen; no test label pixels read by freeze; SHA256='+sha(LOCK))


def gate():
    lock=read_json(LOCK)
    assert lock['status']=='LOCKED_TEST_AUTHORIZED' and lock['selection']['selected']['test_eligible']
    selected=lock['selection']['selected']
    assert selected['formula'] in ('single','fixed_fusion') and selected['deployment_seed']==42
    assert all(s>43.48770322103657 for s in selected['seed_scores_pct']) and selected['mean_val_pct']>=43.98770322103657
    assert lock['driver_sha256']==sha(__file__) and lock['source_sha256']==assert_source(ROOT)['sha256']
    for file,key in [('configs/runtime.server.json','runtime_sha256'),('reports/data_identity.json','data_identity_sha256'),
                     ('configs/p0_original_lock.json','p0_original_lock_sha256'),
                     ('reports/p0_historical_test_reference.json','p0_reference_sha256')]:
        assert sha(ROOT/file)==lock[key],file
    for path,expected in lock['legacy_source_sha256'].items():
        assert sha(path)==expected,path
    for path,expected in lock['candidate_metrics_sha256'].items():
        assert sha(path)==expected,path
    old=read_json(ROOT/'configs/p0_original_lock.json')
    assert sha(old['champion_settings'])==lock['champion_settings_sha256']
    for spec in lock['checkpoints'].values():
        assert sha(spec['path'])==spec['sha256'],spec['path']
    return lock,sha(LOCK)


def p0_phase():
    lock,lock_sha=gate()
    import numpy as np
    import torch
    from PIL import Image
    from qlib.metrics import counts,summarize
    old=read_json(ROOT/'configs/p0_original_lock.json')
    reference=read_json(ROOT/'reports/p0_historical_test_reference.json')
    data=read_json(ROOT/'reports/data_identity.json')['splits']['test']['samples']
    expected={r['sid']:r for r in data}
    destination=Path(lock['p0_cache'])
    destination.mkdir(parents=True,exist_ok=True)
    manifest_path=OUT/'p0_manifest.json'
    manifest={'lock_sha256':lock_sha,'images':{},'status':'RUNNING','created_at':utc_now()}
    if manifest_path.exists():
        manifest=read_json(manifest_path)
        assert manifest['lock_sha256']==lock_sha
        if manifest['status']=='COMPLETE' and all(sha(destination/(sid+'.npz'))==r['sha256'] for sid,r in manifest['images'].items()):
            print('Exact locked P0 test probabilities already complete; no repeated inference')
            return
    remaining_raw=sum(16*r['hw'][0]*r['hw'][1] for r in data if r['sid'] not in manifest['images'])
    capacity=os.statvfs(destination)
    assert capacity.f_bavail*capacity.f_frsize>=remaining_raw+1024**3, 'Insufficient RAM cache storage; no test inference launched'
    sys.path.insert(0,str(LEGACY))
    sys.path.insert(0,str(LEGACY/'legacy'))
    from tools.final_eval_locked import load_candidate,candidate_probability,kernels
    import runtime
    settings=read_json(old['champion_settings'])
    for rel,h in settings['expected_sha256'].items():
        path=Path(rel) if Path(rel).is_absolute() else Path(settings['repo'])/rel
        assert sha(path)==h,str(path)
    runtime.inject_repo(settings['repo'])
    cfg=runtime.load_config(settings['repo'],settings['config'])
    from mmseg.datasets import build_dataset
    d=copy.deepcopy(cfg.data['test'])
    d.test_mode=True;d.data_root=old['ddr_root']
    d.img_dir=str(Path(old['ddr_root'])/'images/test');d.ann_dir=str(Path(old['ddr_root'])/'labels/test')
    dataset=build_dataset(d)
    assert len(dataset)==227
    torch.set_num_threads(4);kernels(False)
    champion=runtime.ChampionEnsemble(settings,settings['champion'],device='cuda')
    members=[(load_candidate(m),m['weight']) for m in old['candidates']['convnext_unet']['members']]
    started=time.perf_counter()
    with torch.no_grad():
        for i in range(len(dataset)):
            sample=dataset[i]
            meta=runtime._unwrap_meta(sample['img_metas'])
            sid=Path(meta[0].get('ori_filename',meta[0]['filename'])).stem
            assert sid in expected
            target_path=destination/(sid+'.npz')
            if sid in manifest['images'] and target_path.exists() and sha(target_path)==manifest['images'][sid]['sha256']:
                continue
            rgb_path=Path(old['ddr_root'])/'images/test'/(sid+'.jpg')
            gt_path=Path(old['ddr_root'])/'labels/test'/(sid+'.png')
            assert sha(rgb_path)==expected[sid]['image_sha256'] and sha(gt_path)==expected[sid]['label_sha256']
            with Image.open(rgb_path) as image: rgb=np.asarray(image.convert('RGB'))
            with Image.open(gt_path) as image: label=np.asarray(image)
            img=runtime._unwrap_img(sample['img']).unsqueeze(0).cuda()
            kernels(False)
            pc=champion.predict(img,meta).float().cpu().numpy()
            total=sum(w for _,w in members)
            pu=None
            for item,w in members:
                p=candidate_probability(rgb,item)*float(w/total)
                pu=p if pu is None else pu+p
            p0=(.75*pc+.25*pu).astype(np.float32)
            gt=np.stack([label==c for c in range(1,5)])
            valid=np.broadcast_to(label!=255,gt.shape)
            ct=counts(p0,gt,valid).tolist()
            assert ct==reference['per_image'][sid], 'Historical P0 test per-image identity differs: '+sid
            temporary=target_path.with_suffix('.tmp')
            with temporary.open('wb') as f: np.savez(f,prob=p0)
            temporary.replace(target_path)
            manifest['images'][sid]={'counts':ct,'sha256':sha(target_path),'dtype':'float32','shape':list(p0.shape)}
            write_json(manifest_path,manifest)
            print(f'Fixed historical P0 reference {len(manifest["images"])}/227',flush=True)
    assert set(manifest['images'])==set(expected)
    aggregate=summarize(np.asarray([r['counts'] for r in manifest['images'].values()]).sum(0))
    assert aggregate['counts']==reference['counts']
    manifest.update(status='COMPLETE',metrics=aggregate,seconds=time.perf_counter()-started,
        storage='FP32_uncompressed_ephemeral_RAM_cache',selection_use=False,completed_at=utc_now())
    write_json(manifest_path,manifest)
    print('P0 original test per-image and aggregate counts exactly reproduced')


def lesion_buckets(prob,gt,valid):
    import cv2
    import numpy as np
    result=np.zeros((4,5,5),np.int64)
    for c in range(4):
        mask=(gt[c].astype(bool)&valid[c].astype(bool)).astype(np.uint8)
        n,cc,stat,_=cv2.connectedComponentsWithStats(mask,connectivity=8)
        pred=prob[c]>.5
        for k in range(1,n):
            x,y,w,h,area=stat[k]
            lesion=cc[y:y+h,x:x+w]==k
            detected=pred[y:y+h,x:x+w]&lesion
            eroded=cv2.erode(lesion.astype(np.uint8),np.ones((3,3),np.uint8),
                borderType=cv2.BORDER_CONSTANT,borderValue=0).astype(bool)
            missed=lesion&~pred[y:y+h,x:x+w]
            bucket=min(int(np.searchsorted([16,64,256,1024],area,side='left')),4)
            result[c,bucket]+=[1,int(not detected.any()),int(area),int(missed.sum()),int((missed&~eroded).sum())]
    return result


def q_phase():
    lock,lock_sha=gate()
    import cv2
    import numpy as np
    import torch
    from qlib.data import DDR
    from qlib.model import build_model
    from qlib.geometry import tiles,rgb_window,center_logits
    from qlib.evaluate import load_probability
    from qlib.metrics import counts,summarize,paired_bootstrap
    cv2.setNumThreads(1)
    runtime=read_json(ROOT/'configs/runtime.server.json')
    reference=read_json(ROOT/'reports/p0_historical_test_reference.json')
    p0_manifest=read_json(OUT/'p0_manifest.json')
    assert p0_manifest['lock_sha256']==lock_sha and p0_manifest['status']=='COMPLETE'
    selected=lock['selection']['selected']
    dtype={'bfloat16':torch.bfloat16,'float16':torch.float16}[runtime['resources']['dtype']]
    data=read_json(ROOT/'reports/data_identity.json')
    dataset=DDR(runtime['ddr_root'],'test',data['splits']['test'],test_authorized=True)
    assert set(p0_manifest['images'])==set(dataset.ids)
    expected={r['sid']:r for r in data['splits']['test']['samples']}
    for seed in (42,2024):
        path=OUT/f'seed{seed}.json'
        result={'lock_sha256':lock_sha,'seed':seed,'formula':selected['formula'],'update':selected['update'],
                'images':{},'status':'RUNNING','created_at':utc_now()}
        if path.exists():
            result=read_json(path)
            assert result['lock_sha256']==lock_sha
            if result['status']=='COMPLETE':
                print(f'Seed{seed} same locked test already complete; no repeated inference')
                continue
        spec=lock['checkpoints'][str(seed)]
        state=torch.load(spec['path'],map_location='cpu',weights_only=False)
        assert state['identity']==spec['identity'] and state['identity_sha256']==digest(spec['identity'])
        assert state['completed']==selected['update'] and state['cursor']==selected['update']*8
        pretrained=read_json(ROOT/'reports/pretraining_identity.json')
        model,_=build_model(runtime['dinov3_repo'],runtime['pretrained_checkpoint'],pretrained['sha256'])
        model.load_state_dict(state['model'],strict=True)
        del state
        model=model.to('cuda').eval()
        started=time.perf_counter()
        with torch.no_grad():
            for sid in dataset.ids:
                if sid in result['images']:continue
                assert sha(Path(runtime['ddr_root'])/'images/test'/(sid+'.jpg'))==expected[sid]['image_sha256']
                assert sha(Path(runtime['ddr_root'])/'labels/test'/(sid+'.png'))==expected[sid]['label_sha256']
                rgb,g,v=dataset.load(sid)
                hw=rgb.shape[:2]
                q=np.empty((4,*hw),np.float32)
                image_started=time.perf_counter()
                for tile in tiles(*hw):
                    x=torch.from_numpy(rgb_window(rgb,tile))[None].to('cuda')
                    with torch.autocast('cuda',dtype=dtype): z=center_logits(model(x))
                    p=z.float().sigmoid()[0,:,:tile.height,:tile.width].cpu().numpy()
                    q[:,tile.y:tile.y+tile.height,tile.x:tile.x+tile.width]=p
                inference_seconds=time.perf_counter()-image_started
                p0_path=Path(lock['p0_cache'])/(sid+'.npz')
                assert sha(p0_path)==p0_manifest['images'][sid]['sha256']
                p0=load_probability(p0_path,hw)
                probability=q if selected['formula']=='single' else .75*p0+.25*q
                gt,valid=g.transpose(2,0,1),v.transpose(2,0,1)
                old_counts=counts(p0,gt,valid).tolist()
                assert old_counts==reference['per_image'][sid]
                result['images'][sid]={'selected':counts(probability,gt,valid).tolist(),'p0':old_counts,
                    'selected_lesion_buckets':lesion_buckets(probability,gt,valid).tolist(),
                    'p0_lesion_buckets':lesion_buckets(p0,gt,valid).tolist(),'q_inference_seconds':inference_seconds}
                write_json(path,result)
                print(f'Locked formula seed{seed} test {len(result["images"])}/227',flush=True)
        rows=[result['images'][sid] for sid in dataset.ids]
        assert len(rows)==227
        result.update(status='COMPLETE',selected=summarize(np.asarray([r['selected'] for r in rows]).sum(0)),
                      p0=summarize(np.asarray([r['p0'] for r in rows]).sum(0)),n_images=227,
                      seconds=time.perf_counter()-started,completed_at=utc_now())
        result['paired_image_uncertainty']=paired_bootstrap([r['p0'] for r in rows],[r['selected'] for r in rows])
        result['selected_lesion_buckets']=np.asarray([r['selected_lesion_buckets'] for r in rows]).sum(0).tolist()
        result['p0_lesion_buckets']=np.asarray([r['p0_lesion_buckets'] for r in rows]).sum(0).tolist()
        result['lesion_area_upper_bounds']=[16,64,256,1024,None]
        result['lesion_columns']=['lesions','completely_missed','gt_pixels','fn_pixels','boundary_fn_pixels']
        result['boundary_definition']='one-pixel inner ring from 3x3 constant-zero erosion of each 8-connected GT component'
        result['target_met']=result['selected']['miou_pct']>=lock['target_test_pct']
        write_json(path,result)
        del model
        torch.cuda.empty_cache()
    scores=[read_json(OUT/f'seed{s}.json')['selected']['miou_pct'] for s in (42,2024)]
    summary={'status':'COMPLETE','lock_sha256':lock_sha,'formula':selected['formula'],
        'update':selected['update'],'deployment_seed':42,'seed_scores_pct':scores,
        'mean_test_pct':sum(scores)/2,'range_test_pct':[min(scores),max(scores)],
        'target_test_pct':lock['target_test_pct'],'both_seeds_target_met':all(x>=lock['target_test_pct'] for x in scores),
        'mean_target_met':sum(scores)/2>=lock['target_test_pct'],'selection_changed_after_test':False,
        'historical_limitation':lock['historical_limitation'],'completed_at':utc_now()}
    write_json(OUT/'summary.json',summary)
    print(json.dumps(summary))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--phase',required=True,choices=['freeze','p0','q'])
    phase=ap.parse_args().phase
    {'freeze':freeze,'p0':p0_phase,'q':q_phase}[phase]()


if __name__=='__main__':main()
