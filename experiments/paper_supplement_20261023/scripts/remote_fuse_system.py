"""Actual A/H4 forward plus immutable U/S/C caches, preserving the four frozen formulas."""
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from remote_cache_branch import make_cohort

ROOT=Path(__file__).resolve().parents[1];EXP=ROOT.parent
READY=EXP/'HRDecoder_Ready_20261002'


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()


def write(p,value):
    t=p.with_suffix('.tmp');t.write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8');os.replace(str(t),str(p))


def build_external():
    torch.set_num_threads(4);cv2.setNumThreads(2)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    sys.path.insert(0,str(READY/'legacy'))
    sys.path.append(str(EXP/'HRDecoder_ArchitectureDesign_20261008/training_execution/python_deps'))
    import runtime
    settings=json.loads((READY/'configs/champion_server.json').read_text(encoding='utf-8'))
    settings['repo']=str(READY/'vendor/champion_repo');settings['data_root']=str(EXP/'data/idrid')
    settings['reference_checkpoint']=str(READY/'best_pipeline/weights/B_weights.pth')
    settings['champion']['head']=str(READY/'best_pipeline/weights/H4_best.pth')
    for m in settings['champion']['members']:m['checkpoint']=str(READY/'best_pipeline/weights'/Path(m['checkpoint']).name)
    runtime.inject_repo(settings['repo']);cfg=runtime.load_config(settings['repo'],settings['config'])
    from mmseg.datasets import build_dataset
    d=copy.deepcopy(cfg.data['val']);d.data_root=settings['data_root'];d.img_dir='images/testing';d.ann_dir='labels/testing'
    d.pipeline=copy.deepcopy(cfg.data['test']['pipeline']);d.test_mode=True
    if 'split' in d:d.split=None
    dataset=build_dataset(d)
    champion=runtime.ChampionEnsemble(settings,settings['champion'],device='cuda')
    return runtime,champion,dataset,cfg


def main():
    out=ROOT/'results/idrid_holdout_v1';out.mkdir(exist_ok=False)
    for name in ['per_image','predictions']:(out/name).mkdir()
    cache=ROOT/'results/remote_probability_cache'
    for arm in ['S','C','U']:
        completion=json.loads((cache/arm/'completion.json').read_text(encoding='utf-8'))
        assert completion['status']=='COMPLETE' and completion['external_images']==27
    cohort=make_cohort();write(out/'cohort.json',cohort)
    source=json.loads((ROOT/'evidence/source_DDR_manifest.json').read_text(encoding='utf-8'))
    receipt=json.loads((READY/'best_pipeline/weights/exact_transport_receipt.json').read_text(encoding='utf-8'))
    assert receipt['status']=='PASS'
    for row in receipt['files']:
        assert row['source_file_sha256']==source['assets'][row['name']]['sha256']
        assert sha(READY/'best_pipeline/weights'/row['name'])==row['reconstructed_file_sha256']
    assert sha(READY/'best_pipeline/weights/H4_best.pth')==source['assets']['H4_best.pth']['sha256']
    cache_manifests={a:json.loads((cache/a/'manifest.json').read_text(encoding='utf-8')) for a in ['S','C','U']}
    assert cache_manifests['S']['weight_sha256']==source['assets']['S20']['sha256']
    assert cache_manifests['U']['weight_sha256']==source['assets']['U0']['sha256']
    compact=json.loads((ROOT/'evidence/C20_export_receipt.json').read_text(encoding='utf-8'))
    assert cache_manifests['C']['weight_sha256']==compact['compact_sha256']
    assert compact['source_sha256']==source['assets']['C20']['sha256']
    manifest={'run':'idrid_holdout_v1','mode':'external','execution_host':'remote','expected_images':27,
              'scope':'independent_heldout_subset','sample_ids':[r['sid'] for r in cohort['samples']],
              'cohort_sha256':sha(out/'cohort.json'),'source_client_holdout_manifest_sha256':sha(ROOT/'evidence/external_holdout_manifest.json'),
              'threshold':.5,'strict_greater':True,'new_training':False,'historical_exact_replay':False,
              'runtime_amendment_sha256':sha(ROOT/'plan/remote_runtime_amendment.md'),
              'script_sha256':sha(Path(__file__)),'cache_manifests':cache_manifests,'exact_transport_receipt':receipt,
              'source_assets':source['assets'],'full_formula':'.75*(.75*A+.25*U)+.25*S',
              'timing_scope':'A actual forward; U/S/C cache load and fusion; not full-system end-to-end latency',
              'GT':'four independently valid masks in stored native coordinates, overlaps retained',
              'legacy_torch':torch.__version__,'GPU':torch.cuda.get_device_name()}
    write(out/'manifest.json',manifest)
    runtime,champion,dataset,cfg=build_external()
    lookup={Path(r['filename']).stem:i for i,r in enumerate(dataset.img_infos)}
    assert set(lookup)==set(manifest['sample_ids'])
    # Validation canary before any external A/H4 output. Other cache branches also used this canary first.
    from mmseg.datasets import build_dataset
    canary_cfg=copy.deepcopy(cfg.data['val']);canary_cfg.data_root=str(EXP/'data');canary_cfg.img_dir='canary';canary_cfg.ann_dir=None
    canary_cfg.pipeline=copy.deepcopy(cfg.data['test']['pipeline']);canary_cfg.test_mode=True
    if 'split' in canary_cfg:canary_cfg.split=None
    canary_dataset=build_dataset(canary_cfg)
    jobs=[({'sid':'canary_007-1774-100','image_path':str(EXP/'data/canary/007-1774-100.jpg')},canary_dataset[0])]
    jobs += [(row,None) for row in cohort['samples']]
    start=time.perf_counter()
    try:
        for i,(row,item) in enumerate(jobs):
            sid=row['sid'];warmup=i==0
            if item is None:item=dataset[lookup[sid]]
            image=runtime._unwrap_img(item['img']).unsqueeze(0).cuda();meta=runtime._unwrap_meta(item['img_metas'])
            h,w=map(int,meta[0]['ori_shape'][:2]);t=time.perf_counter()
            with torch.no_grad():
                pre,ps,pr=champion.pre_head_probs(image,meta)
                imgf=F.interpolate(image,size=(h,w),mode='bilinear',align_corners=False)[0]
                a=pre.clone();a[3]=champion.corrected_class_prob(3,pre,ps,pr,imgf,champion.heads['MA'])
            a,pre=a.cpu().numpy(),pre.cpu().numpy();del image,ps,pr,imgf
            torch.cuda.synchronize();a_seconds=time.perf_counter()-t
            loaded={};proofs={}
            for arm in ['S','C','U']:
                p=cache/arm/(sid+'.npy');r=json.loads((cache/arm/(sid+'.json')).read_text(encoding='utf-8'))
                assert sha(p)==r['probability_sha256']
                if not warmup:assert r['image_sha256']==row['image_sha256']
                loaded[arm]=np.load(str(p),allow_pickle=False);proofs[arm]=r
            s,c,u=loaded['S'],loaded['C'],loaded['U']
            for q in [a,pre,s,c,u]:assert q.dtype==np.float32 and q.shape==(4,h,w) and np.isfinite(q).all()
            del q
            p0=.75*a+.25*u
            masks={'full':(.75*p0+.25*s)>.5,'no_h4':(.75*(.75*pre+.25*u)+.25*s)>.5,
                   'no_convnext':(np.float32(9/13)*a+np.float32(4/13)*s)>.5,'matched_c20':(.75*p0+.25*c)>.5}
            assert np.array_equal(masks['full'][:3],masks['no_h4'][:3])
            if warmup:
                label=np.asarray(Image.open(EXP/'data/canary/007-1774-100.png'));gt=np.stack([label==k for k in range(1,5)])
            else:
                assert sha(row['image_path'])==row['image_sha256'];gt=[]
                for name in ['EX','HE','SE','MA']:
                    m=row['masks'][name];assert sha(m['path'])==m['sha256'];gt.append(np.asarray(Image.open(m['path']))>0)
                gt=np.stack(gt)
            counts={k:np.stack([(b&gt).sum((1,2)),(b&~gt).sum((1,2)),(~b&gt).sum((1,2))],axis=1).astype(np.int64).tolist() for k,b in masks.items()}
            if warmup:
                write(out/'validation_canary.json',{'sid':sid,'four_row_counts':counts,'branch_canaries':proofs,
                      'scope':'validation compatibility only, no threshold/weight tuning','external_A_predictions_seen':False})
            else:
                target=out/'predictions'/(sid+'.npz');np.savez_compressed(str(target),**masks)
                write(out/'per_image'/(sid+'.json'),{'sid':sid,'warmup':False,'hw':[h,w],'counts':counts,
                    'image_sha256':row['image_sha256'],'prediction_path':str(target),'mask_sha256':sha(target),
                    'timing':{'A_inference_seconds':a_seconds,'with_cached_partners':True},'branch_probability_records':proofs})
            del a,pre,s,c,u,loaded,p0,masks,gt;torch.cuda.empty_cache()
            progress={'stage':'RUNNING','completed':max(i,0),'expected':27,'last_sid':sid,'elapsed_seconds':time.perf_counter()-start}
            write(out/'progress.json',progress);print(json.dumps(progress),flush=True)
        write(out/'completion.json',{'status':'COMPLETE','images':27,'validation_canaries':1,'seconds':time.perf_counter()-start,
                                    'all_predeclared_rows_retained':True,'new_training':False})
    except BaseException as error:
        write(out/'failure.json',{'type':type(error).__name__,'message':str(error),'formal_results_complete':False});raise


if __name__=='__main__':main()
