"""Only the predeclared A/U/P0 shared cache; no seed or fusion search."""
import argparse
import copy
import json
from pathlib import Path
import sys
import time
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from evaluation59_common import ROOT,EXP,DATA,READY,sha,write,samples,image_rgb,counts,cache_dir,load_cached,save_probability

def build_champion():
    sys.path.insert(0,str(READY/'legacy'))
    sys.path.append(str(EXP/'HRDecoder_ArchitectureDesign_20261008/training_execution/python_deps'))
    import runtime
    settings=json.loads((READY/'configs/champion_server.json').read_text(encoding='utf-8'))
    settings['repo']=str(READY/'vendor/champion_repo');settings['data_root']=str(DATA)
    settings['reference_checkpoint']=str(READY/'best_pipeline/weights/B_weights.pth')
    settings['champion']['head']=str(READY/'best_pipeline/weights/H4_best.pth')
    for m in settings['champion']['members']:m['checkpoint']=str(READY/'best_pipeline/weights'/Path(m['checkpoint']).name)
    runtime.inject_repo(settings['repo']);cfg=runtime.load_config(settings['repo'],settings['config'])
    from mmseg.datasets import build_dataset
    sets={};lookups={}
    for name in ('canary','ddr','idrid'):
        d=copy.deepcopy(cfg.data['val']);d.data_root=str(EXP/'data/idrid') if name=='idrid' else str(DATA)
        d.img_dir='images/testing' if name=='idrid' else 'images/'+('val' if name=='canary' else 'test')
        d.ann_dir=None;d.pipeline=copy.deepcopy(cfg.data['test']['pipeline']);d.test_mode=True
        if 'split' in d:d.split=None
        sets[name]=build_dataset(d);lookups[name]={Path(r['filename']).stem:i for i,r in enumerate(sets[name].img_infos)}
    champion=runtime.ChampionEnsemble(settings,settings['champion'],device='cuda')
    return runtime,champion,sets,lookups

def identity(branch):
    source=json.loads((ROOT/'evidence/source_DDR_manifest.json').read_text(encoding='utf-8'))
    result={'branch':branch,'script_sha256':sha(Path(__file__)),'helper_sha256':sha(ROOT/'scripts/evaluation59_common.py'),
            'protocol_sha256':sha(ROOT/'plan/paired_evaluation_runtime.md'),'torch':torch.__version__,
            'GPU':torch.cuda.get_device_name() if branch!='P0' else None,'precision':'FP32','threshold_or_weight_search':False}
    if branch=='A':
        receipt=json.loads((READY/'best_pipeline/weights/exact_transport_receipt.json').read_text(encoding='utf-8'))
        assert receipt['status']=='PASS'
        for r in receipt['files']:
            assert r['source_file_sha256']==source['assets'][r['name']]['sha256']
            assert sha(READY/'best_pipeline/weights'/r['name'])==r['reconstructed_file_sha256']
        assert sha(READY/'best_pipeline/weights/H4_best.pth')==source['assets']['H4_best.pth']['sha256']
        result['weight_transport_receipt']=receipt
    elif branch=='U':
        p=EXP/'HRDecoder_ArchitectureDesign_20261008/cache_execution/inference_weights/U0_inference.pth'
        assert sha(p)==source['assets']['U0']['sha256'];result['weight_sha256']=sha(p)
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--branch',choices=['A','U','P0'],required=True);args=p.parse_args()
    branch=args.branch
    cv2.setNumThreads(1);torch.set_num_threads(1)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    manifest=identity(branch)
    cohorts=['canary','ddr'] if branch=='U' else ['canary','ddr','idrid']
    for cohort in cohorts:
        out=cache_dir(cohort,branch);out.mkdir(parents=True,exist_ok=True)
        target=out/'manifest.json'
        if target.exists():assert json.loads(target.read_text())==manifest,'Identity changed on resume'
        else:write(target,manifest)
    if branch=='A':runtime,champion,sets,lookup=build_champion()
    elif branch=='U':
        sys.path.insert(0,str(READY))
        from trainer.models.convnext_unet import ConvNeXtUNet
        from trainer.eval import pad_normalized_image
        from trainer.data.common import normalize_image
        model=ConvNeXtUNet()
        state=torch.load(str(EXP/'HRDecoder_ArchitectureDesign_20261008/cache_execution/inference_weights/U0_inference.pth'),map_location='cpu')
        model.load_state_dict(state['model_state'],strict=True);del state
        model=model.cuda().eval().requires_grad_(False)
    start=time.perf_counter()
    try:
        for cohort in cohorts:
            out=cache_dir(cohort,branch)
            rows=samples(cohort)
            for i,row in enumerate(rows):
                sid=row['sid'];record=out/(sid+'.json')
                if record.exists():load_cached(cohort,branch,sid);continue
                rgb=image_rgb(row);h,w=rgb.shape[:2];sources={}
                if branch!='P0':torch.cuda.reset_peak_memory_stats()
                t=time.perf_counter()
                with torch.no_grad():
                    if branch=='A':
                        item=sets[cohort][lookup[cohort][sid]]
                        image=runtime._unwrap_img(item['img']).unsqueeze(0).cuda();meta=runtime._unwrap_meta(item['img_metas'])
                        assert tuple(meta[0]['ori_shape'][:2])==(h,w)
                        pre,ps,pr=champion.pre_head_probs(image,meta)
                        imgf=F.interpolate(image,size=(h,w),mode='bilinear',align_corners=False)[0]
                        a=pre.clone();a[3]=champion.corrected_class_prob(3,pre,ps,pr,imgf,champion.heads['MA'])
                        assert torch.equal(a[:3],pre[:3])
                        q=a.cpu().numpy();del image,meta,pre,ps,pr,imgf,a,item
                    elif branch=='U':
                        chw=normalize_image(rgb,[123.675,116.28,103.53],[58.395,57.12,57.375]).transpose(2,0,1)
                        padded,_=pad_normalized_image(np.ascontiguousarray(chw));x=torch.from_numpy(padded[None]).cuda().float()
                        q=model.eval_forward(x)[...,:h,:w].float().sigmoid()[0].cpu().numpy();del x,chw,padded
                    else:
                        a,ra=load_cached(cohort,'A',sid,3600);u,ru=load_cached(cohort,'U',sid,3600)
                        assert a.shape==u.shape==(4,h,w)
                        q=np.float32(.75)*a+np.float32(.25)*u
                        sources={'A':ra,'U':ru};del a,u
                assert q.shape==(4,h,w)
                digest=save_probability(out/(sid+'.npy'),q)
                r={'sid':sid,'cohort':cohort,'branch':branch,'hw':[h,w],'image_sha256':row['image_sha256'],
                   'probability_sha256':digest,'seconds_including_persistence':time.perf_counter()-t,
                   'GPU_peak_allocated_bytes':torch.cuda.max_memory_allocated() if branch!='P0' else 0,'sources':sources}
                if cohort=='canary':r['fixed_validation_counts']=counts(q>.5,row)
                write(record,r)
                del q,rgb
                if branch!='P0':torch.cuda.empty_cache()
                progress={'stage':'RUNNING','branch':branch,'cohort':cohort,'completed':i+1,'expected':len(rows),'elapsed_seconds':time.perf_counter()-start}
                write(out/'progress.json',progress);print(json.dumps(progress),flush=True)
            write(out/'completion.json',{'status':'COMPLETE','cohort':cohort,'branch':branch,'images':len(rows),'no_tuning':True})
    except BaseException as e:
        write(ROOT/'results/common_cache59'/('failure_'+branch+'.json'),{'error_type':type(e).__name__,'message':str(e),'formal_complete':False})
        raise

if __name__=='__main__':main()
