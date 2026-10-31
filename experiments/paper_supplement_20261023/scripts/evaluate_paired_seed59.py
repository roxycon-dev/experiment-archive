"""One fixed20k EMA on both frozen cohorts; uniform FP32/reference for all three seeds."""
import argparse
import json
from pathlib import Path
import sys
import time
import cv2
import numpy as np
import torch
from evaluation59_common import ROOT,EXP,sha,write,samples,image_rgb,counts,load_cached,save_probability

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--job',type=Path,required=True);args=parser.parse_args()
    job=json.loads(args.job.read_text(encoding='utf-8'));arm=job['arm'];seed=job['seed']
    assert arm in ('C','S') and seed in (42,2024,3407)
    assert job['completed']==20000 and job['EMA_completed']==20000
    assert job['parent_full_sha256']=='68d7b5821fab1210b3336cf2a63b9de893f24fc32876f5255d7d662084b5e728'
    assert job['runtime_protocol_sha256']==sha(ROOT/'plan/paired_evaluation_runtime.md')
    weight=Path(job['weight_path']);assert sha(weight)==job['weight_sha256']
    out=ROOT/'results/paired_seed59'/(arm+'_seed'+str(seed));out.mkdir(parents=True,exist_ok=True)
    identity={'job':job,'script_sha256':sha(Path(__file__)),'helper_sha256':sha(ROOT/'scripts/evaluation59_common.py'),
              'torch':torch.__version__,'CUDA':torch.version.cuda,'GPU':torch.cuda.get_device_name(),
              'precision':'FP32;reference MSDA;TF32 disabled','single_threshold':.5,'strict_greater':True,
              'system_formula':'.75*P0+.25*branch; P0=.75*A+.25*U','cohorts':{'ddr':227,'idrid':27}}
    if (out/'manifest.json').exists():assert json.loads((out/'manifest.json').read_text())==identity,'Cannot mix different inference identities'
    else:write(out/'manifest.json',identity)
    torch.set_num_threads(1);cv2.setNumThreads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    sys.path.insert(0,str(EXP/'HRDecoder_FinalDecision_20261005/execution'))
    sys.path.insert(0,str(EXP/'HRDecoder_Q_20261003/vendor/dinov3'))
    from dinov3.hub.backbones import dinov3_vitb16
    from qlib.model import Q
    from final_model import FinalModel
    from qlib.geometry import tiles,rgb_window,center_logits
    torch.manual_seed(seed)
    model=FinalModel(Q(dinov3_vitb16(pretrained=False)),arm,seed,backend='reference')
    saved=torch.load(weight,map_location='cpu',weights_only=False,mmap=True)
    assert saved.get('completed',saved.get('source_completed'))==20000
    model.load_state_dict(saved['ema'],strict=True);del saved
    model=model.cuda().eval().requires_grad_(False)
    began=time.perf_counter()
    try:
        for cohort in ('canary','ddr','idrid'):
            folder=out/cohort;folder.mkdir(exist_ok=True)
            cohort_rows=samples(cohort)
            for i,row in enumerate(cohort_rows):
                sid=row['sid'];record=folder/(sid+'.json');target=folder/(sid+'.npy')
                if record.exists():
                    r=json.loads(record.read_text());assert r['image_sha256']==row['image_sha256'] and sha(target)==r['probability_sha256'];continue
                rgb=image_rgb(row);h,w=rgb.shape[:2]
                q=np.empty((4,h,w),np.float32);t=time.perf_counter();torch.cuda.reset_peak_memory_stats()
                with torch.no_grad():
                    for tile in tiles(h,w):
                        x=torch.from_numpy(rgb_window(rgb,tile))[None].cuda()
                        z=center_logits(model(x))
                        assert z.dtype==torch.float32
                        q[:,tile.y:tile.y+tile.height,tile.x:tile.x+tile.width]=z.sigmoid()[0,:,:tile.height,:tile.width].cpu().numpy()
                torch.cuda.synchronize();forward_seconds=time.perf_counter()-t
                digest=save_probability(target,q)
                r={'sid':sid,'cohort':cohort,'arm':arm,'seed':seed,'image_sha256':row['image_sha256'],'hw':[h,w],
                   'probability_sha256':digest,'single_counts':counts(q>.5,row),'forward_seconds':forward_seconds,
                   'GPU_peak_allocated_bytes':torch.cuda.max_memory_allocated()}
                write(record,r)
                del q,rgb,x,z;torch.cuda.empty_cache()
                progress={'stage':'GPU_INFERENCE','arm':arm,'seed':seed,'cohort':cohort,'completed':i+1,'expected':len(cohort_rows),
                          'elapsed_seconds':time.perf_counter()-began}
                write(out/'status.json',progress);print(json.dumps(progress),flush=True)
        del model;torch.cuda.empty_cache()
        write(out/'status.json',{'stage':'CPU_FUSION','arm':arm,'seed':seed,'GPU_phase_complete':True})
        for cohort in ('ddr','idrid'):
            folder=out/cohort;predictions=folder/'predictions';predictions.mkdir(exist_ok=True)
            final_rows=folder/'per_image';final_rows.mkdir(exist_ok=True)
            rows=samples(cohort)
            for i,row in enumerate(rows):
                sid=row['sid'];record=final_rows/(sid+'.json');prediction=predictions/(sid+'.npz')
                if record.exists():
                    r=json.loads(record.read_text());assert sha(prediction)==r['mask_sha256'];continue
                r=json.loads((folder/(sid+'.json')).read_text());qpath=folder/(sid+'.npy')
                assert sha(qpath)==r['probability_sha256'];q=np.load(qpath,allow_pickle=False)
                p0,source=load_cached(cohort,'P0',sid,wait_seconds=43200)
                assert source['image_sha256']==row['image_sha256'] and p0.shape==q.shape
                single=q>.5;fusion=(np.float32(.75)*p0+np.float32(.25)*q)>.5
                recomputed=counts(single,row);assert recomputed==r['single_counts']
                temp=prediction.with_suffix('.npz.tmp')
                with temp.open('wb') as f:np.savez_compressed(f,single=single,fixed_fusion=fusion)
                temp.replace(prediction)
                write(record,dict(r,counts={'single':recomputed,'fixed_fusion':counts(fusion,row)},mask_sha256=sha(prediction),
                                  P0_sha256=source['probability_sha256'],label_sha256=row.get('label_sha256'),
                                  external_GT_sha256={k:m['sha256'] for k,m in row.get('masks',{}).items()}))
                del q,p0,single,fusion
                write(out/'status.json',{'stage':'CPU_FUSION','arm':arm,'seed':seed,'cohort':cohort,'completed':i+1,'expected':len(rows),'GPU_phase_complete':True})
            write(folder/'completion.json',{'status':'COMPLETE','images':len(rows),'threshold_or_blend_search':False})
        write(out/'completion.json',{'status':'COMPLETE','arm':arm,'seed':seed,'completed_training_update':20000,
                                    'EMA_completed':20000,'ddr_images':227,'idrid_images':27,'seconds':time.perf_counter()-began,
                                    'local_recovery_and_independent_analysis':'PENDING'})
        write(out/'status.json',{'stage':'COMPLETE','arm':arm,'seed':seed})
    except BaseException as e:
        write(out/'failure.json',{'error_type':type(e).__name__,'message':str(e),'all_formal_results_complete':False})
        write(out/'status.json',{'stage':'TECHNICAL_PAUSE','arm':arm,'seed':seed,'error':repr(e)})
        raise

if __name__=='__main__':main()
