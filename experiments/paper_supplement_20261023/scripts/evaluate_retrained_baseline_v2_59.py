"""Validation-selected new baseline, original test pipeline and native-coordinate counts."""
import argparse
import copy
import json
from pathlib import Path
import sys
import time
import cv2
import numpy as np
import torch
from evaluation59_common import ROOT,EXP,DATA,sha,write,samples,counts

TASK=EXP.parent;BASE=TASK/'baseline_v2'

def unwrap(value):
    from mmcv.parallel import DataContainer
    while isinstance(value,(DataContainer,list,tuple)):
        value=value.data if isinstance(value,DataContainer) else value[0]
    return value

def main():
    p=argparse.ArgumentParser();p.add_argument('--job',type=Path,required=True);args=p.parse_args()
    job=json.loads(args.job.read_text());checkpoint=Path(job['checkpoint_path'])
    assert job['training_finished_at']==40000 and job['selected_by']=='Original validation mIoU hook, never test'
    assert sha(checkpoint)==job['checkpoint_sha256']
    assert sha(BASE/'baseline_reproduction_ddp2.py')==job['config_sha256']
    out=ROOT/'results/baseline_reconstructed_v2_59';out.mkdir(parents=True,exist_ok=True)
    manifest={'scope':'corrected_baseline_v2_restored_original_scale_ratio_not_recovered_historical42.20','job':job,'torch':torch.__version__,
              'CUDA':torch.version.cuda,'GPU':torch.cuda.get_device_name(),'script_sha256':sha(Path(__file__)),
              'helper_sha256':sha(ROOT/'scripts/evaluation59_common.py'),'precision':'FP32',
              'threshold':.5,'strict_greater':True,'channels':['EX','HE','SE','MA'],
              'protocol_sha256':sha(ROOT/'plan/paired_evaluation_runtime.md'),
              'inference':'Original2048 square test_pipeline, single whole inference, sigmoid after native-coordinate logit interpolation; no TTA'}
    if (out/'manifest.json').exists():assert json.loads((out/'manifest.json').read_text())==manifest
    else:write(out/'manifest.json',manifest)
    torch.set_num_threads(1);cv2.setNumThreads(1)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    sys.path.insert(0,str(BASE/'repo'))
    from mmcv import Config
    from mmseg.models import build_segmentor
    from mmseg.datasets import build_dataset
    cfg=Config.fromfile(str(BASE/'baseline_reproduction_ddp2.py'))
    cfg.model['pretrained']=None;cfg.model['backbone'].pop('pretrained',None)
    model=build_segmentor(copy.deepcopy(cfg.model))
    saved=torch.load(str(checkpoint),map_location='cpu')
    selected=int(checkpoint.stem.split('_')[-1]);assert saved['meta']['iter']==selected-1
    assert Path(saved['meta']['hook_msgs']['best_ckpt']).name==checkpoint.name
    state={k.removeprefix('module.') if hasattr(k,'removeprefix') else (k[7:] if k.startswith('module.') else k):v for k,v in saved['state_dict'].items()}
    model.load_state_dict(state,strict=True);del state,saved
    model=model.cuda().float().eval().requires_grad_(False)
    write(out/'parameters.json',{'registered_parameters':sum(p.numel() for p in model.parameters()),
                               'checkpoint_artifact_bytes_includes_optimizer':checkpoint.stat().st_size})
    began=time.perf_counter()
    try:
        for cohort in ('canary','ddr','idrid'):
            folder=out/cohort;folder.mkdir(exist_ok=True)
            for name in ('predictions','per_image'):(folder/name).mkdir(exist_ok=True)
            d=copy.deepcopy(cfg.data['val']);d.data_root=str(EXP/'data/idrid') if cohort=='idrid' else str(DATA)
            d.img_dir='images/testing' if cohort=='idrid' else 'images/'+('val' if cohort=='canary' else 'test')
            d.ann_dir=None;d.pipeline=copy.deepcopy(cfg.data['test']['pipeline']);d.test_mode=True
            if 'split' in d:d.split=None
            dataset=build_dataset(d);lookup={Path(r['filename']).stem:i for i,r in enumerate(dataset.img_infos)}
            rows=samples(cohort)
            for i,row in enumerate(rows):
                sid=row['sid'];record=folder/'per_image'/(sid+'.json');pred=folder/'predictions'/(sid+'.npz')
                if record.exists():
                    r=json.loads(record.read_text());assert r['image_sha256']==row['image_sha256'] and sha(pred)==r['mask_sha256'];continue
                assert sha(row['image_path'])==row['image_sha256']
                torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();t=time.perf_counter()
                item=dataset[lookup[sid]];image=unwrap(item['img']).unsqueeze(0).cuda();meta=[unwrap(item['img_metas'])]
                assert isinstance(meta[0],dict) and not meta[0]['flip']
                h,w=map(int,meta[0]['ori_shape'][:2])
                with torch.no_grad():q=model.inference(image,meta,rescale=True).float()[0].cpu().numpy()
                b=q>.5;torch.cuda.synchronize();seconds=time.perf_counter()-t
                assert q.dtype==np.float32 and q.shape==(4,h,w) and np.isfinite(q).all() and 0<=q.min()<=q.max()<=1
                integer_counts=counts(b,row)
                temp=pred.with_suffix('.npz.tmp')
                with temp.open('wb') as f:np.savez_compressed(f,baseline=b)
                temp.replace(pred)
                write(record,{'sid':sid,'cohort':cohort,'hw':[h,w],'image_sha256':row['image_sha256'],
                              'label_sha256':row.get('label_sha256'),'external_GT_sha256':{k:m['sha256'] for k,m in row.get('masks',{}).items()},
                              'counts':{'baseline':integer_counts},'mask_sha256':sha(pred),
                              'timing':{'pipeline_to_native_binary_mask_seconds':seconds,'GPU_peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                                        'includes':'image read/transform,CPU-GPU transfer,forward,original-coordinate alignment,sigmoid,threshold,return to CPU',
                                        'excludes':'model startup,ground-truth reading and evaluation,result persistence'},'warmup':cohort=='canary'})
                del q,b,image,item;torch.cuda.empty_cache()
                progress={'stage':'INFERENCE','cohort':cohort,'completed':i+1,'expected':len(rows),'elapsed_seconds':time.perf_counter()-began}
                write(out/'status.json',progress);print(json.dumps(progress),flush=True)
            write(folder/'completion.json',{'status':'COMPLETE','images':len(rows),'canary':cohort=='canary'})
        write(out/'completion.json',{'status':'COMPLETE','ddr_images':227,'idrid_images':27,'warmup_images_excluded':1,
                                    'checkpoint_selected_before_test':True,'same_hardware4060_profile_and_local_recovery':'PENDING'})
        write(out/'status.json',{'stage':'COMPLETE'})
    except BaseException as e:
        write(out/'failure.json',{'error_type':type(e).__name__,'message':str(e),'all_formal_results_complete':False})
        write(out/'status.json',{'stage':'TECHNICAL_PAUSE','error':repr(e)});raise

if __name__=='__main__':main()
