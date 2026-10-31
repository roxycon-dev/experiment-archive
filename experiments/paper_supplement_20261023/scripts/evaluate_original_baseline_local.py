"""Frozen historical baseline: same4060 cost, full original validation and native test masks.

No training, checkpoint selection, threshold search or changes to recovered source files.
Run only in the isolated legacy M2MRF Python environment.
"""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import cv2
import numpy as np
from PIL import Image
import torch

ROOT=Path(__file__).resolve().parents[1]
DATA=Path('D:/HRDecoder/data/DDR_split_613')
sys.path.insert(0,str(ROOT/'runtime_deps/legacy_profile'))
sys.path.insert(0,str(ROOT/'local_baseline_profile_source/baseline_v1/repo'))
from profile_retrained_baseline_local import sha,write,unwrap


def rows_for(cohort):
    if cohort=='idrid':
        return json.loads((ROOT/'evidence/external_holdout_manifest.json').read_text(encoding='utf-8'))['samples']
    split='val' if cohort=='val' else 'test'
    rows=json.loads((ROOT/'evidence/local_asset_audit.json').read_text(encoding='utf-8'))['data'][split]['samples']
    return [dict(r,image_path=str(DATA/'images'/split/(r['sid']+'.jpg')),
                 label_path=str(DATA/'labels'/split/(r['sid']+'.png'))) for r in rows]


def integer_counts(b,row,cohort):
    if cohort=='idrid':
        masks=[]
        for c in ('EX','HE','SE','MA'):
            r=row['masks'][c];assert sha(r['path'])==r['sha256']
            masks.append(np.asarray(Image.open(r['path']))>0)
        g=np.stack(masks);v=np.ones(g.shape[1:],bool)
    else:
        assert sha(row['label_path'])==row['label_sha256']
        label=np.asarray(Image.open(row['label_path']));g=np.stack([label==c for c in range(1,5)]);v=label!=255
    assert b.shape==g.shape
    return np.stack((np.count_nonzero(b&g&v,axis=(1,2)),
        np.count_nonzero(b&~g&v,axis=(1,2)),np.count_nonzero(~b&g&v,axis=(1,2))),axis=1).astype(np.int64)


def main():
    import mmcv,mmengine
    from mmcv import Config
    from mmseg.models import build_segmentor
    from mmseg.datasets import build_dataset
    identity=json.loads((ROOT/'evidence/recovered_original_baseline_identity.json').read_text(encoding='utf-8'))
    checkpoint=Path(identity['compact_checkpoint']);config=Path(identity['inference_config'])
    assert sha(checkpoint)==identity['compact_sha256'] and sha(config)==identity['inference_config_sha256']
    torch.set_num_threads(2);cv2.setNumThreads(2)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    cfg=Config.fromfile(str(config));cfg.model['pretrained']=None;cfg.model['backbone'].pop('pretrained',None)
    model=build_segmentor(copy.deepcopy(cfg.model))
    saved=torch.load(str(checkpoint),map_location='cpu')
    state={k[7:] if k.startswith('module.') else k:v for k,v in saved['state_dict'].items()}
    model.load_state_dict(state,strict=True);del saved,state
    assert model.scale_ratio==(.75,1.25) and model.crop_num==4
    write(ROOT/'evidence/recovered_original_baseline_strict_load.json',{
        'status':'PASS_STRICT_ALL_1844_STATE_KEYS','source_checkpoint_sha256':identity['source_checkpoint_sha256'],
        'model_state_tensors':len(model.state_dict()),'effective_scale_ratio':model.scale_ratio,
        'effective_crop_num':model.crop_num,'training_updates':0})
    model=model.cuda().float().eval().requires_grad_(False)
    old=ROOT/'results/full_system_profile_v1'
    full_manifest=json.loads((old/'manifest.json').read_text())
    assert torch.cuda.get_device_name()==full_manifest['hardware']
    cohort=json.loads((old/'cohort.json').read_text());assert sha(old/'cohort.json')==full_manifest['cohort_sha256']
    out=ROOT/'results/baseline_original_local';out.mkdir(exist_ok=True)
    manifest={'scope':'Recovered historical DDR_split613 original baseline32k; DDR-trained frozen IDRiD-derived external evaluation',
        'job':{'kind':'baseline_original','checkpoint_sha256':identity['source_checkpoint_sha256'],
               'weight_sha256':identity['compact_sha256'],'config_sha256':identity['inference_config_sha256'],
               'selected_node':32000,'selected_by':'Historical validation hook, not newly selected'},
        'torch':torch.__version__,'CUDA':torch.version.cuda,'GPU':torch.cuda.get_device_name(),
        'precision':'FP32','script_sha256':sha(Path(__file__)),
        'threshold':.5,'strict_greater':True,'channels':['EX','HE','SE','MA'],
        'inference':'Original2048 square whole inference, native-coordinate logit interpolation then sigmoid; no TTA',
        'runtime_limit':'Same inference source family and all keys strictly loaded; not a claim of bitwise historical CUDA replay'}
    if (out/'manifest.json').exists():assert json.loads((out/'manifest.json').read_text())==manifest
    else:write(out/'manifest.json',manifest)
    datasets={}
    def dataset_for(cohort):
        if cohort in datasets:return datasets[cohort]
        d=copy.deepcopy(cfg.data['val']);d.test_mode=True;d.ann_dir=None
        d.pipeline=copy.deepcopy(cfg.data['test']['pipeline'])
        if cohort=='idrid':d.data_root='D:/dataset/best_segmentation';d.img_dir='images/testing'
        else:d.data_root=str(DATA);d.img_dir='images/'+('val' if cohort=='val' else 'test')
        if 'split' in d:d.split=None
        data=build_dataset(d);lookup={Path(r['filename']).stem:i for i,r in enumerate(data.img_infos)}
        datasets[cohort]=(data,lookup);return data,lookup
    def forward(row,cohort):
        assert sha(row['image_path'])==row['image_sha256']
        dataset,lookup=dataset_for(cohort)
        torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();t=time.perf_counter()
        item=dataset[lookup[row['sid']]];image=unwrap(item['img']).unsqueeze(0).cuda();meta=[unwrap(item['img_metas'])]
        assert isinstance(meta[0],dict) and not meta[0]['flip']
        with torch.no_grad():q=model.inference(image,meta,rescale=True).float()[0].cpu().numpy()
        b=q>.5;torch.cuda.synchronize();seconds=time.perf_counter()-t
        assert q.dtype==np.float32 and np.isfinite(q).all() and 0<=q.min()<=q.max()<=1
        assert tuple(q.shape[1:])==tuple(meta[0]['ori_shape'][:2])
        peak=torch.cuda.max_memory_allocated();del q,image,item
        return b,seconds,peak
    profile=ROOT/'results/baseline_original_profile_local'
    if not (profile/'completion.json').exists():
        memory=[];stop=threading.Event()
        def sample_memory():
            while not stop.is_set():
                try:
                    raw=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],creationflags=0x08000000).decode()
                    memory.append([time.time(),float(raw.splitlines()[0])])
                except Exception:pass
                stop.wait(.2)
        sampler=threading.Thread(target=sample_memory,daemon=True);sampler.start();timings=[]
        try:
            for i,row in enumerate([cohort['samples'][0]]+cohort['samples']):
                b,seconds,peak=forward(row,'val')
                assert list(b.shape[1:])==row['hw']
                timings.append(dict(sid=row['sid'],warmup=i==0,hw=row['hw'],image_sha256=row['image_sha256'],
                    pipeline_to_native_mask_seconds=seconds,GPU_peak_allocated_bytes=peak,predicted_positive_pixels=b.sum((1,2)).tolist()))
                del b;torch.cuda.empty_cache()
        finally:stop.set();sampler.join(timeout=5)
        assert len(timings)==4 and memory
        formal=timings[1:];times=[r['pipeline_to_native_mask_seconds'] for r in formal]
        write(profile/'summary.json',{'status':'COMPLETE','hardware':torch.cuda.get_device_name(),'torch':torch.__version__,
            'mmcv':mmcv.__version__,'mmengine':mmengine.__version__,'numpy':np.__version__,'OpenCV':cv2.__version__,
            'sample_ids':[r['sid'] for r in formal],'registered_parameters':sum(p.numel() for p in model.parameters()),
            'inference_weight_artifact_bytes':checkpoint.stat().st_size,'checkpoint_artifact_bytes_with_optimizer':identity['source_bytes'],
            'warmup_images_excluded':1,'pipeline_to_native_binary_mask_seconds':{'per_image':times,'mean':float(np.mean(times)),
                'minimum':min(times),'maximum':max(times)},
            'max_single_process_allocated_peak_bytes':max(r['GPU_peak_allocated_bytes'] for r in formal),
            'sampled_whole_GPU_maximum_MiB':max(r[1] for r in memory),
            'sampled_whole_GPU_scope':'Whole GPU includes contexts/apps,200ms sample may miss peaks',
            'includes':'image read/transform,CPU-GPU transfers,whole forward,native alignment,sigmoid/threshold,return to CPU',
            'excludes':'model loading/startup,warmup,GT evaluation,result persistence',
            'timing_limit':'Three fixed val images, descriptive same-machine cost',
            'source_checkpoint_sha256':identity['source_checkpoint_sha256'],'source_config_sha256':identity['inference_config_sha256'],
            'profile_script_sha256':sha(Path(__file__)),'rows':timings})
        write(profile/'GPU_memory_samples.json',memory);write(profile/'completion.json',{'status':'COMPLETE','images':3,'same_hardware':True})
    began=time.perf_counter()
    try:
        for cohort in ('val','ddr','idrid'):
            folder=out/cohort
            for child in ('predictions','per_image'):(folder/child).mkdir(parents=True,exist_ok=True)
            rows=rows_for(cohort);all_counts=[]
            for i,row in enumerate(rows):
                sid=row['sid'];record=folder/'per_image'/(sid+'.json');pred=folder/'predictions'/(sid+'.npz')
                if record.exists():
                    r=json.loads(record.read_text());assert sha(pred)==r['mask_sha256'];all_counts.append(r['counts']['baseline']);continue
                b,seconds,peak=forward(row,cohort);counts=integer_counts(b,row,cohort)
                temp=pred.with_suffix('.npz.tmp')
                with temp.open('wb') as f:np.savez_compressed(f,baseline=b)
                temp.replace(pred)
                write(record,{'sid':sid,'cohort':cohort,'hw':list(b.shape[1:]),'image_sha256':row['image_sha256'],
                    'label_sha256':row.get('label_sha256'),'external_GT_sha256':{c:m['sha256'] for c,m in row.get('masks',{}).items()},
                    'counts':{'baseline':counts.tolist()},'mask_sha256':sha(pred),
                    'timing':{'pipeline_to_native_binary_mask_seconds':seconds,'GPU_peak_allocated_bytes':peak}})
                all_counts.append(counts.tolist());del b;torch.cuda.empty_cache()
                progress={'stage':'INFERENCE','cohort':cohort,'completed':i+1,'expected':len(rows),'elapsed_seconds':time.perf_counter()-began}
                write(out/'status.json',progress)
                if (i+1)%10==0:print(json.dumps(progress),flush=True)
            total=np.asarray(all_counts,np.int64).sum(0);ious=total[:,0]/np.maximum(total.sum(1),1)*100
            write(folder/'completion.json',{'status':'COMPLETE','images':len(rows),'global_counts':total.tolist(),
                'mIoU_pct':float(ious.mean()),'per_class_IoU_pct':ious.tolist()})
            print(json.dumps({'cohort_complete':cohort,'images':len(rows),'mIoU_pct':float(ious.mean())}),flush=True)
        write(out/'completion.json',{'status':'COMPLETE','validation_images':76,'ddr_images':227,'idrid_images':27,
            'original_checkpoint_recovered':True,'local_independent_recount_and_MA':'PENDING'})
        write(out/'status.json',{'stage':'COMPLETE'})
    except BaseException as e:
        write(out/'failure.json',{'error_type':type(e).__name__,'message':str(e)});raise

if __name__=='__main__':main()
