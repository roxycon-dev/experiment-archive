"""The frozen three validation cases on the same4060, after the original val-selected checkpoint is locally backed up."""
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import zipfile
import cv2
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1];DATA=Path('D:/HRDecoder/data/DDR_split_613')

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()

def write(p,value):
    p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(value,indent=2),encoding='utf-8');os.replace(str(t),str(p))

def unwrap(x):
    from mmcv.parallel import DataContainer
    while isinstance(x,(DataContainer,list,tuple)):x=x.data if isinstance(x,DataContainer) else x[0]
    return x

def main():
    job=json.loads((ROOT/'evidence/terminal_backups/baseline_selection.json').read_text(encoding='utf-8'))
    name='baseline_best_u'+str(job['selected_node']);checkpoint=ROOT/'models'/(name+'.pth')
    backup=json.loads((ROOT/'evidence/terminal_backups'/(name+'.json')).read_text())
    assert backup['sha256']==job['checkpoint_sha256']==sha(checkpoint)
    preparation=json.loads((ROOT/'evidence/baseline_reproduction_preparation.json').read_text(encoding='utf-8'))
    bundle=ROOT/'transfer/baseline_sources.zip';assert sha(bundle)==preparation['bundle_sha256']
    source=ROOT/'local_baseline_profile_source'
    source.mkdir(exist_ok=True)
    with zipfile.ZipFile(bundle) as z:
        for m in z.infolist():
            p=(source/m.filename).resolve();p.relative_to(source.resolve())
            if m.is_dir():continue
            data=z.read(m)
            if p.exists():assert p.read_bytes()==data,'Isolated baseline source changed'
            else:p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
    config=source/'baseline_v1/baseline_reproduction_ddp2.py';assert sha(config)==job['config_sha256']
    sys.path.insert(0,str(ROOT/'runtime_deps/legacy_profile'))
    sys.path.insert(0,str(source/'baseline_v1/repo'))
    import mmcv,mmengine
    from mmcv import Config
    from mmseg.models import build_segmentor
    from mmseg.datasets import build_dataset
    torch.set_num_threads(2);cv2.setNumThreads(2)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    cfg=Config.fromfile(str(config));cfg.model['pretrained']=None;cfg.model['backbone'].pop('pretrained',None)
    model=build_segmentor(copy.deepcopy(cfg.model));saved=torch.load(str(checkpoint),map_location='cpu')
    state={k[7:] if k.startswith('module.') else k:v for k,v in saved['state_dict'].items()}
    model.load_state_dict(state,strict=True)
    compact=ROOT/'models/baseline_inference_only.pth'
    if not compact.exists():torch.save({'state_dict':state,'source_checkpoint_sha256':job['checkpoint_sha256'],'selected_update':job['selected_node']},compact)
    check=torch.load(str(compact),map_location='cpu');assert check['source_checkpoint_sha256']==job['checkpoint_sha256']
    assert set(check['state_dict'])==set(state) and all(torch.equal(v,check['state_dict'][k]) for k,v in state.items())
    write(ROOT/'evidence/baseline_inference_export.json',{'status':'PASS','all_state_tensors_bitwise_equal':True,'source_checkpoint_sha256':job['checkpoint_sha256'],
          'compact_sha256':sha(compact),'compact_bytes':compact.stat().st_size,'tensor_count':len(state)})
    del saved,state,check
    old=ROOT/'results/full_system_profile_v1';manifest=json.loads((old/'manifest.json').read_text())
    cohort=json.loads((old/'cohort.json').read_text());assert sha(old/'cohort.json')==manifest['cohort_sha256']
    assert torch.cuda.get_device_name()==manifest['hardware'],'Same hardware comparison required'
    out=ROOT/'results/baseline_profile_local';out.mkdir(exist_ok=True)
    assert not (out/'completion.json').exists(),'Completed profile must not be silently rerun'
    d=copy.deepcopy(cfg.data['val']);d.data_root=str(DATA);d.img_dir='images/val';d.ann_dir=None
    d.pipeline=copy.deepcopy(cfg.data['test']['pipeline']);d.test_mode=True
    if 'split' in d:d.split=None
    dataset=build_dataset(d);lookup={Path(r['filename']).stem:i for i,r in enumerate(dataset.img_infos)}
    model=model.cuda().float().eval().requires_grad_(False);memory=[];stop=threading.Event()
    def sample_memory():
        while not stop.is_set():
            try:
                raw=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],creationflags=0x08000000).decode()
                memory.append([time.time(),float(raw.splitlines()[0])])
            except Exception:pass
            stop.wait(.2)
    sampler=threading.Thread(target=sample_memory,daemon=True);sampler.start();rows=[]
    try:
        for i,row in enumerate([cohort['samples'][0]]+cohort['samples']):
            sid=row['sid'];warmup=i==0;assert sha(row['image_path'])==row['image_sha256']
            torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();t=time.perf_counter()
            item=dataset[lookup[sid]];image=unwrap(item['img']).unsqueeze(0).cuda();meta=[unwrap(item['img_metas'])]
            with torch.no_grad():q=model.inference(image,meta,rescale=True).float()[0].cpu().numpy()
            b=q>.5;torch.cuda.synchronize();seconds=time.perf_counter()-t
            assert list(q.shape[1:])==row['hw'] and q.dtype==np.float32 and np.isfinite(q).all()
            r={'sid':sid,'warmup':warmup,'hw':row['hw'],'image_sha256':row['image_sha256'],'pipeline_to_native_mask_seconds':seconds,
               'GPU_peak_allocated_bytes':torch.cuda.max_memory_allocated(),'predicted_positive_pixels':b.sum((1,2)).tolist()}
            rows.append(r);del q,b,image,item;torch.cuda.empty_cache();print(json.dumps(r),flush=True)
    finally:stop.set();sampler.join(timeout=5)
    assert len(rows)==4 and len(memory)>0
    formal=rows[1:];times=[r['pipeline_to_native_mask_seconds'] for r in formal]
    result={'status':'COMPLETE','hardware':torch.cuda.get_device_name(),'torch':torch.__version__,'mmcv':mmcv.__version__,'mmengine':mmengine.__version__,
            'numpy':np.__version__,'OpenCV':cv2.__version__,'sample_ids':[r['sid'] for r in formal],
            'registered_parameters':sum(p.numel() for p in model.parameters()),'inference_weight_artifact_bytes':compact.stat().st_size,
            'checkpoint_artifact_bytes_with_optimizer':checkpoint.stat().st_size,'warmup_images_excluded':1,
            'pipeline_to_native_binary_mask_seconds':{'per_image':times,'mean':float(np.mean(times)),'minimum':min(times),'maximum':max(times)},
            'max_single_process_allocated_peak_bytes':max(r['GPU_peak_allocated_bytes'] for r in formal),
            'sampled_whole_GPU_maximum_MiB':max(r[1] for r in memory),'sampled_whole_GPU_scope':'Includes contexts, driver and other apps;200ms sample can miss instantaneous maxima',
            'includes':'image read/transform,CPU-GPU transfers,whole forward,original-coordinate alignment,sigmoid/threshold,return to CPU',
            'excludes':'model loading/startup,warmup,GT evaluation,result persistence','timing_limit':'Three fixed validation images only; descriptive same-machine cost, not a hardware-independent benchmark',
            'source_checkpoint_sha256':job['checkpoint_sha256'],'source_config_sha256':job['config_sha256'],'profile_script_sha256':sha(Path(__file__)),'rows':rows}
    write(out/'summary.json',result);write(out/'GPU_memory_samples.json',memory)
    write(out/'completion.json',{'status':'COMPLETE','images':3,'same_hardware':True})

if __name__=='__main__':main()
