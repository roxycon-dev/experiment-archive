"""CPU-only input throughput; no model, backward, optimizer or test access."""
import json,os,sys,time
from pathlib import Path
BASE=Path('/root/autodl-tmp/nativepyramid_sp_v3_20261016')
sys.path.insert(0,str(BASE/'code'))
import cv2
import torch
from torch.utils.data import DataLoader
from data import SegmentationData,ResumableBatches,worker_setup

if __name__=='__main__':
    torch.set_num_threads(1);cv2.setNumThreads(1)
    dataset=SegmentationData(BASE/'data',BASE/'code/preparation_20261016/data_identity.json','train',2048)
    results=[]
    for workers in (4,8):
        loader=DataLoader(dataset,batch_sampler=ResumableBatches(454,0,20,42),num_workers=workers,
                          worker_init_fn=worker_setup,prefetch_factor=1,persistent_workers=False,
                          pin_memory=False,generator=torch.Generator().manual_seed(42))
        t0=time.monotonic();last=t0;latencies=[]
        for index,batch in enumerate(loader):
            current=time.monotonic();latencies.append(current-last);last=current
            if not bool((batch['label']!=255).any()):raise RuntimeError('Unexpected all-ignore benchmark batch')
            del batch
        result={'workers':workers,'batches':20,'first_batch_seconds':latencies[0],
                'steady_seconds_per_batch':sum(latencies[4:])/len(latencies[4:]),
                'total_seconds':time.monotonic()-t0,'global_batch':2,'canvas':2048,
                'pinned_memory':False,'GPU_used':False,'SGD_updates':0,'test_accessed':False}
        results.append(result);print(json.dumps(result),flush=True)
        del loader
    (BASE/'receipts/input_benchmark.json').write_text(json.dumps(results,indent=2)+'\n')
