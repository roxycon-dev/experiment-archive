"""Execution-only acceleration; sampling, tensors and optimizer mathematics stay fixed."""
from concurrent.futures import ThreadPoolExecutor
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from qlib.data import DDR, WindowDataset
from qlib.geometry import center_logits
from qlib.loss import masked_loss


class MemorySource:
    def __init__(self, original, values):
        self.source, self.ids, self.values = original.source, original.ids, values

    def load(self, sid):
        return self.values[sid]


def cache_sources(datasets, config, progress=None):
    """Decode once in the parent; Linux fork workers share read-only array pages."""
    from PIL import Image
    tasks=[]; expected=0
    for source, dataset in datasets.items():
        if isinstance(dataset, MemorySource):
            continue
        if isinstance(dataset, DDR):
            dataset.cache.clear(); dataset.cache_bytes=0; dataset.cache_limit=0
        for sid in dataset.ids:
            path=dataset.images[sid] if isinstance(dataset, DDR) else dataset.rows[sid]['image']
            with Image.open(path) as image:
                w,h=image.size
            expected+=w*h*11
            tasks.append((source,sid,dataset))
    if expected > config['decoded_cache_limit_bytes']:
        raise MemoryError('Decoded training cache exceeds its explicit memory budget')
    started=time.perf_counter(); values={source:{} for source in datasets}; actual=0
    def decode(task):
        source,sid,dataset=task
        result=dataset.load(sid)
        for a in result: a.flags.writeable=False
        return source,sid,result
    with ThreadPoolExecutor(max_workers=config['decode_threads']) as pool:
        for n,(source,sid,result) in enumerate(pool.map(decode,tasks),1):
            values[source][sid]=result; actual+=sum(a.nbytes for a in result)
            if progress and (n%50==0 or n==len(tasks)):
                progress({'decoded_images':n,'expected_images':len(tasks),'decoded_bytes':actual})
    cached={source:(dataset if isinstance(dataset,MemorySource) else MemorySource(dataset,values[source]))
            for source,dataset in datasets.items()}
    return cached, {'decoded_bytes':actual,'images':len(tasks),'seconds':time.perf_counter()-started,
                    'storage':'read-only parent RAM shared by fork workers; no data-disk copy',
                    'estimated_bytes':expected}


class AuditedWindows(WindowDataset):
    def __getitem__(self,index):
        x,g,v=super().__getitem__(index)
        # Binary sums are exact integers below2**24, identical to the original
        # main-process FP32 sums. Moving them to workers removes GPU idle gaps.
        legal=v.numpy().sum(axis=(1,2),dtype=np.float32)
        positive=(g.numpy()*v.numpy()).sum(axis=(1,2),dtype=np.float32)
        return x,g,v,torch.from_numpy(legal),torch.from_numpy(positive)


def worker_init(_):
    import cv2
    cv2.setNumThreads(1)
    torch.set_num_threads(1)


def loader_for(datasets,rows,start,end,resources,seed,config):
    return iter(DataLoader(AuditedWindows(datasets,rows,start*8,end*8),
        batch_size=resources['micro_batch'],shuffle=False,num_workers=config['workers'],
        pin_memory=True,generator=torch.Generator().manual_seed(seed+333),
        prefetch_factor=config['prefetch_factor'],persistent_workers=True,
        multiprocessing_context='fork',worker_init_fn=worker_init))


def update(model,optimizer,scheduler,loader,resources,completed,valid_audit=None,stream_rows=None):
    model.set_update(completed); optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize(); started=time.perf_counter(); waiting=0.; losses=[]
    for micro_index in range(resources['accumulation']):
        before=time.perf_counter(); x,g,v,legal,foreground=next(loader)
        waiting+=time.perf_counter()-before
        if valid_audit is not None:
            for index,(vv,gg) in enumerate(zip(legal.tolist(),foreground.tolist())):
                slot=micro_index*resources['micro_batch']+index
                source=stream_rows[completed*8+slot]['source']
                audit=valid_audit.setdefault(source,{'valid_pixels':[0]*4,'positive_pixels':[0]*4,'windows':0})
                audit['windows']+=1
                audit['valid_pixels']=[a+b for a,b in zip(audit['valid_pixels'],vv)]
                audit['positive_pixels']=[a+b for a,b in zip(audit['positive_pixels'],gg)]
        x,g,v=(t.to('cuda',non_blocking=True) for t in (x,g,v))
        with torch.autocast('cuda',dtype=torch.bfloat16):
            z=center_logits(model(x))
        loss=masked_loss(z,g,v)/resources['accumulation']
        losses.append(loss.detach()); loss.backward()
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
    if not torch.isfinite(norm):
        raise FloatingPointError('Nonfinite gradient; no skipped update')
    optimizer.step(); scheduler.step(); torch.cuda.synchronize()
    # Keep the original Python float summation order while transferring once.
    loss_value=sum(torch.stack(losses).cpu().tolist())
    return {'loss':loss_value,'seconds_per_update':time.perf_counter()-started,
            'data_wait_seconds':waiting,'grad_norm':float(norm)}
