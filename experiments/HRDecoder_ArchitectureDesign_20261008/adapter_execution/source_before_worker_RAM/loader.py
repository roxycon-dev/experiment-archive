"""Audited exposure order, shared-cache reads and one future CUDA batch."""
import json
import os
from pathlib import Path
import queue
import threading
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler

ORIGINAL_LOAD = np.load
class RamArchive:
    def __init__(self,array): self.array=array
    def __enter__(self): return self
    def __exit__(self,*args): return False
    def __getitem__(self,key):
        assert key=='prob'
        return self.array

def install_ram(root):
    path=Path(root)/'reports/sacd/RAM_CACHE_READY.json'
    ready=json.loads(path.read_text()); assert ready['status']=='READY'
    assert b'sacd_execution/ram_cache_owner.py' in Path('/proc/%d/cmdline'%ready['pid']).read_bytes()
    mapping=ready['mapping']
    def load(file,*args,**kwargs):
        entry=mapping.get(str(file)) if isinstance(file,(str,os.PathLike)) else None
        if entry is not None:
            try:
                if entry['format']=='npz': return ORIGINAL_LOAD(entry['path'],allow_pickle=False)
                return RamArchive(ORIGINAL_LOAD(entry['path'],mmap_mode='r',allow_pickle=False))
            except (OSError,ValueError): pass
        return ORIGINAL_LOAD(file,*args,**kwargs)
    np.load=load
    return ready

class ResumableBatches(Sampler):
    def __init__(self,n,start=0,stop=40000,seed=42):
        assert n%2==0
        self.n,self.start,self.stop,self.seed=n,start,stop,seed
    def __len__(self): return self.stop-self.start
    def __iter__(self):
        previous_epoch=None
        for update in range(self.start,self.stop):
            epoch=update*2//self.n
            if epoch!=previous_epoch:
                order=torch.randperm(self.n,generator=torch.Generator().manual_seed(self.seed+epoch)).tolist()
                previous_epoch=epoch
            offset=update*2%self.n
            yield [(order[offset+j],self.seed+update*2+j) for j in range(2)]

def worker_setup(worker_id):
    import cv2
    cv2.setNumThreads(1); torch.set_num_threads(1)

def move(batch):
    return {k:v.cuda(non_blocking=True) if torch.is_tensor(v) else v for k,v in batch.items()}

class CUDAPrefetch:
    def __init__(self,dataset,sampler,workers=8):
        self.loader=DataLoader(dataset,batch_sampler=sampler,num_workers=workers,
            prefetch_factor=1,persistent_workers=True,pin_memory=True,
            worker_init_fn=worker_setup,generator=torch.Generator().manual_seed(42),
            multiprocessing_context='spawn')
        self.wait_times=[]
    def __iter__(self):
        iterator=iter(self.loader)
        pending=queue.Queue(maxsize=1); available=threading.Semaphore(1)
        stream=torch.cuda.Stream()
        def producer():
            try:
                for cpu in iterator:
                    available.acquire()
                    with torch.cuda.stream(stream):
                        gpu=move(cpu); event=torch.cuda.Event(); event.record(stream)
                    pending.put((gpu,event,cpu))
                pending.put(None)
            except BaseException as error: pending.put(error)
        thread=threading.Thread(target=producer,daemon=True); thread.start()
        first=True
        while True:
            started=time.monotonic(); item=pending.get(); self.wait_times.append(time.monotonic()-started)
            if item is None: thread.join(); return
            if isinstance(item,BaseException): raise item
            gpu,event,cpu=item; available.release()
            current=torch.cuda.current_stream(); current.wait_event(event)
            for value in gpu.values():
                if torch.is_tensor(value): value.record_stream(current)
            if first:
                for name,value in gpu.items():
                    if torch.is_tensor(value): assert torch.equal(value.cpu(),cpu[name]),name
                first=False
            yield gpu
            del item,cpu
