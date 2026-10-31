"""Read-only shared NPY mappings; unchanged persistent FP32 tensors and fallback."""
import json
import os
from pathlib import Path
import numpy as np

ORIGINAL_LOAD = np.load

class RamArchive:
    def __init__(self,array): self.array=array
    def __enter__(self): return self
    def __exit__(self,*values): return False
    def __getitem__(self,key):
        if key != 'prob': raise KeyError(key)
        return self.array

def install_ram_loader(root):
    ready=json.loads((Path(root)/'reports/RAM_CACHE_READY.json').read_text())
    assert ready['status']=='READY'
    os.kill(ready['pid'],0)
    mapping=ready['mapping']
    def load(file,*args,**kwargs):
        entry=mapping.get(str(file)) if isinstance(file,(str,os.PathLike)) else None
        if entry is not None:
            try:
                if entry.get('format','npy') == 'npz':
                    return ORIGINAL_LOAD(entry['path'],allow_pickle=False)
                return RamArchive(ORIGINAL_LOAD(entry['path'],mmap_mode='r',allow_pickle=False))
            except (OSError,ValueError):
                pass  # Memory pressure or power cycle: original persistent tensor.
        return ORIGINAL_LOAD(file,*args,**kwargs)
    np.load=load
    return ready
