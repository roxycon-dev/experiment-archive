"""Lossless fp32 cache, fast level-1 deflate to fit the 50 GB data disk."""
from pathlib import Path
import json
import zipfile
import numpy as np

def save_probability(path, probability):
    path=Path(path).with_suffix('.npz')
    tmp=path.with_suffix('.tmp')
    if probability.dtype!=np.float32:raise ValueError('fp32 probability required')
    with zipfile.ZipFile(tmp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        with z.open('prob.npy','w',force_zip64=True) as f:
            np.lib.format.write_array(f,np.ascontiguousarray(probability),allow_pickle=False)
    tmp.replace(path)
    return path

def load_probability(directory,sid):
    directory=Path(directory)
    index=directory/'index.json'
    if index.exists():
        metadata=json.loads(index.read_text())
        if metadata.get('storage')=='lazy_fixed_half_lossless_float32_mean':
            a,b=metadata['components_cache']
            return (load_probability(a,sid)+load_probability(b,sid))*.5
    p=directory/f'{sid}.npz'
    if p.exists():
        with np.load(p,allow_pickle=False) as z:return z['prob']
    return np.load(directory/f'{sid}.npy',mmap_mode='r',allow_pickle=False)
