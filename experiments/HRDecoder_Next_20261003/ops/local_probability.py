"""Read archived exact FP32 probabilities without changing original remote metadata."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
LOCAL=ROOT/'server_results'
PREFIXES={
 '/dev/shm/hrdecoder_next_20261003/work/':'work/',
 '/root/autodl-tmp/HRDecoder_Next_20261003/work/':'work/',
 '/root/autodl-tmp/HRDecoder_Next_20261003/cold_storage/G/':'work/G/',
}

def local_cache(directory):
 text=str(directory).replace('\\','/')
 for prefix,replacement in PREFIXES.items():
  if text.startswith(prefix):directory=LOCAL/(replacement+text[len(prefix):]);break
 else:
  p=Path(directory)
  directory=p if p.is_absolute() else LOCAL/p
 directory=Path(directory).resolve()
 if LOCAL.resolve() not in directory.parents:raise ValueError('Cache must be inside this local archive')
 return directory

def load_probability(directory,sid,verify=False,_active=None):
 if Path(sid).name!=sid or '/' in sid or '\\' in sid:raise ValueError('Image stem required')
 directory=local_cache(directory);active=set() if _active is None else _active
 if directory in active:raise ValueError('Cyclic lazy probability components')
 active=active|{directory};index=directory/'index.json'
 if index.exists():
  metadata=json.loads(index.read_text(encoding='utf-8-sig'))
  if metadata.get('storage') in ['lazy_weighted_lossless_float32','lazy_fixed_half_lossless_float32_mean']:
   a,b=metadata['components_cache']
   left=load_probability(a,sid,verify,active);right=load_probability(b,sid,verify,active)
   if metadata['storage']=='lazy_fixed_half_lossless_float32_mean':result=(left+right)*.5
   else:
    weight=float(metadata['weight']);result=weight*left+(1-weight)*right
   if result.dtype!=np.float32:raise ValueError('Unexpected fusion precision')
   return result
 path=directory/(sid+'.npz')
 if not path.exists():path=directory/(sid+'.npy')
 if verify:
  manifest=json.loads((ROOT/'reports/mirror_manifest.json').read_text(encoding='utf-8-sig'))
  rel=path.relative_to(LOCAL).as_posix();expected=manifest[rel]
  h=hashlib.sha256()
  with path.open('rb') as f:
   for chunk in iter(lambda:f.read(4*1024**2),b''):h.update(chunk)
  if h.hexdigest()!=expected['sha256'] or path.stat().st_size!=expected['size']:
   raise ValueError('Archived probability differs from its verified backup receipt')
 if path.suffix=='.npz':
  with np.load(path,allow_pickle=False) as z:result=z['prob']
 else:result=np.load(path,mmap_mode='r',allow_pickle=False)
 if result.dtype!=np.float32 or result.ndim!=3 or result.shape[0]!=4:
  raise ValueError('Probability contract is four channels in exact float32')
 return result

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--cache',required=True);ap.add_argument('--sid',required=True)
 ap.add_argument('--verify',action='store_true');a=ap.parse_args()
 p=load_probability(a.cache,a.sid,a.verify)
 print(json.dumps(dict(shape=list(p.shape),dtype=str(p.dtype),min=float(p.min()),max=float(p.max()),
  above_05=(p>.5).sum(axis=(1,2)).tolist(),verify=a.verify,source='local archive; original metadata untouched')))
if __name__=='__main__':main()
