"""Spawned-worker RAM mapping and bitwise disk parity; three train maps only."""
import json
import multiprocessing as mp
from pathlib import Path
import numpy as np
import torch

HERE=Path(__file__).resolve().parent;ROOT=HERE.parent

def child(queue):
 from loader_ram_patch import worker_setup,ORIGINAL_LOAD
 worker_setup(0)
 ready=json.loads((ROOT/'reports/sacd/RAM_CACHE_READY.json').read_text())
 assert getattr(np.load,'_hrdecoder_RAM_pid',None)==ready['pid']
 rows=[]
 for member in ('A','U','S'):
  key=next(key for key,entry in ready['mapping'].items() if entry['member']==member)
  with ORIGINAL_LOAD(key,allow_pickle=False) as archive:reference=archive['prob']
  with np.load(key,allow_pickle=False) as archive:actual=archive['prob']
  assert np.array_equal(reference,actual) and reference.dtype==actual.dtype==np.float32
  rows.append({'member':member,'sid':ready['mapping'][key]['sid'],'shape':list(actual.shape),
               'format':ready['mapping'][key]['format'],'bitwise_equal':True})
  del reference,actual
 queue.put({'status':'PASS','torch':torch.__version__,'spawned_worker_RAM_pid':ready['pid'],
    'three_training_map_checks':rows,'full_data_cache_hash_rescan':False,'GPU_work':False})

if __name__=='__main__':
 ctx=mp.get_context('spawn');q=ctx.Queue();p=ctx.Process(target=child,args=(q,));p.start()
 report=q.get(timeout=90);p.join(timeout=10);assert p.exitcode==0
 (HERE/'WORKER_RAM_PASS.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps(report))
