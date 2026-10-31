"""Identical real-cache replay and CPU throughput; no optimizer/GPU updates."""
import gc
import json
import os
from pathlib import Path
import statistics
import sys
import time
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(ROOT/'vendor/champion_repo'))
import cv2
import torch
from data import TrainCacheDataset
from train import ResumableBatches,write_json
from ram_loader import install_ram_loader

cv2.setNumThreads(1); torch.set_num_threads(1)
dataset=TrainCacheDataset(ROOT,Path('/root/HRDecoder/data/DDR_split_613'))
ready=json.loads((ROOT/'reports/RAM_CACHE_READY.json').read_text())
assert ready['selected_images']==454 and len(ready['mapping'])==1362
check_index=min(range(len(dataset)),key=lambda i: dataset.rows[i]['hw'][0]*dataset.rows[i]['hw'][1])
reference=dataset[(check_index,20261009)]
install_ram_loader(ROOT)
cached=dataset[(check_index,20261009)]
for name in ('image','label','valid','A','U','S'):
    assert torch.equal(reference[name],cached[name]), name
del reference,cached; gc.collect()
report_path=ROOT/'reports/RAM_PIPELINE_BENCHMARK_PASS.json'
rows=[] if os.environ.get('HRDECODER_BENCH_FRESH') else (json.loads(report_path.read_text())['CPU_benchmarks'] if os.environ.get('HRDECODER_BENCH_WORKERS') and report_path.exists() else [])
worker_counts=[int(value) for value in os.environ.get('HRDECODER_BENCH_WORKERS','2,4,6').split(',')]
for workers in worker_counts:
    generator=torch.Generator().manual_seed(42)
    start=int(os.environ.get('HRDECODER_BENCH_START','1000'))
    sampler=ResumableBatches(len(dataset),start=start,stop=start+40)
    def setup(worker): cv2.setNumThreads(1); torch.set_num_threads(1)
    loader=torch.utils.data.DataLoader(dataset,batch_sampler=sampler,num_workers=workers,
        prefetch_factor=1,pin_memory=False,worker_init_fn=setup,generator=generator)
    iterator=iter(loader); times=[]; memory=[]
    for unused in range(40):
        started=time.monotonic(); batch=next(iterator)
        times.append(time.monotonic()-started)
        memory.append(int(Path('/sys/fs/cgroup/memory.current').read_text()))
        del batch
    del iterator,loader; gc.collect()
    measured=times[5:]
    row={'workers':workers,'prefetch_factor':1,'measured_batches':len(measured),
        'mean_CPU_batch_seconds':statistics.mean(measured),
        'median_CPU_batch_seconds':statistics.median(measured),
        'p90_CPU_batch_seconds':sorted(measured)[int(.9*(len(measured)-1))],
        'peak_cgroup_memory_bytes':max(memory),'GPU_or_optimizer_used':False}
    rows.append(row); print(json.dumps(row),flush=True)
    time.sleep(2)
limit=int(Path('/sys/fs/cgroup/memory.max').read_text())
ready=json.loads((ROOT/'reports/RAM_CACHE_READY.json').read_text())
os.kill(ready['pid'],0)
assert len(ready['mapping'])==1362,'Memory pressure reduced RAM cache during CPU benchmark'
safe=[row for row in rows if row['peak_cgroup_memory_bytes'] < limit-8*1024**3]
assert safe,'No safe loader configuration for full RAM cache'
selected=min(safe,key=lambda row:row['mean_CPU_batch_seconds'])
result={'status':'PASS','full_shared_FP32_cache_images':454,'cache_bytes':ready['bytes'],
    'same_real_2048_image_label_valid_A_U_S_bits':True,'CPU_benchmarks':rows,
    'selected_workers':selected['workers'],'prefetch_factor':1,
    'memory_safety_headroom_bytes':limit-selected['peak_cgroup_memory_bytes'],
    'not_a_GPU_optimality_proof':True}
write_json(ROOT/'reports/RAM_PIPELINE_BENCHMARK_PASS.json',result)
write_json(ROOT/'reports/performance_config.json',{'workers':selected['workers'],'prefetch':1,'full_shared_RAM':True})
print(json.dumps(result),flush=True)
