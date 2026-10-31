"""Independent SHA, native geometry, FP32 algebra and GPU integer audit of the shared cache."""
from pathlib import Path
import json
import time
import numpy as np
import torch
from evaluation59_common import ROOT,sha,write,samples,truth,cache_dir

torch.set_num_threads(1)
started=time.perf_counter();rows=[]
for cohort in ('canary','ddr','idrid'):
    for branch in ('A','U','P0'):
        assert json.loads((cache_dir(cohort,branch)/'completion.json').read_text())['status']=='COMPLETE'
    for row in samples(cohort):
        sid=row['sid'];arrays={};hashes={}
        for branch in ('A','U','P0'):
            folder=cache_dir(cohort,branch);p=folder/(sid+'.npy')
            record=json.loads(p.with_suffix('.json').read_text());digest=sha(p)
            assert digest==record['probability_sha256'] and record['image_sha256']==row['image_sha256']
            a=np.load(p,allow_pickle=False,mmap_mode='r');assert a.dtype==np.float32 and a.ndim==3 and a.shape[0]==4
            arrays[branch]=a;hashes[branch]=digest
        expected=np.float32(.75)*arrays['A']+np.float32(.25)*arrays['U']
        assert np.array_equal(expected,arrays['P0']),('P0 algebra',cohort,sid)
        gt,valid=truth(row);assert expected.shape==gt.shape
        q=torch.tensor(np.asarray(arrays['P0']),device='cuda')
        assert torch.isfinite(q).all().item() and q.min().item()>=0 and q.max().item()<=1
        g=torch.from_numpy(gt).cuda();v=torch.from_numpy(valid).cuda();b=q>.5
        gpu=torch.stack(((b&g&v).sum((1,2)),(b&~g&v).sum((1,2)),(~b&g&v).sum((1,2))),dim=1).cpu().numpy()
        cb=expected>.5
        cpu=np.stack((np.count_nonzero(cb&gt&valid,axis=(1,2)),np.count_nonzero(cb&~gt&valid,axis=(1,2)),np.count_nonzero(~cb&gt&valid,axis=(1,2))),axis=1)
        assert np.array_equal(gpu,cpu),('Independent GPU/CPU counts',cohort,sid)
        rows.append({'cohort':cohort,'sid':sid,'probability_SHA':hashes,'P0_integer_counts':gpu.tolist()})
        del arrays,expected,gt,valid,q,g,v,b,cb,cpu,gpu;torch.cuda.empty_cache()
        if len(rows)%25==0:print(json.dumps({'audited_images':len(rows),'seconds':time.perf_counter()-started}),flush=True)
write(ROOT/'evidence/shared_cache59_integrity.json',{'status':'PASS','images':len(rows),'probability_files':len(rows)*3,
     'hashes_shapes_finite_ranges':'PASS','P0_algebra_bitwise_on_CPU':'PASS','independent_GPU_CPU_integer_counts':'PASS',
     'GPU':torch.cuda.get_device_name(),'seconds':time.perf_counter()-started,'rows':rows})
print(json.dumps({'status':'PASS','images':len(rows),'probability_files':len(rows)*3,'seconds':time.perf_counter()-started}),flush=True)
