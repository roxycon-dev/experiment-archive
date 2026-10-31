"""Independent post-inference SHA/range checks and GPU integer recount of all output masks."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
run = ROOT / 'results/idrid_holdout_v1'
cache = ROOT / 'results/remote_probability_cache'
methods = ['full', 'no_h4', 'no_convnext', 'matched_c20']
totals = {k: np.zeros((4,3), dtype=np.int64) for k in methods}
started = time.monotonic()
cohort = json.loads((run/'cohort.json').read_text())
checks = []

def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8*1024**2), b''):
            digest.update(chunk)
    return digest.hexdigest()

with torch.inference_mode():
    for row in cohort['samples']:
        sid = row['sid']
        rec = json.loads((run/'per_image'/(sid+'.json')).read_text())
        for arm in ['S','C','U']:
            path = cache/arm/(sid+'.npy')
            meta = json.loads(path.with_suffix('.json').read_text())
            assert meta['sid'] == sid and meta['arm'] == arm
            assert meta['image_sha256'] == row['image_sha256']
            assert sha(path) == meta['probability_sha256']
            arr = np.load(path, allow_pickle=False)
            assert arr.dtype == np.float32 and arr.shape == (4,*row['hw'])
            t = torch.from_numpy(arr).cuda()
            lo, hi = float(t.min()), float(t.max())
            assert torch.isfinite(t).all().item() and 0 <= lo <= hi <= 1
            checks.append({'sid':sid,'arm':arm,'sha256':meta['probability_sha256'],
                           'min':lo,'max':hi,'finite':True})
            del t, arr
        masks = []
        for cls in ['EX','HE','SE','MA']:
            mr = row['masks'][cls]
            path = Path(mr['path'])
            assert sha(path) == mr['sha256']
            masks.append(np.asarray(Image.open(path)) > 0)
        gt = torch.from_numpy(np.stack(masks)).cuda()
        path = run/'predictions'/(sid+'.npz')
        assert sha(path) == rec['mask_sha256']
        with np.load(path, allow_pickle=False) as z:
            assert set(z.files) == set(methods)
            assert np.array_equal(z['full'][:3], z['no_h4'][:3])
            for method in methods:
                b = z[method]
                assert b.dtype == bool and b.shape == (4,*row['hw'])
                pred = torch.from_numpy(b).cuda()
                count = torch.stack([(pred & gt).sum((1,2),dtype=torch.int64),
                                     (pred & ~gt).sum((1,2),dtype=torch.int64),
                                     (~pred & gt).sum((1,2),dtype=torch.int64)],dim=1).cpu().numpy()
                assert count.tolist() == rec['counts'][method], (sid,method)
                totals[method] += count
                del pred
        del gt
        print(json.dumps({'verified':len(checks)//3,'expected_images':27,'sid':sid}),flush=True)
torch.cuda.synchronize()
assert len(checks) == 81
report = {'status':'PASS','images':27,'probability_files':81,'methods':methods,
          'cached_probabilities':checks,'independent_GPU_integer_counts':{k:v.tolist() for k,v in totals.items()},
          'seconds':time.monotonic()-started,'GPU':torch.cuda.get_device_name(0),'torch':torch.__version__,
          'new_training':False,'prediction_or_configuration_changes':False,
          'purpose':'Post-inference integrity audit; not an additional model experiment or performance benchmark'}
(ROOT/'evidence/remote_independent_integrity_audit.json').write_text(json.dumps(report,indent=2))
print(json.dumps({'status':'PASS','images':27,'probability_files':81,'seconds':report['seconds']}),flush=True)
