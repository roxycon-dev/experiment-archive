set -eu
BASE=/root/autodl-tmp/nativepyramid_sp_v3_20261016
/root/miniconda3/envs/hrdecoder_q_20261003/bin/python - <<'PY'
from pathlib import Path
import hashlib,tarfile,json
base=Path('/root/autodl-tmp/nativepyramid_sp_v3_20261016')
archive=base/'packages/data.tar'
assert archive.stat().st_size==684257280,'Data transfer is not complete'
asset=base/'assets/hrnetv2_w48-d2186c55.pth'
h=hashlib.sha256()
with asset.open('rb') as f:
    for block in iter(lambda:f.read(4*1024**2),b''):h.update(block)
assert h.hexdigest()=='d2186c55782308392ae14cf179ac4840143b09abdbba81aa3f9274c795ebc960','Weight transfer hash mismatch'
dest=base/'data'
with tarfile.open(archive) as t:
    members=t.getmembers()
    assert len(members)==1514
    assert all(m.isfile() and (dest/m.name).resolve().is_relative_to(dest.resolve()) for m in members)
    t.extractall(dest,filter='data')
receipt={'copied_image_label_files':len(members),'pretrained_sha256':h.hexdigest(),
         'test_pixels_decoded':False,'dataset_hash_audit':'train/val will be verified by preflight; test only by locked evaluator'}
(base/'receipts/assets_ready.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
PY
