"""Exact1514-file SHA acceptance using only the system Python standard library."""
import hashlib
import json
from pathlib import Path
import tarfile

ROOT=Path('/home/studentuser001/paper_supplement_20261023')
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
archive=ROOT/'assets/ddr_split613.tar'
assert sha(archive)=='2897951584cf7b8a968f2f6fe8626c5f5ea0e3221234bc2473739f8db4b823e6'
with tarfile.open(archive) as z:
    for m in z.getmembers():
        (ROOT/m.name).resolve().relative_to((ROOT/'data/DDR_split_613').resolve())
        assert m.isfile() or m.isdir()
    z.extractall(ROOT)
manifest=json.loads((ROOT/'baseline_v1/data_identity.json').read_text())
counts={}
for split in ('train','val','test'):
    rows=manifest['splits'][split]['samples']
    for row in rows:
        assert sha(ROOT/'data/DDR_split_613/images'/split/(row['sid']+'.jpg'))==row['image_sha256']
        assert sha(ROOT/'data/DDR_split_613/labels'/split/(row['sid']+'.png'))==row['label_sha256']
    counts[split]=len(rows)
assert counts=={'train':454,'val':76,'test':227}
(ROOT/'logs/data_acceptance.json').write_text(json.dumps({'status':'PASS','image_and_label_files':1514,'splits':counts},indent=2))
print('BASELINE_DATA_ACCEPTANCE_PASS',flush=True)
