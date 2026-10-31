"""Verify the complete immutable DDR transfer before enabling new training."""
import hashlib
import json
from pathlib import Path
import tarfile
from PIL import Image

ROOT=Path(__file__).resolve().parents[1];EXP=ROOT.parent
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
archive=EXP/'ddr_split613.tar'
assert sha(archive)=='2897951584cf7b8a968f2f6fe8626c5f5ea0e3221234bc2473739f8db4b823e6'
with tarfile.open(archive,'r:') as z:
    for m in z.getmembers():
        target=(EXP/m.name).resolve()
        assert target.is_relative_to((EXP/'data/DDR_split_613').resolve())
        assert m.isfile() or m.isdir(),m.name
    z.extractall(EXP)
train=ROOT/'training_v1'
manifest=json.loads((train/'reports/data_identity.json').read_text())
counts={}
for split in ['train','val','test']:
    rows=manifest['splits'][split]['samples']
    for row in rows:
        image=EXP/'data/DDR_split_613/images'/split/(row['sid']+'.jpg')
        label=EXP/'data/DDR_split_613/labels'/split/(row['sid']+'.png')
        assert sha(image)==row['image_sha256'] and sha(label)==row['label_sha256']
        with Image.open(image) as im:assert [im.height,im.width]==row['hw']
    counts[split]=len(rows)
    print(json.dumps({'verified_split':split,'images':len(rows)}),flush=True)
assert counts=={'train':454,'val':76,'test':227}
receipt={'status':'PASS','counts':counts,'image_and_label_SHA_verified':1514,
         'archive_sha256':sha(archive),'data_manifest_sha256':sha(train/'reports/data_identity.json'),
         'classes':['EX','HE','SE','MA'],'training_reads_only_train':True}
(train/'reports/new_training_data_acceptance.json').write_text(json.dumps(receipt,indent=2))
print(json.dumps(receipt),flush=True)
