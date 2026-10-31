"""Produce a portable code/data/weights package without runs or SSH credentials."""
import argparse
import hashlib
import json
import tarfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True,type=Path)
    a=ap.parse_args()
    selected=[]
    for directory in ['trainer','tools','configs','data_contract','campaign','tests','reports']:
        for p in (ROOT/directory).rglob('*'):
            rel=p.relative_to(ROOT)
            if p.is_file() and not any(x.startswith('_tmp') or x=='__pycache__' or x=='resolved' for x in rel.parts) and p.suffix!='.pyc':
                selected.append(p)
    selected += [ROOT/'train.py']
    for directory in ['assets/external','assets/maples_labels','vendor/HRDecoder']:
        selected += [p for p in (ROOT/directory).rglob('*') if p.is_file() and p.suffix!='.pyc' and '__pycache__' not in p.parts]
    for rel in ['assets/convnext_small_1k_224_ema.pth','assets/hrnetv2_w48-d2186c55.pth',
                'vendor/HACDR-Net/mmseg/models/backbones/hacdr.py','vendor/HACDR-Net/HACDRNet&DDHANet/HACDR_ddr.py','vendor/HACDR-Net/README.md']:
        selected.append(ROOT/rel)
    if (ROOT/'README.md').exists():selected.append(ROOT/'README.md')
    with tarfile.open(a.out,'w:gz',compresslevel=1) as tar:
        for p in sorted(set(selected)):
            tar.add(p,arcname='HRDecoder_Ready_20261002/'+p.relative_to(ROOT).as_posix(),recursive=False)
    h=hashlib.sha256()
    with a.out.open('rb') as f:
        for c in iter(lambda:f.read(1024*1024),b''):h.update(c)
    result=dict(path=str(a.out),sha256=h.hexdigest(),size=a.out.stat().st_size,files=len(set(selected)))
    (ROOT/'reports/package.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result))

if __name__=='__main__':main()
