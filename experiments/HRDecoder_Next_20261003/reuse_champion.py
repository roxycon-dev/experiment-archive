"""Reuse existing server champion files by exact SHA, without network or changing originals."""
import argparse
import shutil
from pathlib import Path
from nextlib.common import ROOT,read_json,sha


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--source-dir',required=True);a=ap.parse_args()
    source=Path(a.source_dir).resolve();specs=read_json(ROOT/'configs/assets.json')
    for name in ('D','B','E','E4','H4'):
        spec=specs[name];old=source/spec['file'];new=ROOT/'assets'/spec['file']
        if sha(old)!=spec['sha256']:raise ValueError(f'Old champion identity differs {name}')
        new.parent.mkdir(parents=True,exist_ok=True)
        if new.exists():
            if sha(new)!=spec['sha256']:raise ValueError('Existing prepared champion has wrong SHA')
        else:
            shutil.copy2(old,new)
        print(f'{name}: exact verified reuse, source kept unchanged')
