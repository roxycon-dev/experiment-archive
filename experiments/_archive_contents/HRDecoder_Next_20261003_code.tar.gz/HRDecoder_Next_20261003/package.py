"""Prepare small code and self-contained asset archives; no network or server access."""
import argparse
import hashlib
import tarfile
from pathlib import Path
from nextlib.common import ROOT,sha,read_json,write_json,source_identity


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--code-only',action='store_true');a=ap.parse_args()
    code=ROOT.parent/'HRDecoder_Next_20261003_code.tar.gz'
    with tarfile.open(code,'w:gz',compresslevel=1) as tar:
        for p in sorted(ROOT.rglob('*')):
            if not p.is_file():continue
            relative=p.relative_to(ROOT)
            if any(part in ('assets','work','__pycache__','.git') for part in relative.parts):continue
            if p.suffix in ('.pyc','.pth','.tmp'):continue
            tar.add(p,arcname=str(Path(ROOT.name)/relative),recursive=False)
    result=dict(code=dict(path=str(code),bytes=code.stat().st_size,sha256=sha(code)),source=source_identity())
    if not a.code_only:
        archive=ROOT.parent/'HRDecoder_Next_20261003_assets.tar.gz'
        specs=read_json(ROOT/'configs/assets.json')
        with tarfile.open(archive,'w:gz',compresslevel=1) as tar:
            for name,spec in specs.items():
                path=ROOT/'assets'/spec['file']
                if sha(path)!=spec['sha256']:raise ValueError(f'Prepared asset changed: {name}')
                tar.add(path,arcname=str(Path(ROOT.name)/'assets'/path.name),recursive=False)
        result['assets']=dict(path=str(archive),bytes=archive.stat().st_size,sha256=sha(archive))
        wanted={spec['file']:spec['sha256'] for spec in specs.values()}
        seen=set()
        with tarfile.open(archive,'r|gz') as tar:
            for member in tar:
                filename=Path(member.name).name
                if filename not in wanted or filename in seen:raise ValueError('Asset archive inventory differs')
                h=hashlib.sha256()
                f=tar.extractfile(member)
                for chunk in iter(lambda:f.read(4*1024**2),b''):h.update(chunk)
                if h.hexdigest()!=wanted[filename]:raise ValueError('Archive content differs from asset SHA')
                seen.add(filename)
        if seen!=set(wanted):raise ValueError('Asset archive is incomplete')
        result['assets']['all_member_sha256_verified']=True
    write_json(ROOT/'reports/package_manifest.json',result)
    print({k:{key:value for key,value in v.items() if key!='source'} for k,v in result.items() if k!='source'})


if __name__=='__main__':main()
