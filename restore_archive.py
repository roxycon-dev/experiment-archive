"""Download and restore archived experiments, checking every SHA-256."""
from pathlib import Path, PurePosixPath
import argparse, hashlib, json, shutil, urllib.request, zipfile

BASE=Path(__file__).resolve().parent
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def safe(base,name):
 p=PurePosixPath(name)
 if p.is_absolute() or '..' in p.parts or ':' in name:raise ValueError('Unsafe archive path')
 out=(base/Path(*p.parts)).resolve()
 if not out.is_relative_to(base.resolve()):raise ValueError('Path leaves destination')
 return out
def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--phase',choices=['data','weights'],required=True)
 parser.add_argument('--output',type=Path,default=Path('restored'))
 args=parser.parse_args();out=args.output.resolve();cache=BASE/'.download-cache';cache.mkdir(exist_ok=True)
 manifest=json.loads((BASE/'archive_manifest.json').read_text(encoding='utf-8'))
 assets=json.loads((BASE/'release_assets.json').read_text(encoding='utf-8'))
 index={a['name']:a for a in assets if a['phase']==args.phase}
 object_cache=cache/'objects';object_cache.mkdir(exist_ok=True)
 for name,asset in index.items():
  local=cache/name
  if not local.exists() or sha(local)!=asset['sha256']:
   tag=manifest['data_release'] if args.phase=='data' else manifest['weights_release']
   url=f'https://github.com/{manifest["repository"]}/releases/download/{tag}/{name}'
   temp=local.with_suffix(local.suffix+'.partial')
   print('Downloading',name,flush=True)
   with urllib.request.urlopen(url,timeout=120) as response,temp.open('wb') as f:shutil.copyfileobj(response,f,1024*1024)
   if temp.stat().st_size!=asset['bytes'] or sha(temp)!=asset['sha256']:raise ValueError('Asset checksum failed: '+name)
   temp.replace(local)
  if name.endswith('.zip'):
   with zipfile.ZipFile(local) as z:
    for member in z.namelist():
     target=safe(cache,member);target.parent.mkdir(parents=True,exist_ok=True)
     if not target.exists():
      with z.open(member) as src,target.open('wb') as dest:shutil.copyfileobj(src,dest,1024*1024)
 count=0
 for entry in manifest['files']:
  if entry['status']=='git_text' and args.phase=='data':
   source=safe(BASE/'experiments',entry['path'])
  elif entry['status']=='release_object' and entry['download']['asset'] in index:
   d=entry['download'];source=safe(cache,d['member']) if 'member' in d else cache/d['asset']
  else:continue
  if sha(source)!=entry['sha256']:raise ValueError('Object checksum failed: '+entry['path'])
  target=safe(out,entry['path']);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target);count+=1
 print(f'Restored and verified {count} paths to {out}')
if __name__=='__main__':main()
