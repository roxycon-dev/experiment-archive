"""Keep analyzed nonwinners across shutdown without delaying critical local backup."""
from pathlib import Path
import json,sys,shutil,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from nextlib.common import read_json,write_json,sha

def main():
 names=['U1_native_whole','U1_native_slide','U1_long2048_whole']
 g=ROOT/'work/G';cold=ROOT/'cold_storage/G';cold.mkdir(parents=True,exist_ok=True)
 assert (ROOT/'reports/G_analysis_complete.json').exists()
 best=read_json(ROOT/'reports/G_deployment_candidates.json')['best']['key']
 assert best not in names and all('highres' not in x and 'champion' not in x for x in names)
 sources=[g/name for name in names]
 required=sum(p.stat().st_size for s in sources if not s.is_symlink() for p in s.rglob('*') if p.is_file())
 assert shutil.disk_usage(cold).free >= required+1024**3, 'Insufficient persistent storage: retain original arrays and complete normal local backup'
 results=[]
 for source in sources:
  destination=cold/source.name
  if source.is_symlink():
   assert source.resolve()==destination.resolve()
   assert len(list(destination.glob('*.npz')))==76
   results.append(dict(cache=source.name,already_preserved=True));continue
  assert source.resolve().parent==g.resolve()
  assert g.resolve().as_posix().startswith('/dev/shm/hrdecoder_next_20261003/work/')
  assert not destination.exists()
  paths=[p for p in source.rglob('*') if p.is_file()]
  assert len(list(source.glob('*.npz')))==76
  before={p.relative_to(source).as_posix():dict(sha256=sha(p),bytes=p.stat().st_size) for p in paths}
  partial=destination.with_name(destination.name+'.copying')
  assert not partial.exists()
  shutil.copytree(source,partial)
  assert all(sha(partial/rel)==row['sha256'] and (partial/rel).stat().st_size==row['bytes'] for rel,row in before.items())
  partial.rename(destination)
  previous=source.with_name(source.name+'.verified_moved')
  assert not previous.exists()
  source.rename(previous);source.symlink_to(destination,target_is_directory=True)
  assert source.resolve()==destination.resolve()
  # Only the duplicate new RAM copy is removed after exact persistent-copy validation.
  assert previous.resolve().parent==g.resolve() and not previous.is_symlink()
  shutil.rmtree(previous)
  results.append(dict(cache=source.name,destination=str(destination),files=before,
      bytes=sum(x['bytes'] for x in before.values()),status='PERSISTENT_SERVER_SHA_VERIFIED'))
  write_json(ROOT/'reports/persistent_nonwinner_G.json',dict(status='IN_PROGRESS',time=time.time(),caches=results,
      no_unbacked_array_lost=True,no_model_checkpoint_removed=True))
 write_json(ROOT/'reports/persistent_nonwinner_G.json',dict(status='PASS',time=time.time(),caches=results,
     no_unbacked_array_lost=True,no_model_checkpoint_removed=True,
     purpose='These analyzed negative arrays remain on persistent data disk across instance shutdown; all required deployment components, M best arrays, node reports and model weights still require local SHA backup'))
 print('PERSISTENT_NONWINNER_G_PASS',flush=True)
if __name__=='__main__':main()
