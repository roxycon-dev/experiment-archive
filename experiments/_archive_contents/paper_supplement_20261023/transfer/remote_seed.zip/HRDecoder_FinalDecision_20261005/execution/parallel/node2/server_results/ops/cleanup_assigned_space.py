"""Exact six retired probability directories; never scripts/logs/models/data."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from ops.remote import connect,command
from qlib.common import write_json,utc_now,read_json
directories=['/root/autodl-tmp/HRDecoder_Next_20261003/cold_storage/G/'+name for name in ('U1_native_whole','U1_native_slide','U1_long2048_whole')]
directories += ['/root/autodl-tmp/HRDecoder_Ready_20261002/cache/val/'+name for name in ('A0','H1','U1')]
ids=[row['sid'] for row in read_json(ROOT/'reports/data_identity.json')['splits']['val']['samples']]
code="""
from pathlib import Path
import shutil,json
directories=%r;expected=set(%r);deleted=[]
base=Path('/root/autodl-tmp').resolve()
for name in directories:
 d=Path(name);assert base in d.resolve().parents and not d.is_symlink()
 paths=sorted(d.glob('*.npz'));assert len(paths)==76 and {p.stem for p in paths}==expected
 for p in paths:assert p.suffix=='.npz' and p.parent==d and not p.is_symlink()
 for p in paths:
  deleted.append({'path':str(p),'bytes':p.stat().st_size,'reason':'Retired G/A0/H1/U1 probabilities outside final C/S/D/SD/P0/R protocol'})
  p.unlink()
print(json.dumps({'removed':deleted,'bytes':sum(row['bytes'] for row in deleted),'free_disk_bytes':shutil.disk_usage(base).free,
 'scripts_logs_reports_models_DDR_P0_champion_U0_preserved':True}))
"""%(directories,ids)
with connect() as client:result=json.loads(command(client,"/root/miniconda3/bin/python - <<'PY'\n"+code+"\nPY",60))
result['time']=utc_now();write_json(ROOT/'reports/assigned_space_cleanup.json',result);print(json.dumps({k:v for k,v in result.items() if k!='removed'}))
